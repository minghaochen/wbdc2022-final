import json
import random
import zipfile
from io import BytesIO
from functools import partial
import re

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from transformers import BertTokenizer

from config_double_model2 import parse_args

from util import setup_device, setup_seed, setup_logging
from category_id_map import category_id_to_lv2id

from emojiswitch import emojize, demojize
from harvesttext import HarvestText


def create_dataloaders(args, val_idx=0):
    dataset = MultiModalDataset(args, args.train_annotation, args.train_zip_feats)
    size = len(dataset)
    val_size = int(size * args.val_ratio)

    a, b, c, d, e = torch.utils.data.random_split(dataset, [val_size, val_size, val_size, val_size, val_size],
                                                               generator=torch.Generator().manual_seed(args.seed))

    if val_idx == 1:
        train_dataset = b + c + d + e
        val_dataset = a
    elif val_idx == 2:
        train_dataset = a + c + d + e
        val_dataset = b
    elif val_idx == 3:
        train_dataset = a + b + d + e
        val_dataset = c
    elif val_idx == 4:
        train_dataset = a + b + c + e
        val_dataset = d
    else:
        train_dataset = a + b + c + d
        val_dataset = e

    if args.num_workers > 0:
        dataloader_class = partial(DataLoader, pin_memory=True, num_workers=args.num_workers,
                                   prefetch_factor=args.prefetch)
    else:
        # single-thread reading does not support prefetch_factor arg
        dataloader_class = partial(DataLoader, pin_memory=True, num_workers=0)

    train_sampler = RandomSampler(train_dataset)
    val_sampler = SequentialSampler(val_dataset)
    train_dataloader = dataloader_class(train_dataset,
                                        batch_size=args.batch_size,
                                        sampler=train_sampler,
                                        drop_last=True)
    val_dataloader = dataloader_class(val_dataset,
                                      batch_size=args.val_batch_size,
                                      sampler=val_sampler,
                                      drop_last=False)
    return train_dataloader, val_dataloader


class MultiModalDataset(Dataset):
    """ A simple class that supports multi-modal inputs.
    For the visual features, this dataset class will read the pre-extracted
    features from the .npy files. For the title information, it
    uses the BERT tokenizer to tokenize. We simply ignore the ASR & OCR text in this implementation.
    Args:
        ann_path (str): annotation file path, with the '.json' suffix.
        zip_feats (str): visual feature zip file path.
        test_mode (bool): if it's for testing.
    """

    def __init__(self,
                 args,
                 ann_path: str,
                 zip_feats: str,
                 test_mode: bool = False):
        self.max_frame = args.max_frames
        self.bert_seq_length = args.bert_seq_length
        self.test_mode = test_mode

        self.zip_feat_path = zip_feats
        self.num_workers = args.num_workers
        if self.num_workers > 0:
            # lazy initialization for zip_handler to avoid multiprocessing-reading error
            self.handles = [None for _ in range(args.num_workers)]
        else:
            self.handles = zipfile.ZipFile(self.zip_feat_path, 'r')

        # load annotations
        with open(ann_path, 'r', encoding='utf8') as f:
            self.anns = json.load(f)
        if args.debuge == True:
            self.anns = self.anns[:1000]
            

        # initialize the text tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(args.bert_dir, use_fast=True, cache_dir=args.bert_cache,
                                                       add_special_tokens=False)

    def __len__(self) -> int:
        return len(self.anns)
    
    def clean_text(self, text):
        text = re.sub(r"(\s)+", r"\1", text)   # 合并正文中过多的空格
        return text

    def get_visual_feats(self, idx: int) -> tuple:
        # read data from zipfile
        vid = self.anns[idx]['id']
        if self.num_workers > 0:
            worker_id = torch.utils.data.get_worker_info().id
            if self.handles[worker_id] is None:
                self.handles[worker_id] = zipfile.ZipFile(self.zip_feat_path, 'r')
            handle = self.handles[worker_id]
        else:
            handle = self.handles
        raw_feats = np.load(BytesIO(handle.read(name=f'{vid}.npy')), allow_pickle=True)
        raw_feats = raw_feats.astype(np.float32)  # float16 to float32
        num_frames, feat_dim = raw_feats.shape

        feat = np.zeros((self.max_frame, feat_dim), dtype=np.float32)
        mask = np.ones((self.max_frame,), dtype=np.int32)
        if num_frames <= self.max_frame:
            feat[:num_frames] = raw_feats
            mask[num_frames:] = 0
        else:
            # if the number of frames exceeds the limitation, we need to sample
            # the frames.
            if self.test_mode:
                # uniformly sample when test mode is True
                step = num_frames // self.max_frame
                select_inds = list(range(0, num_frames, step))
                select_inds = select_inds[:self.max_frame]
            else:
                # randomly sample when test mode is False
                select_inds = list(range(num_frames))
                random.shuffle(select_inds)
                select_inds = select_inds[:self.max_frame]
                select_inds = sorted(select_inds)
            for i, j in enumerate(select_inds):
                feat[i] = raw_feats[j]
        feat = torch.FloatTensor(feat)
        mask = torch.LongTensor(mask)
        return feat, mask

    def tokenize_text(self, text: str) -> tuple:
        encoded_inputs = self.tokenizer(text, max_length=self.bert_seq_length, padding='max_length', truncation=True)
        input_ids = torch.LongTensor(encoded_inputs['input_ids'])
        mask = torch.LongTensor(encoded_inputs['attention_mask'])
        return input_ids, mask

    def tokenize_text2(self, title: str, ocr_text: str, asr_text: str) -> tuple:
        if len(title) >= 128:
            title = title[:64] + title[-64:]
        if len(ocr_text) >= 128:
            ocr_text = ocr_text[:64] + ocr_text[-64:]
        if len(asr_text) >= 128:
            asr_text = asr_text[:64] + asr_text[-64:]

        encoded_titles = self.tokenizer(title, max_length=128, padding='max_length', truncation=True)
        encoded_ocr = self.tokenizer(ocr_text, max_length=128, padding='max_length', truncation=True)
        encoded_asr = self.tokenizer(asr_text, max_length=128, padding='max_length', truncation=True)

        text_input_ids = torch.LongTensor(
            [self.tokenizer.cls_token_id] + encoded_titles['input_ids'] + [self.tokenizer.sep_token_id]
            + encoded_ocr['input_ids'] + [self.tokenizer.sep_token_id] + encoded_asr['input_ids']
            + [self.tokenizer.sep_token_id]
        )
        text_mask = torch.LongTensor(
            [1, ] + encoded_titles['attention_mask'] + [1, ] + encoded_ocr['attention_mask'] + [1, ]
            + encoded_asr['attention_mask'] + [1, ]
        )
        text_token_type_ids = torch.zeros_like(text_input_ids)
        return text_input_ids, text_mask, text_token_type_ids

    
    def tokenize_text3(self, tfidf: str, title: str, ocr_text: str, asr_text: str) -> tuple:
        if len(tfidf) >= 60:
            tfidf = tfidf[:30] + tfidf[-30:]
            
        if len(title) >= 68:
            title = title[:34] + title[-34:]
        if len(ocr_text) >= 128:
            ocr_text = ocr_text[:64] + ocr_text[-64:]
        if len(asr_text) >= 128:
            asr_text = asr_text[:64] + asr_text[-64:]
            
        encoded_tfidf = self.tokenizer(tfidf, max_length=60, padding='max_length', truncation=True)
        encoded_titles = self.tokenizer(title, max_length=68, padding='max_length', truncation=True)
        encoded_ocr = self.tokenizer(ocr_text, max_length=128, padding='max_length', truncation=True)
        encoded_asr = self.tokenizer(asr_text, max_length=128, padding='max_length', truncation=True)

        text_input_ids = torch.LongTensor(
            [self.tokenizer.cls_token_id] + 
            encoded_tfidf['input_ids'] + [self.tokenizer.sep_token_id] + 
            encoded_titles['input_ids'] + [self.tokenizer.sep_token_id] + 
            encoded_ocr['input_ids'] + [self.tokenizer.sep_token_id] + 
            encoded_asr['input_ids'] + [self.tokenizer.sep_token_id]
        )
        text_mask = torch.LongTensor(
            [1, ] + 
            encoded_tfidf['attention_mask'] + [1, ] + 
            encoded_titles['attention_mask'] + [1, ] + 
            encoded_ocr['attention_mask'] + [1, ] + 
            encoded_asr['attention_mask'] + [1, ]
        )
        text_token_type_ids = torch.zeros_like(text_input_ids)
        return text_input_ids, text_mask, text_token_type_ids

    def tokenize_img(self, idx: int) -> tuple:
        frame_input, frame_mask = self.get_visual_feats(idx)
        frame_token_type_ids = torch.ones_like(frame_mask)
        return frame_input, frame_mask, frame_token_type_ids

    def __getitem__(self, idx: int) -> dict:
        # Step 1, load visual features from zipfile.
        # frame_input, frame_mask = self.get_visual_feats(idx)
        frame_input, frame_mask, frame_token_type_ids = self.tokenize_img(idx)

        # Step 2, load title tokens
        # title_input, title_mask = self.tokenize_text(self.anns[idx]['title'])
        vid = self.anns[idx]['id']
        
        title, asr = self.anns[idx]['title'], self.anns[idx]['asr']
        ocr = sorted(self.anns[idx]['ocr'], key=lambda x: x['time'])
        ocr = ','.join([x['text'] for x in ocr])
        
        title = demojize(title, delimiters=("", ""), lang="zh")
        title = self.clean_text(title)
        
        ht = HarvestText()
        title_tfidf = ','.join(ht.extract_keywords(title, 10, method="jieba_tfidf"))
        asr_tfidf = ','.join(ht.extract_keywords(asr, 10, method="jieba_tfidf"))
        ocr_tfidf = ','.join(ht.extract_keywords(ocr, 10, method="jieba_tfidf"))
        tfidf = title_tfidf + asr_tfidf + ocr_tfidf
        
        text_input, text_mask, text_token_type_ids = self.tokenize_text3(tfidf, title, ocr, asr)

        # Step 3, summarize into a dictionary
        data = dict(
            frame_input=frame_input,
            frame_mask=frame_mask,
            frame_token_type_ids=frame_token_type_ids,
            text_input=text_input,
            text_mask=text_mask,
            text_token_type_ids=text_token_type_ids,
            vid=vid,
        )

        # Step 4, load label if not test mode
        if not self.test_mode:
            label = category_id_to_lv2id(self.anns[idx]['category_id'])
            data['label'] = torch.LongTensor([label])

        return data


