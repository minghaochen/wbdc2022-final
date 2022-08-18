import os
import json
import zipfile
import random
import zipfile
import torch
import gc
import re

from PIL import Image
from io import BytesIO
from functools import partial
from transformers import BertTokenizer
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from torchvision.transforms import Compose, Resize, CenterCrop, Normalize, ToTensor

from category_id_map import category_id_to_lv2id

# from emojiswitch import emojize, demojize
# from harvesttext import HarvestText


def create_dataloaders(args, val_idx=0):
    dataset = MultiModalDataset(args, args.train_annotation, args.train_zip_frames)
    size = len(dataset)
    val_size = int(size * args.val_ratio)
    # val_size = int(size / args.num_folds)
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [size - val_size, val_size],
                                                               generator=torch.Generator().manual_seed(args.seed))
    
#     a, b, c, d, e = torch.utils.data.random_split(dataset, [val_size, val_size, val_size, val_size, val_size],
#                                                                generator=torch.Generator().manual_seed(args.seed))
        
#     if val_idx == 1:
#         train_dataset = b + c + d + e
#         val_dataset = a
#     elif val_idx == 2:
#         train_dataset = a + c + d + e
#         val_dataset = b
#     elif val_idx == 3:
#         train_dataset = a + b + d + e
#         val_dataset = c
#     elif val_idx == 4:
#         train_dataset = a + b + c + e
#         val_dataset = d
#     else:
#         train_dataset = a + b + c + d
#         val_dataset = e
        
        

    if args.num_workers > 0:
        dataloader_class = partial(DataLoader, pin_memory=True, num_workers=args.num_workers,
                                   prefetch_factor=args.prefetch)
    else:
        # single-thread reading does not support prefetch_factor arg
        dataloader_class = partial(DataLoader, pin_memory=True, num_workers=0)

    # train_sampler = RandomSampler(train_dataset)
    # val_sampler = SequentialSampler(val_dataset)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True, seed=args.seed)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False, seed=args.seed)
    
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
                 zip_frame_dir: str,
                 test_mode: bool = False):
        
        self.max_frame = args.max_frames
        self.bert_seq_length = args.bert_seq_length
        self.test_mode = test_mode

        self.zip_frame_dir = zip_frame_dir
        # load annotations
        with open(ann_path, 'r', encoding='utf8') as f:
            self.anns = json.load(f)
        if (args.debuge): 
            self.anns = self.anns[:10000]
            gc.collect()
            
        # initialize the text tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(args.bert_dir, use_fast=True, cache_dir=args.bert_cache)

        # we use the standard image transform as in the offifical Swin-Transformer.
        self.transform = Compose([
            Resize(256),
            CenterCrop(224),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self) -> int:
        return len(self.anns)
    
    def clean_text(self, text):
        text = re.sub(r"(\s)+", r"\1", text)   # 合并正文中过多的空格
        return text

    def get_visual_frames(self, idx: int) -> tuple:
        # read data from zipfile
        vid = self.anns[idx]['id']
        zip_path = os.path.join(self.zip_frame_dir, f'{vid[-3:]}/{vid}.zip')
        handler = zipfile.ZipFile(zip_path, 'r')
        namelist = sorted(handler.namelist())

        num_frames = len(namelist)
        frame = torch.zeros((self.max_frame, 3, 224, 224), dtype=torch.float32)
        mask = torch.zeros((self.max_frame, ), dtype=torch.long)
        if num_frames <= self.max_frame:
            # load all frame
            select_inds = list(range(num_frames))
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
            mask[i] = 1
            img_content = handler.read(namelist[j])
            img = Image.open(BytesIO(img_content))
            img_tensor = self.transform(img)
            frame[i] = img_tensor
        return frame, mask

    def tokenize_text(self, text: str) -> tuple:
        encoded_inputs = self.tokenizer(text, max_length=self.bert_seq_length, padding='max_length', truncation=True)
        input_ids = torch.LongTensor(encoded_inputs['input_ids'])
        mask = torch.LongTensor(encoded_inputs['attention_mask'])
        return input_ids, mask

    def tokenize_text2(self, title: str, ocr_text: str, asr_text: str) -> tuple:
        if len(title) >= 84:
            title = title[:42] + title[-42:]
        if len(ocr_text) >= 84:
            ocr_text = ocr_text[:42] + ocr_text[-42:]
        if len(asr_text) >= 84:
            asr_text = asr_text[:42] + asr_text[-42:]
        
        encoded_titles = self.tokenizer(title, max_length=84, padding='max_length', truncation=True)
        encoded_ocr = self.tokenizer(ocr_text, max_length=84, padding='max_length', truncation=True)
        encoded_asr = self.tokenizer(asr_text, max_length=84, padding='max_length', truncation=True)


        # encoded_titles = self.tokenizer(title, max_length=84, padding='max_length', truncation=True, add_special_tokens=False)
        # encoded_ocr = self.tokenizer(ocr_text, max_length=84, padding='max_length', truncation=True, add_special_tokens=False)
        # encoded_asr = self.tokenizer(asr_text, max_length=84, padding='max_length', truncation=True, add_special_tokens=False)
        
#         if len(encoded_titles['input_ids']) == 0:
#             encoded_titles['input_ids'] = [0] * 84
#             encoded_titles['attention_mask'] = [0] * 84
            
#         if len(encoded_ocr['input_ids']) == 0:
#             encoded_ocr['input_ids'] = [0] * 84
#             encoded_ocr['attention_mask'] = [0] * 84
            
#         if len(encoded_asr['input_ids']) == 0:
#             encoded_asr['input_ids'] = [0] * 84
#             encoded_asr['attention_mask'] = [0] * 84

        text_input_ids = torch.LongTensor(
            [self.tokenizer.cls_token_id]+ [self.tokenizer.cls_token_id, self.tokenizer.sep_token_id] + 
            [self.tokenizer.sep_token_id] +
            encoded_titles['input_ids'] + [self.tokenizer.sep_token_id] + 
            encoded_ocr['input_ids'] + [self.tokenizer.sep_token_id] + 
            encoded_asr['input_ids'] + [self.tokenizer.sep_token_id]
        )
        
        text_mask = torch.LongTensor(
            [1, ] + [1, 1] + [1, ] +
            encoded_titles['attention_mask'] + [1, ] + 
            encoded_ocr['attention_mask'] + [1, ] + 
            encoded_asr['attention_mask'] + [1, ]
        )
        text_token_type_ids = torch.zeros_like(text_input_ids)
        return text_input_ids, text_mask, text_token_type_ids

    # def tokenize_img(self, idx: int) -> tuple:
    #     frame_input, frame_mask = self.get_visual_feats(idx)
    #     frame_token_type_ids = torch.ones_like(frame_mask)
    #     return frame_input, frame_mask, frame_token_type_ids

    def __getitem__(self, idx: int) -> dict:
        # Step 1, load visual features from zipfile.
        frame_input, frame_mask = self.get_visual_frames(idx)
        frame_token_type_ids = torch.ones_like(frame_mask)

        # frame_input, frame_mask, frame_token_type_ids = self.tokenize_img(idx)

        # Step 2, load title tokens
        # title_input, title_mask = self.tokenize_text(self.anns[idx]['title'])
        vid = self.anns[idx]['id']
        title = self.anns[idx]['title']
        asr = self.anns[idx]['asr']
        ocr = sorted(self.anns[idx]['ocr'], key=lambda x: x['time'])
        ocr = ','.join([x['text'] for x in ocr])
        
        asr = re.findall('[\u3002\uff1b\uff0c\uff1a\u201c\u201d\uff08\uff09\u3001\uff1f\u300a\u300b\u4e00-\u9fa5]',asr)
        asr = ''.join(asr)
        ocr = re.findall('[\u3002\uff1b\uff0c\uff1a\u201c\u201d\uff08\uff09\u3001\uff1f\u300a\u300b\u4e00-\u9fa5]',ocr)
        ocr = ''.join(ocr)
        
        # if title == None:
        #     title = ''
        # if asr == None:
        #     asr = ''
        # if ocr == None:
        #     ocr = ''
        
        text_input, text_mask, text_token_type_ids = self.tokenize_text2(title, ocr, asr)

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
    
def build_transform(is_train, config):
    resize_im = True
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=config.DATA.IMG_SIZE,
            is_training=True,
            color_jitter=config.AUG.COLOR_JITTER if config.AUG.COLOR_JITTER > 0 else None,
            auto_augment=config.AUG.AUTO_AUGMENT if config.AUG.AUTO_AUGMENT != 'none' else None,
            re_prob=config.AUG.REPROB,
            re_mode=config.AUG.REMODE,
            re_count=config.AUG.RECOUNT,
            interpolation=config.DATA.INTERPOLATION,
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(config.DATA.IMG_SIZE, padding=4)
        return transform

    t = []
    if resize_im:
        if config.TEST.CROP:
            size = int((256 / 224) * config.DATA.IMG_SIZE)
            t.append(
                transforms.Resize(size, interpolation=_pil_interp(config.DATA.INTERPOLATION)),
                # to maintain same ratio w.r.t. 224 images
            )
            t.append(transforms.CenterCrop(config.DATA.IMG_SIZE))
        else:
            t.append(
                transforms.Resize((config.DATA.IMG_SIZE, config.DATA.IMG_SIZE),
                                  interpolation=_pil_interp(config.DATA.INTERPOLATION))
            )

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)