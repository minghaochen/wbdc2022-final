import json
import random
import zipfile
from io import BytesIO
from functools import partial
from sklearn.model_selection import StratifiedKFold

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from transformers import BertTokenizer, DataCollatorForWholeWordMask
import jionlp as jio
from category_id_map import category_id_to_lv2id
import re
from torchvision.transforms import Compose, Resize, CenterCrop, Normalize, ToTensor
import os
from PIL import Image
import cv2
# import torchvision.transforms as T


from timm.data import create_transform
from torchvision import transforms
from torchvision.transforms import InterpolationMode
def build_transform(is_train=True, TEST_CROP=True):
    resize_im = 224 > 32
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=224,
            is_training=True,
            color_jitter=0.4,
            auto_augment='rand-m9-mstd0.5-inc1',
            re_prob=0.25,
            re_mode='pixel',
            re_count=1,
            interpolation='bicubic',
        )
        print('using training transform')
        return transform

    t = []
    if resize_im:
        if TEST_CROP:
            t.append(
                # transforms.Resize(256),
                transforms.Resize(256, interpolation=InterpolationMode.BICUBIC),
                # to maintain same ratio w.r.t. 224 images
            )
            t.append(transforms.CenterCrop(224))
        else:
            t.append(
                transforms.Resize((224, 224),
                                  interpolation=InterpolationMode.BICUBIC)
            )

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    print('using test transform')
    return transforms.Compose(t)



# self.transform = Compose([
#             Resize(256),
#             CenterCrop(224),
#             ToTensor(),
#             Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#         ])



def create_dataloaders(args):
    # dataset = MultiModalDataset(args, args.train_annotation, args.train_zip_feats)
    # size = len(dataset)
    # val_size = int(size * args.val_ratio)
    # train_dataset, val_dataset = torch.utils.data.random_split(dataset, [size - val_size, val_size],
    #                                                            generator=torch.Generator().manual_seed(args.seed))
    # skf dataset
    train_dataset = MultiModalDataset(args, args.train_annotation, args.train_zip_frames, fold=args.fold, get_skf='train')
    val_dataset = MultiModalDataset(args, args.train_annotation, args.train_zip_frames, fold=args.fold, get_skf='valid')
    
    # for i in range(10):
    #     val_dataset += val_dataset
    # print(len(val_dataset))


    if args.num_workers > 0:
        dataloader_class = partial(DataLoader, pin_memory=True, num_workers=args.num_workers, prefetch_factor=args.prefetch)
    else:
        # single-thread reading does not support prefetch_factor arg
        dataloader_class = partial(DataLoader, pin_memory=True, num_workers=0)

    train_sampler = RandomSampler(train_dataset, generator=torch.Generator().manual_seed(args.seed))
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
                 zip_frame_dir: str,
                 test_mode: bool = False,
                 fold: int = 0,
                 get_skf: str = 'train'):
        self.max_frame = args.max_frames
        self.bert_seq_length = args.bert_seq_length
        self.test_mode = test_mode
        
        self.get_skf = get_skf

        # self.zip_feat_path = zip_feats
        self.zip_frame_dir = zip_frame_dir
        
        self.num_workers = args.num_workers
        
        with open(ann_path, 'r', encoding='utf8') as f:
            self.anns = json.load(f)
            
        # self.anns = self.anns[:1000]
        print('training data:', len(self.anns))


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
    
    def get_visual_frames(self, idx: int) -> tuple:
        # read data from zipfile
        vid = self.anns[idx]['id']
        zip_path = os.path.join(self.zip_frame_dir, f'{vid[-3:]}/{vid}.zip')
        handler = zipfile.ZipFile(zip_path, 'r')
        namelist = sorted(handler.namelist())

        num_frames = len(namelist)
        # frame = torch.zeros((self.max_frame, 3, 224, 224), dtype=torch.float32)
        frame = torch.zeros((self.max_frame, 3, 224, 224), dtype=torch.float16)
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
                # 前 max_frame
                select_inds = select_inds[:self.max_frame]
                select_inds = sorted(select_inds)
        for i, j in enumerate(select_inds):
            mask[i] = 1
            # print(namelist)
            # print(namelist[j])
            img_content = handler.read(namelist[j])
            # print(img_content)
            img = Image.open(BytesIO(img_content))
            #cv2
            # img = cv2.imdecode(BytesIO(img_content), cv2.IMREAD_COLOR)
            # img = Image.fromarray(cv2.imdecode(np.frombuffer(img_content, np.uint8), cv2.IMREAD_COLOR))
            # print(img.shape)
            img_tensor = self.transform(img)
            
            # img_tensor = self.transform(torch.zeros(3, 224, 224))
      
            frame[i] = img_tensor.half()
            # print(frame[i].dtype)
        
            
            # frame[i] = torch.zeros(3, 224, 224)
            
        return frame, mask


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
            
        # modal drop
        if self.get_skf == 'train':
            prob_drop = random.random()
            # print(prob_drop)
            if  prob_drop < 0.1:
                modal_select = random.randint(1,3)
                # print(modal_select)
                if modal_select == 1:
                    title = ""
                elif modal_select == 2:
                    ocr_text = ""
                else:
                    asr_text = ""

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

    def __getitem__(self, idx: int) -> dict:
        # Step 1, load visual features from zipfile.
        frame_input, frame_mask = self.get_visual_frames(idx)

        # Step 2, load title tokens
        # text_raw = self.anns[idx]['title'] + self.anns[idx]['asr']
        # title_input, title_mask = self.tokenize_text(self.anns[idx]['title'])
        # title_input, title_mask = self.tokenize_text(self.anns[idx]['raw_text'])
        # 改变输入
        title, asr = self.anns[idx]['title'], self.anns[idx]['asr']
        ocr = sorted(self.anns[idx]['ocr'], key=lambda x: x['time'])
        ocr = ','.join([t['text'] for t in ocr])
        # 过滤下
        asr = re.findall('[\u3002\uff1b\uff0c\uff1a\u201c\u201d\uff08\uff09\u3001\uff1f\u300a\u300b\u4e00-\u9fa5]',asr)
        asr = ''.join(asr)
        ocr = re.findall('[\u3002\uff1b\uff0c\uff1a\u201c\u201d\uff08\uff09\u3001\uff1f\u300a\u300b\u4e00-\u9fa5]',ocr)
        ocr = ''.join(ocr)
        title_input, title_mask, text_token_type_ids = self.tokenize_text2(title, ocr, asr)

        # Step 3, summarize into a dictionary
        data = dict(
            frame_input=frame_input,
            frame_mask=frame_mask,
            title_input=title_input,
            title_mask=title_mask
        )

        # Step 4, load label if not test mode
        if not self.test_mode:
            label = category_id_to_lv2id(self.anns[idx]['category_id'])
            data['label'] = torch.LongTensor([label])
            data['label_aux'] = torch.LongTensor([int(self.anns[idx]['category_id'][0:2])])

#         # Step 5, MLM
#         mlm = self.mlm_collator([title_input])
#         data['input_ids_mlm'] = mlm['input_ids']
#         data['labels_mlm'] = mlm['labels']
#         # Step 6, ITM
#         idx = np.random.randint(0, high=len(self.anns))
#         frame_input_false, frame_mask_false = self.get_visual_feats(idx)
#         data['frame_input_false'] = frame_input_false
#         data['frame_mask_false'] = frame_mask_false

        return data
