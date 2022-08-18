import time
import logging

import torch
from torch.utils.data import SequentialSampler, DataLoader
from torch.cuda.amp import autocast as autocast, GradScaler
from apex import amp

from config import parse_args
from data_helper.data_helper_skf import MultiModalDataset

from category_id_map import lv2id_to_category_id
from util import setup_device, setup_seed, setup_logging, setup_pretrain, build_optimizer

from models.doublestream_model import MultiModal
from models.singlestream_model import SingleStream_model
from models.singlestream_with_pretrain_model import SingleStream_model_pretrain, SingleStream_model_pretrain2

import numpy as np
from tqdm import tqdm


def inference():
    args = parse_args()
    setup_logging(args)
    logging.info("Training/evaluation parameters: %s", args)
    # 1. load data
    load_time1 = time.time()
    dataset = MultiModalDataset(args, args.test_annotation, args.test_zip_frames, test_mode=True)
    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset,
                            batch_size=args.test_batch_size,
                            sampler=sampler,
                            drop_last=False,
                            pin_memory=True,
                            num_workers=args.num_workers,
                            prefetch_factor=args.prefetch)
    
    load_time2 = time.time()
    print('load data complete', load_time2-load_time1)
    
    # 2. load model    
    load_time1 = time.time()
    model = SingleStream_model_pretrain2(args)
    checkpoint = torch.load(args.ckpt_file, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # model.half()
    model = model.to('cuda')
    optimizer, scheduler = build_optimizer(args, model, args.warmup_steps, args.max_steps)
    model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
    
    if torch.cuda.is_available():
        model = torch.nn.parallel.DataParallel(model.cuda())
        model.eval()
    
    load_time2 = time.time()
    print('load model complete', load_time2-load_time1)
    
# 3. inference
    load_time1 = time.time()
    predictions = []
    with torch.no_grad():
        for batch in tqdm(dataloader):
        # for batch in dataloader:
            # with autocast():
            prediction, pred_label_id = model(batch, inference=True)
            predictions.extend(pred_label_id.cpu().numpy())
            # pred_matrix.append(prediction)
            # pred_label_id = np.argmax(prediction.cpu().numpy())
            # predictions.append(pred_label_id)
    # pred_mean = pred_matrix.cpu().numpy()
    # predictions = []
    # for row in range(pred_mean.shape[0]):
    #     predictions.append(np.argmax(pred_mean[row]))

    # 4. dump results
    with open(args.test_output_csv, 'w') as f:
        for pred_label_id, ann in zip(predictions, dataset.anns):
            video_id = ann['id']
            category_id = lv2id_to_category_id(pred_label_id)
            f.write(f'{video_id},{category_id}\n')
    load_time2 = time.time()
    
    print(f'inference complete {int(load_time2-load_time1)}s',
          f'dataset {len(dataset)}', 
          f'qps {int(len(dataset) / (load_time2-load_time1))}' ) 

if __name__ == '__main__':
    inference()
