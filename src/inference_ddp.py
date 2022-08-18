import torch
import logging
from torch.utils.data import SequentialSampler, DataLoader

# 新增1:依赖
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from config import parse_args
from data_helper_ddp import MultiModalDataset
from category_id_map import lv2id_to_category_id
from model import MultiModal
import tqdm
import time
from functools import partial
import numpy as np

from apex import amp
from util import setup_device, setup_seed, setup_logging, build_optimizer, evaluate
from util import SequentialDistributedSampler,distributed_concat



def inference():
    args = parse_args()
    print(args)
    # 1. load data
    load_time1 = time.time()
    dataset = MultiModalDataset(args, args.test_annotation, args.test_zip_frames, test_mode=True, get_skf='valid')
    
   
    
    # sampler = SequentialSampler(dataset)
    sampler = SequentialDistributedSampler(dataset, batch_size=args.test_batch_size)
    # dataloader = DataLoader(dataset,
    #                         batch_size=args.test_batch_size,
    #                         sampler=sampler,
    #                         drop_last=False,
    #                         pin_memory=True,
    #                         num_workers=args.num_workers,
    #                         prefetch_factor=args.prefetch)
    dataloader_class = partial(DataLoader, pin_memory=True, num_workers=args.num_workers, prefetch_factor=args.prefetch)
    dataloader = dataloader_class(dataset,
                                  batch_size=args.test_batch_size,
                                  sampler=sampler,
                                  drop_last=False)
    load_time2 = time.time()
    print('load data complete', load_time2-load_time1)
        
    MODEL_DIR = [
    'saves/model_fold_soup_fold0.bin', 
    'saves/model_fold_soup_fold1.bin',
    'saves/model_fold_soup_pretrain1.bin',
    # 'saves/single_model1/model_fold_5_epoch_0_mean_f1_0.0176.bin',
                ]
    
    results = []
    load_time1 = time.time()
    for ckpt_file in MODEL_DIR:
        # 2. load model    
        
        model = MultiModal(args)
        checkpoint = torch.load(ckpt_file, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])

        model = model.half()
        model = model.to('cuda')
        
        model = DDP(model, device_ids=[args.local_rank], 
                    output_device=args.local_rank,
                   find_unused_parameters=True)
        # 混合精度
        optimizer, scheduler = build_optimizer(args, model)
        # model, optimizer = amp.initialize(model, optimizer, opt_level="O1")

        if torch.cuda.is_available():
            model = torch.nn.parallel.DataParallel(model.cuda())
            model.eval()
                
        # 3. inference
        predictions = []
        with torch.no_grad():
            for batch_id, batch in tqdm.tqdm(enumerate(dataloader)):
            # for batch in dataloader:
                for k, v in batch.items():
                    v = v.to(args.local_rank)
                pred_label_id, raw_prediction = model(batch, inference=True)
                predictions.extend(raw_prediction.cpu().numpy())
            predictions = distributed_concat(np.vstack(predictions), 
                                         len(sampler.dataset))
        # predictions = np.vstack(predictions)
        results.append(predictions)
        
    pred_ensemble = np.argmax(results[0] * 0.4 + results[1] * 0.4 + results[2] * 0.2, axis=1)
        
    # 4. dump results
    with open(args.test_output_csv, 'w') as f:
        for pred_label_id, ann in zip(pred_ensemble, dataset.anns):
            video_id = ann['id']
            category_id = lv2id_to_category_id(pred_label_id)
            f.write(f'{video_id},{category_id}\n')
    load_time2 = time.time()
    
    print('dataset', len(dataset), 'qps', int(len(dataset)/(load_time2-load_time1)))

if __name__ == '__main__':
    args = parse_args()
    setup_logging()
    logging.info("Training/evaluation parameters: %s", args)
    
    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(backend='nccl')
    args.device = torch.device("cuda", args.local_rank)
    
    inference()
