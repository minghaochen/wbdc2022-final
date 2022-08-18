import time
import logging
# import os
# os.environ['MASTER_ADDR'] = 'localhost'
# os.environ['MASTER_PORT'] = '5678'

import torch
from torch.utils.data import SequentialSampler, DataLoader

# 新增1:依赖
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


from config import parse_args
from data_helper.data_helper_skf_ddp import MultiModalDataset

from category_id_map import lv2id_to_category_id
from util import setup_device, setup_seed, setup_logging, setup_pretrain, build_optimizer
from util import SequentialDistributedSampler,distributed_concat

from models.singlestream_with_pretrain_model_large import SingleStream_model_pretrain

import numpy as np
from tqdm import tqdm


def inference(args):
    # 1. load data
    dataset = MultiModalDataset(args, args.test_annotation, args.test_zip_frames, test_mode=True)
    # size = len(dataset)
    # val_size = int(size * 0.8)
    # train_dataset, val_dataset = torch.utils.data.random_split(dataset, [size - val_size, val_size],
    #                                                            generator=torch.Generator().manual_seed(args.seed))
    # for i in range(5):
    #     val_dataset += val_dataset
    # dataset = val_dataset
    sampler = SequentialDistributedSampler(dataset, batch_size=args.test_batch_size)
    # sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=False, seed=args.seed)
    dataloader = DataLoader(dataset,
                            batch_size=args.test_batch_size,
                            sampler=sampler,
                            drop_last=False,
                            pin_memory=True,
                            num_workers=args.num_workers,
                            prefetch_factor=args.prefetch)
    

    '''
        K折交叉验证，取平均
        每一折最佳模型路径需要手动输入
    '''
    MODEL_DIR = [
        args.ckpt_file
    # 'saves/single_model1/model_fold_1_epoch_0_mean_f1_0.0877.bin', 
    # 'saves/single_model1/model_fold_2_epoch_0_mean_f1_0.0334.bin',
    # 'saves/single_model1/model_fold_5_epoch_0_mean_f1_0.0176.bin',
                ]
    
    start_time = time.time()
    pred_matrix = np.zeros((1, 200))

    print('-' * 40 + 'model loading...' + '-' * 40)

    for i in range(len(MODEL_DIR)):
        # load model
        model = SingleStream_model_pretrain(args)
        checkpoint = torch.load(MODEL_DIR[i], map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        
        model = model.to(args.device)
        model = DDP(model, device_ids=[args.local_rank], 
                    output_device=args.local_rank,
                   find_unused_parameters=True)
    
        model.eval()

        # 3. inference
        # temp = np.zeros((1, 200))
        with torch.no_grad():
            pred_matrix = []
            for batch in tqdm(dataloader):
                for k, v in batch.items():
                    v = v.to(args.local_rank)
                prediction = model(batch, inference=True)
                pred_matrix.append(prediction)
                # temp = np.vstack((temp, prediction.cpu().numpy()))
                # predictions[i].extend(pred_label_id.cpu().numpy())
            pred_matrix = distributed_concat(torch.cat(pred_matrix, dim=0), 
                                         len(sampler.dataset))
           
        # pred_matrix = np.vstack((pred_matrix, temp[1:, :]))
        
    # pred_matrix = pred_matrix[1:, :]
    # pred_mean = np.zeros((len(dataset), 200))

    print('-' * 40 + 'predicting...' + '-' * 40)
    # for j in range(pred_mean.shape[0]):
    #     for k in range(pred_mean.shape[1]):
    #         for i in range(len(MODEL_DIR)):
    #             pred_mean[j][k] += pred_matrix[i][j][k]
    #         pred_mean[j][k] /= len(MODEL_DIR)
            
    pred_mean = pred_matrix.cpu().numpy()
    
    predictions = []
    for row in range(pred_mean.shape[0]):
        predictions.append(np.argmax(pred_mean[row]))
    
    end_time = time.time()
    cost_time = (end_time - start_time)
    ## 保存概率文件，方便后续模型融合
    # np.save(f'{args.savedmodel_path}/single_no_pretrain_pred.npy', pred_mean)
    logging.info(f"total: {len(sampler.dataset)}, cost time: {int(cost_time)}, average: {int(len(sampler.dataset) / int(cost_time))} qps")

    print('-' * 40 + 'results saving...' + '-' * 40)
    
    
    # 4. dump results
    with open(args.test_output_csv, 'w') as f:
        for pred_label_id, ann in zip(predictions, dataset.anns):
            video_id = ann['id']
            category_id = lv2id_to_category_id(pred_label_id)
            f.write(f'{video_id},{category_id}\n')
    
if __name__ == '__main__':
    args = parse_args()
    setup_logging(args)
    logging.info("Training/evaluation parameters: %s", args)
    
    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(backend='nccl')
    args.device = torch.device("cuda", args.local_rank)
    inference(args)
