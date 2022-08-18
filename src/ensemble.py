import torch
from torch.utils.data import SequentialSampler, DataLoader
import numpy as np
from config import parse_args
from data_helper import MultiModalDataset
from category_id_map import lv2id_to_category_id
from model import MultiModal


def inference():
    args = parse_args()
    print(args)
    # 1. load data
    dataset = MultiModalDataset(args, '../../data/annotations/test_b.json', '../../data/zip_feats/test_b.zip', test_mode=True)
    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset,
                            batch_size=args.test_batch_size,
                            sampler=sampler,
                            drop_last=False,
                            pin_memory=True,
                            num_workers=args.num_workers,
                            prefetch_factor=args.prefetch)

    # 2. load model
    predictions1 = np.load('raw_predictions_1.npy')
    predictions2 = np.load('../method2/raw_predictions_2.npy')
    print('predictions1 shape', predictions1.shape)
    print('predictions2 shape', predictions2.shape)
    
    predictions = 0.4*predictions1 + 0.6*predictions2
    print('final predictions shape', predictions.shape)
    predictions = np.argmax(predictions, axis=1).tolist()


    # 4. dump results
    with open('../../data/result.csv', 'w') as f:
        for pred_label_id, ann in zip(predictions, dataset.anns):
            video_id = ann['id']
            category_id = lv2id_to_category_id(pred_label_id)
            f.write(f'{video_id},{category_id}\n')


if __name__ == '__main__':
    inference()