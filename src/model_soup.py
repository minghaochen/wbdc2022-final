
import logging
import os
import time
import torch
import numpy as np
from config import parse_args
from data_helper import create_dataloaders
from model import MultiModal
# from model_pretrain import MultiModal
import gc
from util import setup_device, setup_seed, setup_logging, build_optimizer, evaluate
from torch_ema import ExponentialMovingAverage
from tqdm import tqdm


def train_and_validate(args):
    # 1. load data
    train_dataloader, val_dataloader = create_dataloaders(args)

    # 2. build model and optimizers
    model = MultiModal(args)

    # load pretrain model
    # checkpoint = torch.load('save/v1/pretrain_model_epoch_4_loss_1.612.bin', map_location='cpu')
    
    checkpoint = torch.load('save/v3/model_fold_1_epoch_3_mean_f1_0.7041.bin', map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'],strict=False)
    
    path = [
        "save/v3/model_fold_1_epoch_2_mean_f1_0.7046.bin",
        "save/v3/model_fold_1_epoch_2_mean_f1_0.7017.bin",
        "save/v3/model_fold_1_epoch_3_mean_f1_0.7041.bin",
        # "save/v1/model_fold_0_epoch_2_mean_f1_0.7101.bin",
        # "save/v1/model_fold_0_epoch_2_mean_f1_0.7064.bin"
    ]
    model_dict = model.state_dict()
    soups = {key:[] for key in model_dict}
    for i, model_path in enumerate(path):
        weight = torch.load(model_path, map_location = 'cpu')['model_state_dict']
        weight_dict = weight.state_dict() if hasattr(weight, "state_dict") else weight
        for k, v in weight_dict.items():
            soups[k].append(v)
    if 0 < len(soups):
        soups = {k:(torch.sum(torch.stack(v), axis = 0) / len(v)).type(v[0].dtype) for k, v in soups.items() if len(v) != 0}
        model_dict.update(soups)
        model.load_state_dict(model_dict)
    
    
    state_dict = model.state_dict()
        
    
    torch.save({'epoch': 0, 'model_state_dict': state_dict, 'mean_f1': 0},
                f'{args.savedmodel_path}/model_fold_soup_pretrain1.bin')
            
                
        
            
def main():
    args = parse_args()
    setup_logging()
    setup_device(args)
    setup_seed(args)
    args.fold = 0

    train_and_validate(args)
        


if __name__ == '__main__':
    main()

    



