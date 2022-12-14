import os, math, random, time, sys, gc,  sys, json, psutil
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import numpy as np
import logging

from config import parse_args
from data_helper.data_helper import create_dataloaders

from util import setup_device, setup_seed, setup_logging, setup_pretrain, build_optimizer
from models.qq_uni_model import QQUniModel
from tqdm import tqdm

import warnings
warnings.filterwarnings('ignore')

def validate(model, data_loader, compute_loss=True, eval_max_num=99999):
    
    model.eval()
    mlm_loss_l, itm_loss_l, loss_l, vid_l = [], [], [], []
    
    with torch.no_grad():
        for batch_num, item in enumerate(data_loader):
            pred, loss, mlm_loss, itm_loss, hidden_states = model(item, task=None)
            
            loss = loss.mean()
            mlm_loss = mlm_loss.mean()
            itm_loss = itm_loss.mean()
            # itm_loss = -1.0
            # mlm_loss = -1.0
            
            if loss is not None:
                loss_l.append(loss.to("cpu"))
                mlm_loss_l.append(mlm_loss.to("cpu"))
                itm_loss_l.append(itm_loss.to("cpu"))
                # itm_loss_l.append(itm_loss)
                # mlm_loss_l.append(mlm_loss)
                            
            vid_l.append(item['vid'])
            
    return np.mean(loss_l), np.mean(mlm_loss_l), np.mean(itm_loss_l), np.concatenate(vid_l)

def train_and_validate(args, fold_idx):
    fold_idx += 1
    # 1. load data and save yaml
    train_dataloader, val_dataloader = create_dataloaders(args, fold_idx)
        
     # 2. build model and optimizers
    model = QQUniModel(args, args.bert_dir, task=args.pretrain_task)
    
    num_total_steps = len(train_dataloader) * args.max_epochs
    warmupsteps = int(args.warmup_ratio * num_total_steps)
    optimizer, scheduler = build_optimizer(args, model, warmupsteps, num_total_steps)
    
    if args.device == 'cuda':
        model = torch.nn.parallel.DataParallel(model.to(args.device))
        
    # 3. training
    start = time.time()
    
    logging.info('-' * 40 + f'[Fold_{fold_idx} PreTraining]' + '-' * 40)
    
    best_val_loss, step = None, 0
    for epoch in range(args.max_epochs):
        for batch in tqdm(train_dataloader, desc="Training data ..."):
            model.train()
            optimizer.zero_grad()
            
            pred, loss, mlm_loss, itm_loss, hidden_states = model(batch)
            # pred, loss, hidden_states = model(batch)
            loss = loss.mean()
            itm_loss = itm_loss.mean()
            mlm_loss = mlm_loss.mean()
            # itm_loss = -1.0
            # mlm_loss = -1.0
            
            loss.backward()
            
            optimizer.step()
            if scheduler:
                scheduler.step()
                
            step += 1
            if (step % 400 == 0 and step > 0):
                elapsed_seconds = time.time() - start# Evaluate the model on val_loader.
                logging.info(f"Epoch={epoch + 1}/{args.max_epochs}|step={step:3}|loss={loss:6.4}, mlm_loss={mlm_loss:6.4}, itm_loss={itm_loss:6.4}|time={elapsed_seconds:0.3}s")
                start = time.time()
                
            if (step % int(len(train_dataloader)) == 0 and step > 0):
                elapsed_seconds = time.time() - start# Evaluate the model on val_loader.
                val_loss, val_mlm_loss, val_itm_loss, vid_l = validate(model, val_dataloader)
                
                improve_str = ''
                if not best_val_loss or val_loss < best_val_loss:
                    best_val_loss = val_loss
                    model_state_dict = model.module.state_dict() if args.device == 'cuda' else model.state_dict()
                    torch.save(model_state_dict, args.pretrained_path)
                    improve_str = f"|New best_val_loss={best_val_loss:6.4}"
                logging.info(f"Epoch={epoch + 1}/{args.max_epochs}|step={step:3}|val_loss={val_loss:6.4}, val_mlm_loss={val_mlm_loss:6.4}, val_itm_loss={val_itm_loss:6.4}|time={elapsed_seconds:0.3}s" + improve_str)
                start = time.time()
            
            
    return best_val_loss

def main():
    args = parse_args()
    setup_logging(args)
    setup_device(args)
    setup_seed(args)
    setup_pretrain(args, task=['mlm', 'itm'])

    logging.info("Pretrain training/evaluation parameters: %s", args)

    for k in range(args.num_folds):
        if k > 0: continue ##?????????????????????
        train_and_validate(args, k)
        

if __name__ == '__main__':
    main()
