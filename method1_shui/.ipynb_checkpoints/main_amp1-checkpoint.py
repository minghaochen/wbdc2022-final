import logging
import os
import time
import torch
import numpy as np
from torch.cuda.amp import autocast as autocast, GradScaler

from config1 import parse_args
from data_helper.data_helper_skf import create_dataloaders

from util import setup_device, setup_seed, setup_logging, setup_pretrain, build_optimizer, evaluate, build_optimizer2
from util import EMA, FGM

from models.doublestream_model import MultiModal
from models.singlestream_model import SingleStream_model
from models.singlestream_with_pretrain_model import SingleStream_model_pretrain, SingleStream_model_pretrain2


from tqdm import tqdm

import warnings
warnings.filterwarnings('ignore')

def validate(model, val_dataloader):
    model.eval()
    predictions = []
    labels = []
    losses = []
    
    logits = []
    with torch.no_grad():
        for batch in tqdm(val_dataloader, desc="Validating data..."):
            with autocast():
                loss, _, pred_label_id, label, logit = model(batch)
                loss = loss.mean()
                
            predictions.extend(pred_label_id.cpu().numpy())
            logits.extend(logit.cpu().numpy().tolist())
            labels.extend(label.cpu().numpy())
            losses.append(loss.cpu().numpy())
    loss = sum(losses) / len(losses)
    results = evaluate(predictions, labels)

    model.train()
    return loss, results, logits


# K折交叉验证
def train_and_validate(args, fold_idx):
    fold_idx += 1
    train_dataloader, val_dataloader = create_dataloaders(args, fold_idx)
    
    if args.model_type == 'double':
        model = MultiModal(args)
    elif args.model_type == 'single':
        model = SingleStream_model_pretrain(args)
    
    if args.ispretrain == 1:
        checkpoint = torch.load(args.pretrained_path)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        logging.info(f'success load pretrained model!')
    
        # ckp = torch.load('./save/single_model3/model_fold_1_epoch_3_mean_f1_0.6869.bin')
        # model.load_state_dict(ckp['model_state_dict'], strict=False)
        # model = SingleStream_model(args)
        
    num_total_steps = len(train_dataloader) * args.max_epochs
    warmupsteps = int(args.warmup_ratio * num_total_steps)
    # num_total_steps = args.max_steps
    # warmupsteps = args.warmup_steps
    logging.info(f'num_total_steps {num_total_steps} warmupsteps {warmupsteps}')
    logging.info(f'pretrain model {args.pretrained_path} learning rate {args.learning_rate}')
    logging.info(f'epoch {args.max_epochs} val rato {args.val_ratio} batch_size {args.batch_size}')
    
    optimizer, scheduler = build_optimizer2(args, model, warmupsteps, num_total_steps)

    ema = EMA(model, 0.999, device=args.device)  # EMA初始化
    ema.register()

    fgm = FGM(model)  # 对抗训练

    if args.device == 'cuda':
        model = torch.nn.parallel.DataParallel(model.to(args.device))

    # 3. training
    best_score = args.best_score
    start_time = time.time()
    
    train_loss, adv_loss, val_loss, train_acc = 0, 0, 0, 0

    logging.info('-' * 40 + f'[Fold_{fold_idx} Training]' + '-' * 40)

    best_val_logits = None
    step = 0
    
    scaler = GradScaler()
    
    for epoch in range(args.max_epochs):
        if epoch > 5:
            break
        for batch in tqdm(train_dataloader, desc="Training data ..."):
            model.train()
            with autocast():
                train_loss, train_acc, _, _, _ = model(batch)
                train_loss = train_loss.mean()
                train_acc = train_acc.mean()
            # train_loss.backward()
            scaler.scale(train_loss).backward()

            fgm.attack()  
            with autocast():
                adv_loss, _, _, _, _ = model(batch)
                adv_loss = adv_loss.mean()
            # adv_loss.backward()  
            scaler.scale(adv_loss).backward()  
            fgm.restore()  
            # adv_loss = -1
            
            scaler.step(optimizer)
            scaler.update()
            
            # optimizer.step()  # 梯度下降，更新参数      用amp则注释这行 
            ema.update()  # 更新EMA参数

            optimizer.zero_grad()
            scheduler.step()
            
            
            
            step += 1
            
            if step % args.print_steps == 0:
                logging.info(f"Train: Epoch {epoch} step {step} : train_loss {train_loss:.3f}, "
                 f"adv_loss {adv_loss:.3f}, train_acc {train_acc:.3f}")
                
            # # 4. train validation
            # if (step % len(train_dataloader) == 0):
            #     trn_loss, trn_results, trn_logits = validate(model, train_dataloader)
            #     trn_results = {k: round(v, 4) for k, v in trn_results.items()}
            #     logging.info(f"Train: Epoch {epoch} step {step}: trn_loss {trn_loss:.3f}, {trn_results}")
                
            if ((best_score>=0.68) and (step % 500 == 0)):                 
                ema.apply_shadow()#新增
                # 4. val validation
                val_loss, results, logits = validate(model, val_dataloader)
                results = {k: round(v, 4) for k, v in results.items()}
                logging.info(f"Validate: Epoch {epoch} step {step}: val_loss {val_loss:.3f}, {results}")

                # 5. save checkpoint
                mean_f1 = results['mean_f1']
                
                if mean_f1 > best_score:
                    best_score = mean_f1
                    best_val_logits = logits
                    
                state_dict = model.module.state_dict() if args.device == 'cuda' else model.state_dict()
                torch.save({'fold': fold_idx, 
                            'epoch': epoch, 'model_state_dict': state_dict, 
                            'optimizer':optimizer.state_dict(), 'scheduler':scheduler.state_dict(),
                            'mean_f1': mean_f1},
                           
                           f'{args.savedmodel_path}/model_fold_{fold_idx}_epoch_{epoch}_mean_f1_{mean_f1}.bin')
                ema.restore()#新增
            elif ((best_score < 0.68) and (step % len(train_dataloader) == 0)):
                ema.apply_shadow()#新增
                # 4. validation
                val_loss, results, logits = validate(model, val_dataloader)
                results = {k: round(v, 4) for k, v in results.items()}
                logging.info(f"Validate: Epoch {epoch} step {step}: val_loss {val_loss:.3f}, {results}")
                
                # 5. save checkpoint
                mean_f1 = results['mean_f1']
                
                if mean_f1 > best_score:
                    best_score = mean_f1
                    best_val_logits = logits
                    
                state_dict = model.module.state_dict() if args.device == 'cuda' else model.state_dict()
                torch.save({'fold': fold_idx, 
                            'epoch': epoch, 'model_state_dict': state_dict, 
                            'optimizer':optimizer.state_dict(), 'scheduler':scheduler.state_dict(),
                            'mean_f1': mean_f1},
                           f'{args.savedmodel_path}/model_fold_{fold_idx}_epoch_{epoch}_mean_f1_{mean_f1}.bin')
                ema.restore()#新增
    # best_val_logits 保存

def main():
    args = parse_args()
    setup_logging(args)
    setup_device(args)
    setup_seed(args)
    setup_pretrain(args)
    
    os.makedirs(args.savedmodel_path, exist_ok=True)
    logging.info("Training/evaluation parameters: %s", args)

    for k in range(args.num_folds):
        if k > 0: continue
        # train_and_validate(args, k)
        args.fold = k
        train_and_validate(args, k)

if __name__ == '__main__':
    main()
