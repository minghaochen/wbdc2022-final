import logging
import os
import time
import torch
import numpy as np
from config import parse_args
# from data_helper_pseudo import create_dataloaders
from data_helper_all import create_dataloaders

from model import MultiModal
# from model_pretrain import MultiModal
import gc
from util import setup_device, setup_seed, setup_logging, build_optimizer, evaluate
from torch_ema import ExponentialMovingAverage
from tqdm import tqdm

from apex import amp

class FGM:
    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self, epsilon=1., emb_name='word_embeddings'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
#                 print(name)
#                 print('fgm attack')
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0:
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, emb_name='word_embeddings'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
#                 print(name)
#                 print('fgm restore')
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


def validate(model, val_dataloader):
    model.eval()
    predictions = []
    raw_predictions = []
    labels = []
    losses = []
    with torch.no_grad():
        for batch in tqdm(val_dataloader):
            loss, _, pred_label_id, label, raw_pred = model(batch)
            loss = loss.mean()
            predictions.extend(pred_label_id.cpu().numpy())
            raw_predictions.extend(raw_pred.cpu().detach().numpy())
            labels.extend(label.cpu().numpy())
            losses.append(loss.cpu().numpy())
    loss = sum(losses) / len(losses)
    results = evaluate(predictions, labels)


    model.train()
    return loss, results, np.array(raw_predictions), np.array(labels)


def train_and_validate(args):
    # 1. load data
    train_dataloader, val_dataloader = create_dataloaders(args)
    del val_dataloader; gc.collect()

    # 2. build model and optimizers
    model = MultiModal(args).to(args.device)

    # load pretrain model
    # checkpoint = torch.load('save/v1/pretrain_model_epoch_4_loss_1.612.bin', map_location='cpu')
    
    # checkpoint = torch.load('save/pretrain_model_epoch_4.bin', map_location='cpu')
    # model.load_state_dict(checkpoint['model_state_dict'],strict=False)

    # args.max_steps = len(train_dataloader) * args.max_epochs
    
    fgm = FGM(model)  # 对抗训练

    optimizer, scheduler = build_optimizer(args, model)
    # ema = ExponentialMovingAverage(model.parameters(), decay=0.995)
    
    # 混合精度
    model, optimizer = amp.initialize(model, optimizer, opt_level="O1")


    if args.device == 'cuda':
        model = torch.nn.parallel.DataParallel(model.to(args.device))
    ema = ExponentialMovingAverage(model.parameters(), decay=0.999)
    # 3. training
    step = 0
    best_score = args.best_score
    start_time = time.time()
    num_total_steps = len(train_dataloader) * args.max_epochs
    early_stop_break = 0
    for epoch in range(args.max_epochs):
        if epoch >= 4: break
        save_step = 0
        for batch in tqdm(train_dataloader):
            model.train()

            loss, accuracy, _, _, _ = model(batch)

            loss = loss.mean()
            accuracy = accuracy.mean()
            # loss.backward()
            # 混合精度
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            
            
            # FGM
            fgm.attack()  
            adv_loss, _, _, _, _ = model(batch)
            adv_loss = adv_loss.mean()
            # adv_loss.backward()  
            # 混合精度
            with amp.scale_loss(adv_loss, optimizer) as scaled_adv_loss:
                scaled_adv_loss.backward()
            fgm.restore() 
            
            
            optimizer.step()

            ema.update()

            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

            step += 1
            if step % args.print_steps == 0:
                time_per_step = (time.time() - start_time) / max(1, step)
                remaining_time = time_per_step * (num_total_steps - step)
                remaining_time = time.strftime('%H:%M:%S', time.gmtime(remaining_time))
                logging.info(f"Epoch {epoch} step {step} eta {remaining_time}: loss {loss:.3f}, accuracy {accuracy:.3f}")
            
#             save_step += 1
#             if (best_score>=0.69) and (save_step % 501 == 0):
#                 ema.store()
#                 ema.copy_to()
#                 loss, results, raw_predictions, labels = validate(model, val_dataloader)
#                 results = {k: round(v, 4) for k, v in results.items()}
#                 logging.info(f"Epoch {epoch} step {step}: loss {loss:.3f}, {results}")
                
#                 mean_f1 = results['mean_f1']
#                 state_dict = model.module.state_dict() if args.device == 'cuda' else model.state_dict()
#                 torch.save({'epoch': epoch, 'model_state_dict': state_dict, 'mean_f1': mean_f1},
#                        f'{args.savedmodel_path}/model_fold_{args.fold}_epoch_{epoch}_mean_f1_{mean_f1}.bin')
                
#                 ema.restore()
            
            
            # break # for debug

        # 4. validation
        ema.store()
        ema.copy_to()
        
#         # 打印训练集的f1
#         infer_begin = time.time()
#         loss, results, raw_predictions, labels = validate(model, train_dataloader)
#         infer_end = time.time()
#         print('infer time:', infer_end-infer_begin)
#         print('len of val_dataloader', len(train_dataloader))
#         print('infer QPS:', len(train_dataloader)*args.batch_size/(infer_end-infer_begin))
        

#         results = {k: round(v, 4) for k, v in results.items()}
#         logging.info(f"Epoch {epoch} step {step}: loss {loss:.3f}, {results}")
        
        # 打印测试机的
        # infer_begin = time.time()
        # loss, results, raw_predictions, labels = validate(model, train_dataloader)
        # infer_end = time.time()
        # print('infer time:', infer_end-infer_begin)
        # print('len of train_dataloader', len(train_dataloader))
        # print('infer QPS:', len(train_dataloader)*args.val_batch_size/(infer_end-infer_begin))
        
        # results = {k: round(v, 4) for k, v in results.items()}
        # logging.info(f"Epoch {epoch} step {step}: loss {loss:.3f}, {results}")
        
        # mean_f1 = results['mean_f1']
        state_dict = model.module.state_dict() if args.device == 'cuda' else model.state_dict()
        torch.save({'epoch': epoch, 'model_state_dict': state_dict},
               f'{args.savedmodel_path}/model_fold_{args.fold}_epoch_{epoch}.bin')
        
        ema.restore()


def main():
    args = parse_args()
    setup_logging()
    setup_device(args)
    setup_seed(args)

    os.makedirs(args.savedmodel_path, exist_ok=True)
    logging.info("Training/evaluation parameters: %s", args)
    
    for fold in [1]:
        args.fold = fold
        train_and_validate(args)
        


if __name__ == '__main__':
    main()
