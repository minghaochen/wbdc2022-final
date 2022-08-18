import logging
import os
import time
import torch
from data_helper import MultiModalDataset

from config import parse_args
from data_helper import create_dataloaders
# from model import MultiModal
from model_pretrain import MultiModal
from torch.utils.data import SequentialSampler, DataLoader, RandomSampler
from util import setup_device, setup_seed, setup_logging, build_optimizer, evaluate
from torch_ema import ExponentialMovingAverage


def validate(model, val_dataloader):
    model.eval()
    predictions = []
    labels = []
    losses = []
    with torch.no_grad():
        for batch in val_dataloader:
            loss, _, pred_label_id, label = model(batch)
            loss = loss.mean()
            predictions.extend(pred_label_id.cpu().numpy())
            labels.extend(label.cpu().numpy())
            losses.append(loss.cpu().numpy())
    loss = sum(losses) / len(losses)
    results = evaluate(predictions, labels)

    model.train()
    return loss, results


def train_and_validate(args):
    # 1. load data
#     train_dataloader, val_dataloader = create_dataloaders(args)
    
    # 全量数据
    dataset = MultiModalDataset(args, '../../data/annotations/unlabeled.json', '../../data/zip_feats/unlabeled.zip', test_mode=True)
    sampler = RandomSampler(dataset)
#     sampler = SequentialSampler(dataset)
    train_dataloader = DataLoader(dataset,
                            batch_size=args.batch_size,
                            sampler=sampler,
                            drop_last=True,
                            pin_memory=True,
                            num_workers=args.num_workers,
                            prefetch_factor=args.prefetch)

    # 2. build model and optimizers
    model = MultiModal(args)

    # args.max_steps = len(train_dataloader) * args.max_epochs

    optimizer, scheduler = build_optimizer(args, model)
    # ema = ExponentialMovingAverage(model.parameters(), decay=0.995)

    if args.device == 'cuda':
        model = torch.nn.parallel.DataParallel(model.to(args.device))
    # ema = ExponentialMovingAverage(model.parameters(), decay=0.999)
    # 3. training
    step = 0
    best_score = args.best_score
    start_time = time.time()
    num_total_steps = len(train_dataloader) * args.max_epochs
    for epoch in range(args.max_epochs):

        for batch in train_dataloader:
            model.train()

            # loss, accuracy, _, _ = model(batch)
            ###########  pretrain  #############
            ############### MLM
            loss,mlm_loss,itm_loss = model(batch, do_mlm=True, do_itm=True)
            ############### ITM

#             pos_len = len(batch["frame_input"]) // 2
#             neg_len = len(batch["frame_input"]) - pos_len
#             itm_labels = torch.cat([torch.ones(pos_len), torch.zeros(neg_len)]).cuda()
#             itm_labels = itm_labels[torch.randperm(itm_labels.size(0))]
#             itm_images = torch.stack(
#                             [
#                                 batch["frame_input"][i] if itm_labels[i] == 1 else batch["frame_input_false"][i]
#                                 for i in range(len(itm_labels))
#                             ]
#                         ).cuda()
#             itm_masks = torch.stack(
#                             [
#                                 batch["frame_mask"][i] if itm_labels[i] == 1 else batch["frame_mask_false"][i]
#                                 for i in range(len(itm_labels))
#                             ]
#                         ).cuda()
#             batch['frame_input'] = itm_images
#             batch['frame_mask'] = itm_masks
#             itm_loss = model(batch, do_itm=True, itm_labels=itm_labels)
#             loss = mlm_loss + itm_loss
            
            
            
            loss = loss.mean()
            # accuracy = accuracy.mean()
            loss.backward()
            optimizer.step()

            # ema.update()

            optimizer.zero_grad()
            scheduler.step()

            step += 1
            if step % args.print_steps == 0:
                time_per_step = (time.time() - start_time) / max(1, step)
                remaining_time = time_per_step * (num_total_steps - step)
                remaining_time = time.strftime('%H:%M:%S', time.gmtime(remaining_time))
                logging.info(f"Epoch {epoch} step {step} eta {remaining_time}: loss {loss:.3f}, mlm_loss {mlm_loss:.3f}, itm_loss {itm_loss:.3f}")
            
            
#             break # for debug



        # 4. validation
        # ema.store()
        # ema.copy_to()
        # loss, results = validate(model, val_dataloader)
        # ema.restore()

        # results = {k: round(v, 4) for k, v in results.items()}
        # logging.info(f"Epoch {epoch} step {step}: loss {loss:.3f}, {results}")

        # 5. save checkpoint

        state_dict = model.module.state_dict() if args.device == 'cuda' else model.state_dict()
        torch.save({'epoch': epoch, 'model_state_dict': state_dict},
                   f'{args.savedmodel_path}/pretrain_model_epoch_{epoch}.bin')



def main():
    args = parse_args()
    setup_logging()
    setup_device(args)
    setup_seed(args)

    os.makedirs(args.savedmodel_path, exist_ok=True)
    logging.info("Training/evaluation parameters: %s", args)

    train_and_validate(args)


if __name__ == '__main__':
    main()
