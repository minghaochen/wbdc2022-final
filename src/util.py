import logging
import random
import os
import numpy as np
import math
from sklearn.metrics import f1_score, accuracy_score
import torch
from transformers import AdamW, get_linear_schedule_with_warmup

from category_id_map import lv2id_to_lv1id
import time

def setup_device(args):
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.n_gpu = torch.cuda.device_count()


def setup_seed(args):
    random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt='%H:%M:%S',
        handlers=[
            logging.FileHandler(f"./logs/log_{time.strftime('%m%d_%H%M', time.localtime())}.log"),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)

    return logger

# def setup_logging():
#     logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',
#                         datefmt='%Y-%m-%d %H:%M:%S',
#                         level=logging.INFO)
#     logger = logging.getLogger(__name__)

#     return logger


def build_optimizer(args, model):
    # Prepare optimizer and schedule (linear warmup and decay)
    # 预训练不分层
    # no_decay = ['bias', 'LayerNorm.weight']
    # optimizer_grouped_parameters = [
    #     {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
    #      'weight_decay': args.weight_decay},
    #     {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    # ]
    # 分层
    params = list(model.named_parameters())
    no_decay = ['bias,', 'LayerNorm']
#     params_list = [n for n,p in list(model.named_parameters())]
    other = ['nextvlad', 'enhance', 'fusion', 'classifier', 'classifier2', 'frame_map','cross_modal_text_transform',
             'cross_modal_image_transform', 'token_type_embeddings', 'cross_modal_image_layers',
             'cross_modal_text_layers','cross_modal_image_pooler','cross_modal_text_pooler','mlm_score','itm_score', 'class_text', 'class_video', 'visual_backbone_proj','label_emb','class_text_level_1','class_video_level_1','final_class_level_1','label_emb_text', 'label_emb_video']
#     other = ['classifier']
    no_main = no_decay + other

    optimizer_grouped_parameters = [
        {'params': [p for n, p in params if not any(nd in n for nd in no_main)], 'weight_decay': args.weight_decay, 'lr': args.learning_rate},
        {'params': [p for n, p in params if not any(nd in n for nd in other) and any(nd in n for nd in no_decay)],
         'weight_decay': 0.0, 'lr': args.learning_rate},
        {'params': [p for n, p in params if any(nd in n for nd in other) and any(nd in n for nd in no_decay)],
         'weight_decay': 0.0, 'lr': args.learning_rate*5},
        {'params': [p for n, p in params if any(nd in n for nd in other) and not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay, 'lr': args.learning_rate*5},
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)

    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                num_training_steps=args.max_steps)
    return optimizer, scheduler


def evaluate(predictions, labels):
    # prediction and labels are all level-2 class ids

    lv1_preds = [lv2id_to_lv1id(lv2id) for lv2id in predictions]
    lv1_labels = [lv2id_to_lv1id(lv2id) for lv2id in labels]

    lv2_f1_micro = f1_score(labels, predictions, average='micro')
    lv2_f1_macro = f1_score(labels, predictions, average='macro')
    lv1_f1_micro = f1_score(lv1_labels, lv1_preds, average='micro')
    lv1_f1_macro = f1_score(lv1_labels, lv1_preds, average='macro')
    mean_f1 = (lv2_f1_macro + lv1_f1_macro + lv1_f1_micro + lv2_f1_micro) / 4.0

    eval_results = {'lv1_acc': accuracy_score(lv1_labels, lv1_preds),
                    'lv2_acc': accuracy_score(labels, predictions),
                    'lv1_f1_micro': lv1_f1_micro,
                    'lv1_f1_macro': lv1_f1_macro,
                    'lv2_f1_micro': lv2_f1_micro,
                    'lv2_f1_macro': lv2_f1_macro,
                    'mean_f1': mean_f1}

    return eval_results

class SequentialDistributedSampler(torch.utils.data.sampler.Sampler):
    """
    Distributed Sampler that subsamples indicies sequentially,
    making it easier to collate all results at the end.
    Even though we only use this sampler for eval and predict (no training),
    which means that the model params won't have to be synced (i.e. will not hang
    for synchronization even if varied number of forward passes), we still add extra
    samples to the sampler to make it evenly divisible (like in `DistributedSampler`)
    to make it easy to `gather` or `reduce` resulting tensors at the end of the loop.
    """
    def __init__(self, dataset, batch_size, rank=None, num_replicas=None):
        if num_replicas is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = torch.distributed.get_world_size()
        if rank is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = torch.distributed.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.batch_size = batch_size
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.batch_size / self.num_replicas)) * self.batch_size
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        # add extra samples to make it evenly divisible
        indices += [indices[-1]] * (self.total_size - len(indices))
        # subsample
        indices = indices[self.rank * self.num_samples : (self.rank + 1) * self.num_samples]
        return iter(indices)

    def __len__(self):
        return self.num_samples
    
def distributed_concat(tensor, num_total_examples):
    output_tensors = [tensor.clone() for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(output_tensors, tensor)
    concat = torch.cat(output_tensors, dim=0)
    # truncate the dummy elements added by SequentialDistributedSampler
    return concat[:num_total_examples]
