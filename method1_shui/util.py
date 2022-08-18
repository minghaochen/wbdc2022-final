import logging
import random
import time
import math

import numpy as np
from sklearn.metrics import f1_score, accuracy_score
import torch
from transformers import AdamW, get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup

from category_id_map import lv2id_to_lv1id

# import jieba
# jieba.setLogLevel(logging.INFO)


def setup_logging(args):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt='%H:%M:%S',
        handlers=[
            logging.FileHandler(f"./logs/log_{args.version}_{time.strftime('%m%d_%H%M', time.localtime())}.log"),
            logging.StreamHandler()
        ]
    )

    return logger


def setup_pretrain(args, task=['']):
    # args.model_config = {
    #                 #     'INPUT_SIZE': 1792,
    #                     'HIDDEN_SIZE': 256,
    #                     'NUM_CLASSES': 200,
    #                 #     'FEATURE_SIZE': 1536,
    #                 #     'OUTPUT_SIZE': 1024,
    #                     'EXPANSION_SIZE': 2,
    #                     'CLUSTER_SIZE': 64,
    #                     'NUM_GROUPS': 8,
    #                     'DROPOUT_PROB': 0.2,
    #                 }
    # args.bert_cfg_dict = {}
    # args.bert_cfg_dict['uni'] = {
    #         'hidden_size':768,
    #         'num_hidden_layers':6,
    #         'num_attention_heads':12,
    #         'intermediate_size':3072,
    #         'hidden_dropout_prob':0.0,
    #         'attention_probs_dropout_prob':0.0
    #     }
    args.pretrain_task = task
    


def setup_device(args):
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.n_gpu = torch.cuda.device_count()


def setup_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)


def setup_logging(args):
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO,
                        handlers=[
        logging.FileHandler(f"./logs/log{args.version}_seed{args.seed}_{time.strftime('%m%d_%H%M', time.localtime())}.log"),
        logging.StreamHandler()
    ]
                       )
    logger = logging.getLogger(__name__)

    return logger


def build_optimizer(args, model,  warmupsteps, num_total_steps):
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, 
                      lr=args.learning_rate, 
                      eps=args.adam_epsilon)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmupsteps,
                                                num_training_steps=num_total_steps)
    # scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmupsteps,
    #                                             num_training_steps=num_total_steps)
    # scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
    #                                             num_training_steps=args.max_steps)
    return optimizer, scheduler

def build_optimizer2(args, model, warmupsteps, num_total_steps):
    
    params = list(model.named_parameters())
    
    no_decay = ['bias,', 'LayerNorm']
    
    # 分类层
    other = [
        'my_classifer', 'classify_dense',
    ]
    no_main = no_decay + other
    
#     optimizer_grouped_parameters = [
#         {'params': [p for n, p in params if not any(nd in n for nd in no_main)], 
#          'weight_decay': args.weight_decay, 
#          'lr': args.learning_rate*3},
        
#         {'params': [p for n, p in params if not any(nd in n for nd in other) and any(nd in n for nd in no_decay)],
#          'weight_decay': 0.0, 
#          'lr': args.learning_rate*3},
        
#         {'params': [p for n, p in params if any(nd in n for nd in other) and any(nd in n for nd in no_decay)],
#          'weight_decay': 0.0, 
#          'lr': args.learning_rate*5},
        
#         {'params': [p for n, p in params if any(nd in n for nd in other) and not any(nd in n for nd in no_decay)],
#          'weight_decay': args.weight_decay, 
#          'lr': args.learning_rate*5},
#     ]

    optimizer_grouped_parameters = [
        ### swim
        {'params': [p for n, p in params if ('visual_backbone' in n) and not any(nd in n for nd in no_decay)], 
         'weight_decay': args.weight_decay, 
         'lr': args.learning_rate*3},
        {'params': [p for n, p in params if ('visual_backbone' in n) and any(nd in n for nd in no_decay)], 
         'weight_decay': 0.0, 
         'lr': args.learning_rate*3},
        
        ### bert 衰减
        {'params': [p for n, p in params if ('roberta' in n) and not any(nd in n for nd in no_decay)], 
         'weight_decay': args.weight_decay, 
         'lr': args.learning_rate*3},
        ### bert 衰减
        {'params': [p for n, p in params if ('roberta' in n) and any(nd in n for nd in no_decay)],
         'weight_decay': 0.0, 
         'lr': args.learning_rate*3},
        
        ### 分类层
        {'params': [p for n, p in params if any(nd in n for nd in other) and not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay, 
         'lr': args.learning_rate*5},
        ### 分类层 
        {'params': [p for n, p in params if any(nd in n for nd in other) and any(nd in n for nd in no_decay)],
         'weight_decay': 0.0, 
         'lr': args.learning_rate*5},
        
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)

    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmupsteps,
                                                num_training_steps=num_total_steps)
    return optimizer, scheduler

def build_optimizer3(args, model,  warmupsteps, num_total_steps):
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, 
                      lr=args.learning_rate, 
                      eps=args.adam_epsilon)
    # scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmupsteps,
    #                                             num_training_steps=num_total_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmupsteps,
                                                num_training_steps=num_total_steps)
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


class EMA():
    def __init__(self, model, decay, device=None):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self.device = device 
        if self.device is not None:
            self.model.to(device=self.device)

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
                if self.device is not None:
                    self.shadow[name].to(device=self.device)

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


class FGM():
    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self, epsilon=1., emb_name='word_embeddings'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, emb_name='word_embeddings'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}
        

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
