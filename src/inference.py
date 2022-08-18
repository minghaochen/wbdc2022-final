import torch
from torch.utils.data import SequentialSampler, DataLoader

from config import parse_args
from data_helper import MultiModalDataset
from category_id_map import lv2id_to_category_id
from model import MultiModal
import tqdm
import time
from functools import partial

from apex import amp
from util import setup_device, setup_seed, setup_logging, build_optimizer, evaluate



def inference():
    args = parse_args()
    print(args)
    # 1. load data
    load_time1 = time.time()
    dataset = MultiModalDataset(args, args.test_annotation, args.test_zip_frames, test_mode=True, get_skf='valid')
    
    dataloader_class = partial(DataLoader, pin_memory=True, num_workers=args.num_workers, prefetch_factor=args.prefetch)
    
    sampler = SequentialSampler(dataset)
    # dataloader = DataLoader(dataset,
    #                         batch_size=args.test_batch_size,
    #                         sampler=sampler,
    #                         drop_last=False,
    #                         pin_memory=True,
    #                         num_workers=args.num_workers,
    #                         prefetch_factor=args.prefetch)
    
    dataloader = dataloader_class(dataset,
                                  batch_size=args.test_batch_size,
                                  sampler=sampler,
                                  drop_last=False)
    
    load_time2 = time.time()
    print('load data complete', load_time2-load_time1)
    # 2. load model    
    load_time1 = time.time()
    model = MultiModal(args)
    checkpoint = torch.load(args.ckpt_file, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    model = model.half()
    model = model.to('cuda')
    # 混合精度
    optimizer, scheduler = build_optimizer(args, model)
    # model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
    
    if torch.cuda.is_available():
        model = torch.nn.parallel.DataParallel(model.cuda())
        model.eval()
    load_time2 = time.time()
    print('load model complete', load_time2-load_time1)
    
    load_time1 = time.time()
    # 3. inference
    predictions = []
    with torch.no_grad():
        for batch_id, batch in tqdm.tqdm(enumerate(dataloader)):
            pred_label_id, raw_prediction = model(batch, inference=True)
            predictions.extend(pred_label_id.cpu().numpy())

    # 4. dump results
    with open(args.test_output_csv, 'w') as f:
        for pred_label_id, ann in zip(predictions, dataset.anns):
            video_id = ann['id']
            category_id = lv2id_to_category_id(pred_label_id)
            f.write(f'{video_id},{category_id}\n')
    load_time2 = time.time()
    print('dataset', len(dataset), 'qps', int(len(dataset)/(load_time2-load_time1)))

if __name__ == '__main__':
    inference()
