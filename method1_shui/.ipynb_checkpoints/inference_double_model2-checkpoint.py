import torch
from torch.utils.data import SequentialSampler, DataLoader

from config_double_model2 import parse_args
from data_helper.data_helper_double_model2 import MultiModalDataset

from category_id_map import lv2id_to_category_id
from models.doublestream_model import MultiModal

import numpy as np
from tqdm import tqdm


def inference():
    args = parse_args()
    # 1. load data
    dataset = MultiModalDataset(args, args.test_annotation, args.test_zip_feats, test_mode=True)
    sampler = SequentialSampler(dataset)
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
    
    MODEL_DIR = ['saves/double_model2/model_fold_1_epoch_3_mean_f1_0.6752.bin', 
                 'saves/double_model2/model_fold_2_epoch_2_mean_f1_0.6814.bin',
                 'saves/double_model2/model_fold_3_epoch_3_mean_f1_0.6768.bin', 
                 'saves/double_model2/model_fold_4_epoch_3_mean_f1_0.6707.bin',
                 'saves/double_model2/model_fold_5_epoch_3_mean_f1_0.6777.bin']

    pred_matrix = np.zeros((args.num_folds, len(dataset), 200))

    print('-' * 40 + 'model loading...' + '-' * 40)
    for i in range(args.num_folds):
        model = MultiModal(args)
        checkpoint = torch.load(MODEL_DIR[i], map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        if torch.cuda.is_available():
            model = torch.nn.parallel.DataParallel(model.cuda())
        model.eval()

        # 3. inference
        temp = np.zeros((1, 200))
        with torch.no_grad():
            for batch in tqdm(dataloader):
                prediction = model(batch, inference=True)
                temp = np.vstack((temp, prediction.cpu().numpy()))
                # predictions[i].extend(pred_label_id.cpu().numpy())
        pred_matrix[i] = temp[1:, :]

    pred_mean = np.zeros((len(dataset), 200))

    print('-' * 40 + 'predicting...' + '-' * 40)
    for j in range(pred_mean.shape[0]):
        for k in range(pred_mean.shape[1]):
            for i in range(args.num_folds):
                pred_mean[j][k] += pred_matrix[i][j][k]
            pred_mean[j][k] /= args.num_folds
    
    ## 保存概率文件，方便后续模型融合
    np.save('saves/result/double_model2_pred.npy', pred_mean)
    
    predictions = []
    for row in range(pred_mean.shape[0]):
        predictions.append(np.argmax(pred_mean[row]))

    print('-' * 40 + 'results saving...' + '-' * 40)
    # 4. dump results
    with open(args.test_output_csv, 'w') as f:
        for pred_label_id, ann in zip(predictions, dataset.anns):
            video_id = ann['id']
            category_id = lv2id_to_category_id(pred_label_id)
            f.write(f'{video_id},{category_id}\n')


if __name__ == '__main__':
    inference()
