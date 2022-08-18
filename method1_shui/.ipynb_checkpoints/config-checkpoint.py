import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Baseline for Weixin Challenge 2022")

    # ========================== basic config =============================
    parser.add_argument('--debuge', type=int, default=0, help="debuge")
    parser.add_argument('--version', type=str, default='single18_large_base', help="version")
    parser.add_argument('--model_type', type=str, default='single', help="model type")
    parser.add_argument('--resume', type=int, default=0, help='resume')

    parser.add_argument("--seed", type=int, default=2022, help="random seed.")
    parser.add_argument('--num_folds', type=int, default=10, help='K fold')
    parser.add_argument('--fold', type=int, default=0, help='K fold')
    
    parser.add_argument('--local_rank', type=int, default=-1, help='local_rank')

    # ========================== pretrain =============================
    parser.add_argument('--ispretrain', type=int, default=0, help="whether pretrain")
    parser.add_argument('--pretrained_path', type=str, default='pretrain-model/model_pretrain_ub3.pth', help="pretrain_path")

    # ========================= Data Configs ==========================
    parser.add_argument('--train_annotation', type=str, default='/home/tione/notebook/data/annotations/labeled.json')
    parser.add_argument('--train_zip_frames', type=str, default='/home/tione/notebook/data/zip_frames/labeled/')
    # parser.add_argument('--train_annotation', type=str, default='/home/tione/notebook/data/annotations/unlabeled_new.json')
    # parser.add_argument('--train_zip_frames', type=str, default='/home/tione/notebook/data/zip_frames/unlabeled/')
    

    parser.add_argument('--test_annotation', type=str, default='/home/tione/notebook/data/annotations/labeled.json')
    parser.add_argument('--test_zip_frames', type=str, default='/home/tione/notebook/data/zip_frames/labeled/')

    # parser.add_argument('--test_output_csv', type=str, default='/opt/ml/output/result.csv')
    parser.add_argument('--test_output_csv', type=str, default='result.csv')

    parser.add_argument('--val_ratio', default=0.1, type=float, help='split 10 percentages of training data as validation')

    parser.add_argument('--batch_size', default=16, type=int, help="use for training duration per worker")
    parser.add_argument('--val_batch_size', default=32, type=int, help="use for validation duration per worker")
    
    parser.add_argument('--test_batch_size', default=120, type=int, help="use for testing duration per worker")
    parser.add_argument('--prefetch', default=8, type=int, help="use for training duration per worker")
    parser.add_argument('--num_workers', default=4, type=int, help="num_workers for dataloaders")

    # ======================== SavedModel Configs =========================
    parser.add_argument('--savedmodel_path', type=str, default='save/single18_large_base')
    parser.add_argument('--ckpt_file', type=str, default='./saves/single_model1/model_fold_1_epoch_0_mean_f1_0.6761.bin')
    parser.add_argument('--best_score', default=0.675, type=float, help='save checkpoint if mean_f1 > best_score')

    # ========================== Swin ===================================
    parser.add_argument('--swin_pretrained_path', type=str, default='./opensource_models/swin_tiny_patch4_window7_224_22k.pth')

    # ========================= Learning Configs ==========================
    parser.add_argument('--max_epochs', type=int, default=10, help='How many epochs')
    parser.add_argument('--max_steps', default=50000, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--print_steps', type=int, default=200, help="Number of steps to log training metrics.")
    parser.add_argument('--warmup_steps', default=1000, type=int, help="warm ups for parameters not in bert or vit")
    parser.add_argument('--warmup_ratio', default=0.06, type=float, help="warm ups for parameters not in bert or vit")
    parser.add_argument('--minimum_lr', default=0., type=float, help='minimum learning rate')
    parser.add_argument('--learning_rate', default=5e-5/2, type=float, help='initial learning rate')
    parser.add_argument("--weight_decay", default=0.01, type=float, help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-6, type=float, help="Epsilon for Adam optimizer.")

    # ========================== Title BERT =============================
    # parser.add_argument('--bert_dir', type=str, default='hfl/chinese-macbert-base')
    parser.add_argument('--bert_dir', type=str, default='./opensource_models/chinese-roberta-wwm-ext-large')
    parser.add_argument('--bert_cache', type=str, default='./data/cache')
    parser.add_argument('--bert_seq_length', type=int, default=50)
    parser.add_argument('--bert_learning_rate', type=float, default=3e-5)
    parser.add_argument('--bert_warmup_steps', type=int, default=5000)
    parser.add_argument('--bert_max_steps', type=int, default=30000)
    parser.add_argument("--bert_hidden_dropout_prob", type=float, default=0.1)

    # ========================== Video =============================
    parser.add_argument('--frame_embedding_size', type=int, default=768)
    parser.add_argument('--max_frames', type=int, default=8)
    parser.add_argument('--vlad_cluster_size', type=int, default=128)
    parser.add_argument('--vlad_groups', type=int, default=8)
    parser.add_argument('--vlad_hidden_size', type=int, default=1024, help='nextvlad output size using dense')
    parser.add_argument('--se_ratio', type=int, default=8, help='reduction factor in se context gating')

    # ========================== Fusion Layer =============================
    parser.add_argument('--fc_size', type=int, default=1024, help="linear size before final linear")

    return parser.parse_args()
