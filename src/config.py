import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Baseline for Weixin Challenge 2022")

    parser.add_argument("--seed", type=int, default=2017, help="random seed.")
    parser.add_argument('--dropout', type=float, default=0.3, help='dropout ratio')
    parser.add_argument('--local_rank', type=int, default=-1, help='local_rank')

    # ========================= Data Configs ==========================

    parser.add_argument('--train_annotation', type=str, default='/opt/ml/input/data/annotations/labeled.json')
    parser.add_argument('--test_annotation', type=str, default='/opt/ml/input/data/annotations/labeled.json')
    parser.add_argument('--train_zip_frames', type=str, default='/opt/ml/input/data/zip_frames/labeled/')
    parser.add_argument('--test_zip_frames', type=str, default='/opt/ml/input/data/zip_frames/labeled/')
    parser.add_argument('--test_output_csv', type=str, default='result.csv')
    parser.add_argument('--unlabel_path', type=str, default='save/unlabeled_pred_sample10000.json')
    
    parser.add_argument('--val_ratio', default=0.1, type=float, help='split 10 percentages of training data as validation')
    parser.add_argument('--batch_size', default=32, type=int, help="use for training duration per worker")
    parser.add_argument('--val_batch_size', default=64, type=int, help="use for validation duration per worker")
    parser.add_argument('--test_batch_size', default=256, type=int, help="use for testing duration per worker")
    parser.add_argument('--prefetch', default=8, type=int, help="use for training duration per worker")
    parser.add_argument('--num_workers', default=4, type=int, help="num_workers for dataloaders")

    # ======================== SavedModel Configs =========================
    parser.add_argument('--savedmodel_path', type=str, default='save/v8')
    parser.add_argument('--ckpt_file', type=str, default='save/v1/model_fold_0_epoch_2_mean_f1_0.7125.bin')
    parser.add_argument('--best_score', default=0.0, type=float, help='save checkpoint if mean_f1 > best_score')

    # ========================= Learning Configs ==========================
    # pretrain
#     parser.add_argument('--max_epochs', type=int, default=5, help='How many epochs')
#     parser.add_argument('--max_steps', default=1000000//32*5, type=int, metavar='N', help='number of total epochs to run')
#     parser.add_argument('--print_steps', type=int, default=200, help="Number of steps to log training metrics.")
#     parser.add_argument('--warmup_steps', default=1000000//128, type=int, help="warm ups for parameters not in bert or vit")
#     parser.add_argument('--learning_rate', default=2e-5, type=float, help='initial learning rate')
    # finetune
    parser.add_argument('--max_epochs', type=int, default=10, help='How many epochs')
    parser.add_argument('--max_steps', default=100000//32*10, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--print_steps', type=int, default=200, help="Number of steps to log training metrics.")
    parser.add_argument('--warmup_steps', default=1000*1, type=int, help="warm ups for parameters not in bert or vit")
    
    
    parser.add_argument('--minimum_lr', default=0., type=float, help='minimum learning rate')
    parser.add_argument('--learning_rate', default=5e-5, type=float, help='initial learning rate')
    # parser.add_argument('--learning_rate', default=1e-5, type=float, help='initial learning rate')
    parser.add_argument("--weight_decay", default=0.01, type=float, help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-6, type=float, help="Epsilon for Adam optimizer.")
    
    # ========================== Swin ===================================
    parser.add_argument('--swin_pretrained_path', type=str, default='../opensource_models/swin_small_patch4_window7_224_22k.pth')

    # ========================== Title BERT =============================
    parser.add_argument('--bert_dir', type=str, default='../opensource_models/chinese-macbert-base')
    # parser.add_argument('--bert_dir', type=str, default='hfl/chinese-roberta-wwm-ext')
    parser.add_argument('--bert_cache', type=str, default='data/cache')
    parser.add_argument('--bert_seq_length', type=int, default=128*3)
    parser.add_argument('--bert_learning_rate', type=float, default=1e-5)
    # parser.add_argument('--bert_learning_rate', type=float, default=3e-5)
    parser.add_argument('--bert_warmup_steps', type=int, default=5000)
    parser.add_argument('--bert_max_steps', type=int, default=30000)
    parser.add_argument("--bert_hidden_dropout_prob", type=float, default=0.1)

    # ========================== Video =============================
    parser.add_argument('--frame_embedding_size', type=int, default=768)
    parser.add_argument('--max_frames', type=int, default=8)
    parser.add_argument('--vlad_cluster_size', type=int, default=64)
    parser.add_argument('--vlad_groups', type=int, default=8)
    parser.add_argument('--vlad_hidden_size', type=int, default=768, help='nextvlad output size using dense')
    # parser.add_argument('--vlad_hidden_size', type=int, default=1024, help='nextvlad output size using dense')
    parser.add_argument('--se_ratio', type=int, default=8, help='reduction factor in se context gating')

    # ========================== Fusion Layer =============================
    parser.add_argument('--fc_size', type=int, default=512, help="linear size before final linear")

    return parser.parse_args()
