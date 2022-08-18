cd /opt/ml/wxcode

python -u inference_amp.py \
    --test_annotation /opt/ml/input/data/annotations/test.json \
    --test_zip_frames /opt/ml/input/data/zip_frames/test/ \
    --ckpt_file saves/model_fold_1_epoch_4_mean_f1_0.6968.bin
    
# python -m torch.distributed.launch --nproc_per_node 2 -u inference_ddp.py \
#     --test_annotation /opt/ml/input/data/annotations/test.json \
#     --test_zip_frames /opt/ml/input/data/zip_frames/test/ \
#     --ckpt_file save/single_model3/model_fold_1_epoch_4_mean_f1_0.6989.bin
