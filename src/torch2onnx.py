import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import onnx
from model_trt import MultiModal
from config import parse_args
from util import setup_device, setup_seed, setup_logging, build_optimizer, evaluate

print(torch.__version__)

input_name = ['title_input', 'title_mask', 'frame_input', 'frame_mask']
output_name = ['output']


title_input = torch.zeros(20, 388).cuda().int()
title_mask = torch.zeros(20, 388).cuda().int()
frame_input = torch.zeros(20, 8, 3, 224, 224).cuda().float()
frame_mask = torch.zeros(20, 8).cuda().int()


args = parse_args()
setup_logging()
setup_device(args)
setup_seed(args)
args.fold = 0
        
        
model = MultiModal(args).cuda()
checkpoint = torch.load('save/v1/model_fold_0_epoch_2_mean_f1_0.6925.bin', map_location='cpu')
model.load_state_dict(checkpoint['model_state_dict'],strict=True)
model.eval()

with torch.no_grad():
    example_outputs = model(title_input, title_mask, frame_input, frame_mask)

    print(example_outputs.shape)
    
    torch.onnx.export(model, 
                  (title_input, title_mask, frame_input, frame_mask), 
                  'model_onnx.onnx', 
                  opset_version=13, 
                  example_outputs=example_outputs,
                  input_names=input_name, 
                  output_names=output_name, 
#                   dynamic_axes = dynamic_axes,
                  verbose=False)