# 代码说明

## 环境配置

- Python 版本：3.8.10 
- PyTorch 版本：1.8.1 
- CUDA 版本：11.5

所需环境在 requirements.txt 中定义。

## 数据

- 仅使用大赛提供的有标注数据（10万）。
- 未使用任何额外数据及无标签数据

## 预训练模型

- 文本侧使用了 huggingface 上提供的 hfl/chinese-macbert-base 模型。链接为： https://huggingface.co/hfl/chinese-macbert-base
- 视频侧使用了官方提供的 swin_small_patch4_window7_224_22k 模型。

## 算法描述

**方案**：双流模型 + coattention layer

- 对于文本特征，使用 mac-bert 模型来提取特征。文本同时使用标题、asr和ocr，长度分别为128，总长度384
- 对于视觉特征，经过 swin-small 模型来提取特征
- 视觉和文本特征经过 coattention layer 进行交互，最后通过 MLP 预测二级分类的 id
- 微调时采用了EMA和FGM对抗训练


**最终结果**：采用两个不同的种子进行训练然后推理融合，logits = 0.5 * logits1 + 0.5 * logits2

## 运行说明

- train.sh 对应方案的结果复现
- inference.sh 进行不同种子的推理结果融合


## 性能

- A榜测试性能：0.71198
- B榜测试性能：0.71073


## 训练流程

- 直接在有标注数据上微调3个epoch，一个模型的训练时长小于6小时。

## 测试流程

十折交叉验证训练，划分10%的数据作为验证集，取验证集上最好的模型来做测试。