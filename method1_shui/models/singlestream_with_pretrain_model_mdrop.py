import torch
import torch.nn as nn
import torch.nn.functional as F
from category_id_map import CATEGORY_ID_LIST

import math
import random

import sys
from models.masklm import MaskLM, MaskVideo, ShuffleVideo
from transformers.models.bert.modeling_bert import BertConfig, BertOnlyMLMHead
from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertEmbeddings, BertEncoder
from .swin import swin_tiny

def linear_combination(x, y, epsilon):
    return epsilon * x + (1 - epsilon) * y


def reduce_loss(loss, reduction='mean'):
    return loss.mean() if reduction == 'mean' else loss.sum() if reduction == 'sum' else loss

class LabelSmoothingCrossEntropy(torch.nn.Module):
    def __init__(self, epsilon: float = 0.1, reduction='mean'):
        super().__init__()
        self.epsilon = epsilon
        self.reduction = reduction

    def forward(self, preds, target):
        n = preds.size()[-1]
        log_preds = F.log_softmax(preds, dim=-1)
        loss = reduce_loss(-log_preds.sum(dim=-1), self.reduction)
        nll = F.nll_loss(log_preds, target, reduction=self.reduction)
        return linear_combination(loss / n, nll, self.epsilon)
    

def init_params(module_lst):
    for module in module_lst:
        for param in module.parameters():
            if param.dim() > 1:
                torch.nn.init.xavier_uniform_(param)
    return

def init_weights(module):
    if isinstance(module, (nn.Linear, nn.Embedding)):
        module.weight.data.normal_(mean=0.0, std=0.02)
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)

    if isinstance(module, nn.Linear) and module.bias is not None:
        module.bias.data.zero_()

class myClassifer(nn.Module):
    def __init__(self, uni_bert_cfg, dropout = [0.2, 0.2], hidden_layers = 512,  n_weights=12):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout[0])
        self.high_dropout = nn.Dropout(p=dropout[1])

        weights_init = torch.zeros(n_weights).float()
        weights_init.data[:-1] = -3
        self.layer_weights = torch.nn.Parameter(weights_init)

        self.attention = nn.Sequential(
            nn.Linear(uni_bert_cfg.hidden_size, hidden_layers),
            nn.Tanh(),
            nn.Linear(hidden_layers, 1),
            nn.Softmax(dim=1)
        )
        
        init_params([self.attention])

    def forward(self, hidden_states):
        cls_outputs = torch.stack(
            [self.dropout(layer) for layer in hidden_states[-12:]], dim=0
        )
        cls_output = (
                torch.softmax(self.layer_weights, dim=0).unsqueeze(1).unsqueeze(1).unsqueeze(1) * cls_outputs).sum(
            0)

        logits = torch.mean(
            torch.stack(
                [torch.sum(self.attention(self.high_dropout(cls_output)) * cls_output, dim=1) for _ in range(5)],
                dim=0,
            ),
            dim=0,
        )
        return logits
    
class MeanPooling(nn.Module):
    def __init__(self):
        super(MeanPooling, self).__init__()
    def forward(self, last_hidden_state, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        return mean_embeddings
    
class SingleStream_model_pretrain(nn.Module):
    def __init__(self, args, init_from_pretrain=True):
        super().__init__()
        
        ## 预训练部分
        uni_bert_cfg = BertConfig.from_pretrained(f'{args.bert_dir}/config.json')
        if init_from_pretrain:
            self.roberta = UniBertForMaskedLM.from_pretrained(args.bert_dir, config=uni_bert_cfg)
        else:
            self.roberta = UniBertForMaskedLM(uni_bert_cfg)

        self.visual_backbone = swin_tiny(args.swin_pretrained_path)
        
        # multi sample from paper
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)
        self.dropout3 = nn.Dropout(0.5)
        self.dropout4 = nn.Dropout(0.5)
        self.dropout5 = nn.Dropout(0.5)
        
        # self.class_video = nn.Linear(768, len(CATEGORY_ID_LIST))
        # self.class_video.apply(init_weights)
        # self.avgpool = nn.AdaptiveAvgPool1d(1)
        # self.avgpool.apply(init_weights)
        
        ## 任务层部分
        self.my_classifer = myClassifer(uni_bert_cfg, dropout = [0.2, 0.2], hidden_layers = 512,  n_weights=12)
        # self.my_classifer = MeanPooling()
        self.classify_dense = nn.Linear(uni_bert_cfg.hidden_size, len(CATEGORY_ID_LIST))

    def forward(self, inputs, inference=False):
        
        inputs['frame_input'] = self.visual_backbone(inputs['frame_input'])
        video_feature, video_mask = inputs['frame_input'], inputs['frame_mask']
        text_input_ids, text_mask = inputs['text_input'], inputs['text_mask']
        
        # video_pred = self.class_video(self.avgpool(inputs['frame_input'].transpose(1, 2)).squeeze(2))
        
        encoder_outputs, _, mask = self.roberta(video_feature, video_mask, text_input_ids, text_mask, return_mlm=False)
        
        # encoder_outputs, mask = self.bert(video_feature, video_mask, text_input_ids, text_mask)
        
        output = self.my_classifer(encoder_outputs['hidden_states'])
        # output = self.my_classifer(encoder_outputs['last_hidden_state'], mask)
        
        prediction1 = self.classify_dense(self.dropout1(output))
        prediction2 = self.classify_dense(self.dropout2(output))
        prediction3 = self.classify_dense(self.dropout3(output))
        prediction4 = self.classify_dense(self.dropout4(output))
        prediction5 = self.classify_dense(self.dropout5(output))
        prediction = prediction1
        # prediction = (prediction1 + prediction2 + prediction3 + prediction4 + prediction5) / 5
        
        # prediction = self.classify_dense(output)

        if inference:
            return prediction
        else:
            # loss, accuracy, pred_label_id, label = self.cal_loss(prediction, inputs['label'])
            loss, accuracy, pred_label_id, label = self.cal_loss(prediction, inputs['label'],
                                                                 prediction1, prediction2, prediction3, 
                                                                 prediction4, prediction5)
            return loss, accuracy, pred_label_id, label, prediction
    
    @staticmethod  
    def get_mask(frame_mask, text_mask):
        cls_mask = text_mask[:, 0:1]
        text_mask = text_mask[:, 1:]
        mask = torch.cat([cls_mask, frame_mask, text_mask], 1)
        return mask

    @staticmethod
    def cal_loss(prediction, label, 
                 prediction1, prediction2, prediction3, prediction4, prediction5
                ):
        label = label.squeeze(dim=1)
        # loss = LabelSmoothingCrossEntropy(epsilon=0.1)(prediction, label)
        # loss = F.cross_entropy(prediction, label, label_smoothing=0.1)  # label smooth
        
        loss_main1 = F.cross_entropy(prediction1, label)
        loss_main2 = F.cross_entropy(prediction2, label)
        loss_main3 = F.cross_entropy(prediction3, label)
        loss_main4 = F.cross_entropy(prediction4, label)
        loss_main5 = F.cross_entropy(prediction5, label)
        loss = (loss_main1 + loss_main2 + loss_main3 + loss_main4 + loss_main5)/5
        
        with torch.no_grad():
            pred_label_id = torch.argmax(prediction, dim=1)
            accuracy = (label == pred_label_id).float().sum() / label.shape[0]
        return loss, accuracy, pred_label_id, label
    
    @staticmethod
    def cal_loss2(prediction, video_pred, label):
        label = label.squeeze(dim=1)
        # loss = F.cross_entropy(prediction, label)
        loss_main = LabelSmoothingCrossEntropy(reduction='mean')(prediction, label)
        loss_sub1 = LabelSmoothingCrossEntropy(reduction='mean')(video_pred, label)
        loss = loss_main+0.25*loss_sub1
        with torch.no_grad():
            pred_label_id = torch.argmax(prediction, dim=1)
            accuracy = (label == pred_label_id).float().sum() / label.shape[0]
        return loss, accuracy, pred_label_id, label, prediction

class SingleStream_model_pretrain2(nn.Module):
    def __init__(self, args, init_from_pretrain=True):
        super().__init__()
        
        ## 预训练部分
        uni_bert_cfg = BertConfig.from_pretrained(f'{args.bert_dir}/config.json')
        uni_bert_cfg.hidden_dropout_prob = 0.1
        uni_bert_cfg.num_hidden_layers = 12
        uni_bert_cfg.num_attention_heads = 12
        if init_from_pretrain:
            self.roberta = UniBertForMaskedLM.from_pretrained(args.bert_dir, config=uni_bert_cfg)
        else:
            self.roberta = UniBertForMaskedLM(uni_bert_cfg)

        self.visual_backbone = swin_tiny(args.swin_pretrained_path)
        
        # self.class_video = nn.Linear(768, len(CATEGORY_ID_LIST))
        # self.class_video.apply(init_weights)
        # self.avgpool = nn.AdaptiveAvgPool1d(1)
        # self.avgpool.apply(init_weights)
        
        ## 任务层部分
        
        self.text_video_mean = myClassifer()
        self.video_mean = MeanPooling()
        self.drop = nn.Dropout(0.2)
        
        # self.dropout = nn.Dropout(0.1)
        # self.dropout1 = nn.Dropout(0.1)
        # self.dropout2 = nn.Dropout(0.2)
        # self.dropout3 = nn.Dropout(0.3)
        # self.dropout4 = nn.Dropout(0.4)
        # self.dropout5 = nn.Dropout(0.5)
        
        
        self.classify_dense = nn.Linear(768, len(CATEGORY_ID_LIST))

    def forward(self, inputs, inference=False):
        
        inputs['frame_input'] = self.visual_backbone(inputs['frame_input'])
        video_feature, video_mask = inputs['frame_input'], inputs['frame_mask']
        text_input_ids, text_mask = inputs['text_input'], inputs['text_mask']
        
        # video_pred = self.class_video(self.avgpool(inputs['frame_input'].transpose(1, 2)).squeeze(2))
        
        encoder_outputs, _, mask = self.roberta(video_feature, video_mask, text_input_ids, text_mask, return_mlm=False)
        
        # encoder_outputs, mask = self.bert(video_feature, video_mask, text_input_ids, text_mask)
        
        output1 = self.text_video_mean(encoder_outputs['hidden_states'])
        output2 = self.video_mean(video_feature, video_mask)
        output = self.drop((output1 + output2) / 2)
        
        # prediction1 = self.classify_dense(self.dropout1(output))
        # prediction2 = self.classify_dense(self.dropout2(output))
        # prediction3 = self.classify_dense(self.dropout3(output))
        # prediction4 = self.classify_dense(self.dropout4(output))
        # prediction5 = self.classify_dense(self.dropout5(output))
        # prediction = (prediction1 + prediction2 + prediction3 + prediction4 + prediction5) / 5
        
        prediction = self.classify_dense(output)

        if inference:
            return prediction
        else:
            loss, accuracy, pred_label_id, label = self.cal_loss(prediction, inputs['label'])
            return loss, accuracy, pred_label_id, label, prediction
    
    @staticmethod  
    def get_mask(frame_mask, text_mask):
        cls_mask = text_mask[:, 0:1]
        text_mask = text_mask[:, 1:]
        mask = torch.cat([cls_mask, frame_mask, text_mask], 1)
        return mask

    @staticmethod
    def cal_loss(prediction, label):
        label = label.squeeze(dim=1)
        loss = LabelSmoothingCrossEntropy(epsilon=0.1)(prediction, label)
        # loss = F.cross_entropy(prediction, label, label_smoothing=0.1)  # label smooth
        with torch.no_grad():
            pred_label_id = torch.argmax(prediction, dim=1)
            accuracy = (label == pred_label_id).float().sum() / label.shape[0]
        return loss, accuracy, pred_label_id, label
    
    @staticmethod
    def cal_loss2(prediction, video_pred, label):
        label = label.squeeze(dim=1)
        # loss = F.cross_entropy(prediction, label)
        loss_main = LabelSmoothingCrossEntropy(reduction='mean')(prediction, label)
        loss_sub1 = LabelSmoothingCrossEntropy(reduction='mean')(video_pred, label)
        loss = loss_main+0.25*loss_sub1
        with torch.no_grad():
            pred_label_id = torch.argmax(prediction, dim=1)
            accuracy = (label == pred_label_id).float().sum() / label.shape[0]
        return loss, accuracy, pred_label_id, label, prediction

class UniBertForMaskedLM(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = UniBert3(config)
        self.cls = BertOnlyMLMHead(config)
        
    # Copied from transformers.models.bert.modeling_bert.BertModel.forward
    def forward(self, 
                video_feature, video_mask, 
                text_input_ids, text_mask,
                gather_index=None, return_mlm=False):
                
        encoder_outputs, mask = self.bert(video_feature, video_mask, text_input_ids, text_mask)
        video_size = video_feature.size()
        if return_mlm:
            return encoder_outputs, self.cls(encoder_outputs['last_hidden_state'])[:, 1 + video_size[1]: , :], mask
        else:
            return encoder_outputs, None, mask

class VideoTextEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(2, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        
        # self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))
        # if version.parse(torch.__version__) > version.parse("1.6.0"):
        self.register_buffer(
            "token_type_ids",
            torch.zeros(self.position_ids.size(), dtype=torch.long),
            persistent=False,
        )
            
    def forward(
        self,
        text_input_ids: torch.LongTensor,
        frame_inputs_embeds: torch.FloatTensor, 
        past_key_values_length: int = 0,
    ) -> torch.Tensor:

        text_input_shape = text_input_ids.size()
        # torch.Size([16, 257])
        frame_input_shape = frame_inputs_embeds.size()
        # torch.Size([16, 8, 768])
        
        seq_length = text_input_shape[1] + frame_input_shape[1]
        
        text_token_type_ids = torch.zeros(text_input_shape[1] - 1,  dtype=text_input_ids.dtype, device=text_input_ids.device)
        frame_token_type_ids = torch.ones(frame_input_shape[1] + 1, dtype=text_input_ids.dtype, device=frame_inputs_embeds.device)
        
        ## position_ids
        position_ids = self.position_ids[:, past_key_values_length: seq_length + past_key_values_length]
        position_embeddings = self.position_embeddings(position_ids)
        
        ## token_type
        token_type_ids = torch.hstack((frame_token_type_ids, text_token_type_ids))
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        
        ## word embedding
        text_inputs_embeds = self.word_embeddings(text_input_ids)
        # print(text_inputs_embeds.size(), frame_inputs_embeds.size())
        # print(text_inputs_embeds[:, 0:1, :].size())
        
        inputs_embeds = torch.cat((text_inputs_embeds[:, 0:1, :], 
                                   frame_inputs_embeds, 
                                   text_inputs_embeds[:, 1:, :])
                                  , 1)
        
        embeddings = inputs_embeds + token_type_embeddings + position_embeddings
                    
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings
    
###初赛
class UniBert(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.embeddings = BertEmbeddings(config)
        self.video_dense = nn.Linear(768, 768)
        self.encoder = BertEncoder(config)

        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(self, video_feature, video_mask, text_input_ids, text_mask):
        text_emb = self.embeddings(input_ids=text_input_ids)
        # text input is [CLS][SEP] t e x t [SEP]
        cls_emb = text_emb[:, 0:1, :]
        text_emb = text_emb[:, 1:, :]

        cls_mask = text_mask[:, 0:1]
        text_mask = text_mask[:, 1:]

        video_feature = self.video_dense(video_feature)
        frame_emb = self.embeddings(inputs_embeds=video_feature)

        # [CLS] Video [SEP] Text [SEP]
        embedding_output = torch.cat([cls_emb, frame_emb, text_emb], 1)

        mask = torch.cat([cls_mask, video_mask, text_mask], 1)
        extended_mask = mask[:, None, None, :]
        extended_mask = (1.0 - extended_mask) * -10000.0

        encoder_outputs = self.encoder(embedding_output, 
                                       attention_mask=extended_mask,
                                       output_hidden_states=True)
        return encoder_outputs, mask
    
## QQ浏览器
class UniBert2(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.embeddings = BertEmbeddings(config)
        
        self.video_dense = torch.nn.Linear(768, 768)
        self.video_embeddings = BertEmbeddings(config)
        
        self.encoder = BertEncoder(config)

        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(self, video_feature, video_mask, text_input_ids, text_mask):

        text_emb = self.embeddings(input_ids=text_input_ids)
        # text input is [CLS][SEP] t e x t [SEP]
        cls_emb = text_emb[:, 0:1, :]
        text_emb = text_emb[:, 1:, :]

        cls_mask = text_mask[:, 0:1]
        text_mask = text_mask[:, 1:]
        
        video_feature = self.video_dense(video_feature)
        frame_emb = self.video_embeddings(inputs_embeds=video_feature)

        # [CLS] Video [SEP] Text [SEP]
        embedding_output = torch.cat([cls_emb, frame_emb, text_emb], 1)

        mask = torch.cat([cls_mask, video_mask, text_mask], 1)
        extended_mask = mask[:, None, None, :]
        extended_mask = (1.0 - extended_mask) * -10000.0

        encoder_outputs = self.encoder(embedding_output, 
                                       attention_mask=extended_mask,
                                       output_hidden_states=True)
        return encoder_outputs, mask

### VideoTextEmbedding
class UniBert3(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.embeddings = VideoTextEmbedding(config)
        
        if self.config.hidden_size != 768:
            self.video_dense = nn.Linear(768, config.hidden_size)
        self.encoder = BertEncoder(config)
        
        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(self, video_feature, video_mask, text_input_ids, text_mask):
        if self.config.hidden_size != 768:
            video_feature = self.video_dense(video_feature)
         # [CLS] Video [SEP] Text [SEP]
        embedding_output = self.embeddings(text_input_ids=text_input_ids, frame_inputs_embeds=video_feature)

        cls_mask = text_mask[:, 0:1]
        text_mask = text_mask[:, 1:]
        mask = torch.cat([cls_mask, video_mask, text_mask], 1)
        extended_mask = mask[:, None, None, :]
        extended_mask = (1.0 - extended_mask) * -10000.0

        encoder_outputs = self.encoder(embedding_output, 
                                       attention_mask=extended_mask,
                                       output_hidden_states=True)
        return encoder_outputs, mask
    
### 初赛去掉 videodense   
class UniBert4(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)

        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(self, video_feature, video_mask, text_input_ids, text_mask):

        text_emb = self.embeddings(input_ids=text_input_ids)
        # text input is [CLS]  [CLS]video[SEP] [SEP] t e x t [SEP]
        cls_emb = text_emb[:, 0:1, :]
        text_emb = text_emb[:, 1:, :]

        cls_mask = text_mask[:, 0:1]
        text_mask = text_mask[:, 1:]
        
        # [CLS] Video [SEP] Text [SEP]
        embedding_output = torch.cat([cls_emb, video_feature, text_emb], 1)

        mask = torch.cat([cls_mask, video_mask, text_mask], 1)
        extended_mask = mask[:, None, None, :]
        extended_mask = (1.0 - extended_mask) * -10000.0

        encoder_outputs = self.encoder(embedding_output, 
                                       attention_mask=extended_mask,
                                       output_hidden_states=True)
        return encoder_outputs, mask