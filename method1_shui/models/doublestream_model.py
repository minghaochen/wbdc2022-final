import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel
from .swin import swin_tiny

from transformers import AutoConfig, AutoModel,AutoTokenizer,logging

from category_id_map import CATEGORY_ID_LIST

def init_params(module_lst):
    for module in module_lst:
        for param in module.parameters():
            if param.dim() > 1:
                torch.nn.init.xavier_uniform_(param)
    return

class myClassifer(nn.Module):
    def __init__(self, dropout = [0.2, 0.2], hidden_layers = 512,  n_weights=12):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout[0])
        self.high_dropout = nn.Dropout(p=dropout[1])

        weights_init = torch.zeros(n_weights).float()
        weights_init.data[:-1] = -3
        self.layer_weights = torch.nn.Parameter(weights_init)

        self.attention = nn.Sequential(
            nn.Linear(768, hidden_layers),
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


class MultiModal(nn.Module):
    def __init__(self, args):
        super().__init__()
        
        ################################ text #######################################
        self.bert_config = AutoConfig.from_pretrained(args.bert_dir)
        self.bert_config.output_hidden_states = True
        self.bert = AutoModel.from_pretrained(args.bert_dir, config=self.bert_config)
        bert_output_size = 768

        self.text_fc = myClassifer()
        self.LN_text = nn.LayerNorm(bert_output_size, 5e-10)
        
        
        ################################ video #######################################
        # self.nextvlad = NeXtVLAD(args.frame_embedding_size, args.vlad_cluster_size,
        #                          output_size=args.vlad_hidden_size, dropout=args.dropout)
        
        # self.enhance = SENet(channels=args.vlad_hidden_size, ratio=args.se_ratio)

        self.visual_backbone = swin_tiny(args.swin_pretrained_path)
        self.nextvlad1 = NeXtVLAD(args.frame_embedding_size, args.vlad_cluster_size,
                                 output_size=args.vlad_hidden_size, dropout=args.dropout)
        self.nextvlad2 = NeXtVLAD(args.frame_embedding_size, args.vlad_cluster_size,
                                 output_size=args.vlad_hidden_size, dropout=args.dropout)
        self.nextvlad3 = NeXtVLAD(args.frame_embedding_size, args.vlad_cluster_size,
                                 output_size=args.vlad_hidden_size, dropout=args.dropout)
        self.mix_weights = nn.Linear(args.frame_embedding_size, 3)
        # self.bn = nn.BatchNorm1d(args.frame_embedding_size)
        self.video_dense = nn.Sequential(
            nn.Linear(args.vlad_hidden_size, args.vlad_hidden_size),
            nn.ReLU()
        )
        self.LN_video = nn.LayerNorm(args.vlad_hidden_size, 5e-10)
        
        ################################ text + video #######################################
        self.fusion = ConcatDenseSE(args.vlad_hidden_size + bert_output_size, args.fc_size, args.se_ratio, args.dropout)
        self.LN_video_text = nn.LayerNorm(args.fc_size, 5e-10)

        ################################ 分类层 #######################################
        self.classifier = nn.Linear(args.fc_size, len(CATEGORY_ID_LIST))

    def forward(self, inputs, inference=False):
        ################################ text #######################################
        hidden_states = self.bert(inputs['text_input'], inputs['text_mask'])['hidden_states']
        bert_embedding = self.my_classifer(hidden_states)
        ## text layerNorm归一化
        bert_embedding = self.LN_text(bert_embedding)
        
        ################################ video #######################################
        inputs['frame_input'] = self.visual_backbone(inputs['frame_input'])
        ### # frt_mean
        frt_mean_1 = torch.mean(inputs['frame_input'], axis=1)
        # frt_mean_1 = self.bn(frt_mean_1)
        mix_weights_1 = self.mix_weights(frt_mean_1) # b,3
        mix_weights_1 = nn.Softmax(dim=-1)(mix_weights_1)
        # 1
        vision_embedding_a_1 = self.nextvlad1(inputs['frame_input'], inputs['frame_mask']) # b, n
        # 2
        vision_embedding_b_1 = self.nextvlad2(inputs['frame_input'], inputs['frame_mask']) # b, n
        # 3
        vision_embedding_c_1 = self.nextvlad3(inputs['frame_input'], inputs['frame_mask'])
        
        vision_embedding_1 = [vision_embedding_a_1, vision_embedding_b_1, vision_embedding_c_1]
        vision_embedding_1 = torch.stack(vision_embedding_1, dim=1) # b, 3, n 
        # b, 3, 1 * b, 3, n = b, 3, n ->sum b, n
        mix_vision_embedding_1 = torch.sum(torch.multiply(torch.unsqueeze(mix_weights_1, -1), vision_embedding_1), axis=1)
        
        #### video embedding
        # vision_embedding = self.nextvlad(inputs['frame_input'], inputs['frame_mask'])
        # vision_embedding = self.enhance(vision_embedding)
        
        vision_embedding = self.video_dense(mix_vision_embedding_1)
        vision_embedding = self.LN_video(vision_embedding)

        ################################ text + video #######################################
        final_embedding = self.fusion([bert_embedding, vision_embedding])
        final_embedding = self.LN_video_text(final_embedding)

        ################################ 分类层 #######################################
        prediction = self.classifier(final_embedding)

        if inference:
            # return torch.argmax(prediction, dim=1)
            return prediction
        else:
            loss, accuracy, pred_label_id, label = self.cal_loss(prediction, inputs['label'])
            return loss, accuracy, pred_label_id, label, prediction

    @staticmethod
    def cal_loss(prediction, label):
        label = label.squeeze(dim=1)
        loss = F.cross_entropy(prediction, label)
        
        with torch.no_grad():
            pred_label_id = torch.argmax(prediction, dim=1)
            accuracy = (label == pred_label_id).float().sum() / label.shape[0]
        return loss, accuracy, pred_label_id, label


class NeXtVLAD(nn.Module):
    def __init__(self, feature_size, cluster_size, output_size=1024, expansion=2, groups=8, dropout=0.2):
        super().__init__()
        self.feature_size = feature_size
        self.output_size = output_size
        self.expansion_size = expansion
        self.cluster_size = cluster_size
        self.groups = groups
        self.drop_rate = dropout

        self.new_feature_size = self.expansion_size * self.feature_size // self.groups

        self.dropout = torch.nn.Dropout(self.drop_rate)
        self.expansion_linear = torch.nn.Linear(self.feature_size, self.expansion_size * self.feature_size)
        self.group_attention = torch.nn.Linear(self.expansion_size * self.feature_size, self.groups)
        self.cluster_linear = torch.nn.Linear(self.expansion_size * self.feature_size, self.groups * self.cluster_size,
                                              bias=False)
        self.cluster_weight = torch.nn.Parameter(
            torch.nn.init.normal_(torch.rand(1, self.new_feature_size, self.cluster_size), std=0.01))
        self.fc = torch.nn.Linear(self.new_feature_size * self.cluster_size, self.output_size)
        
        self.bn0 = nn.BatchNorm1d(self.groups * self.cluster_size)


    def forward(self, inputs, mask):
        # todo mask
        inputs = self.expansion_linear(inputs)
        attention = self.group_attention(inputs)
        attention = torch.sigmoid(attention)
        attention = attention.reshape([-1, inputs.size(1) * self.groups, 1])
        reshaped_input = inputs.reshape([-1, self.expansion_size * self.feature_size])
        activation = self.cluster_linear(reshaped_input)
        
        activation = self.bn0(activation)
        
        activation = activation.reshape([-1, inputs.size(1) * self.groups, self.cluster_size])
        activation = torch.softmax(activation, dim=-1)
        activation = activation * attention
        a_sum = activation.sum(-2, keepdim=True)
        a = a_sum * self.cluster_weight
        activation = activation.permute(0, 2, 1).contiguous()
        reshaped_input = inputs.reshape([-1, inputs.shape[1] * self.groups, self.new_feature_size])
        vlad = torch.matmul(activation, reshaped_input)
        vlad = vlad.permute(0, 2, 1).contiguous()
        vlad = F.normalize(vlad - a, p=2, dim=1)
        vlad = vlad.reshape([-1, self.cluster_size * self.new_feature_size])
        vlad = self.dropout(vlad)
        vlad = self.fc(vlad)
        return vlad


class SENet(nn.Module):
    def __init__(self, channels, ratio=8):
        super().__init__()
        self.sequeeze = nn.Linear(in_features=channels, out_features=channels // ratio, bias=False)
        self.relu = nn.ReLU()
        self.excitation = nn.Linear(in_features=channels // ratio, out_features=channels, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        gates = self.sequeeze(x)
        gates = self.relu(gates)
        gates = self.excitation(gates)
        gates = self.sigmoid(gates)
        x = torch.mul(x, gates)

        return x


class ConcatDenseSE(nn.Module):
    def __init__(self, multimodal_hidden_size, hidden_size, se_ratio, dropout):
        super().__init__()
        self.fusion = nn.Linear(multimodal_hidden_size, hidden_size)
        self.fusion_dropout = nn.Dropout(dropout)
        self.enhance = SENet(channels=hidden_size, ratio=se_ratio)
        

    def forward(self, inputs):
        embeddings = torch.cat(inputs, dim=1)
        embeddings = self.fusion_dropout(embeddings)
        embedding = self.fusion(embeddings)
        embedding = self.enhance(embedding)
        
        return embedding
    
    
