import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, VisualBertModel, AutoConfig, RobertaConfig, BertConfig

from category_id_map import CATEGORY_ID_LIST
from label_smoothing import LabelSmoothingCrossEntropy
from coatt_module import DCNLayer
import heads
from bert_model import BertCrossLayer, BertAttention
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union

from transformers import AutoTokenizer, BertTokenizer

class MaskLM(object):
    def __init__(self, tokenizer_path='bert-base-chinese', mlm_probability=0.15):
        self.mlm_probability = 0.15
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_path)

    def torch_mask_tokens(self, inputs: Any, special_tokens_mask: Optional[Any] = None) -> Tuple[Any, Any]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        labels = inputs.clone()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels

class ShuffleVideo(object):
    def __init__(self):
        pass

    def torch_shuf_video(self, video_feature):
        bs = video_feature.size()[0]
        # batch 内前一半 video 保持原顺序，后一半 video 逆序
        shuf_index = torch.tensor(list(range(bs // 2)) + list(range(bs // 2, bs))[::-1])
        # shuf 后的 label
        label = (torch.tensor(list(range(bs))) == shuf_index).float()
        video_feature = video_feature[shuf_index]
        return video_feature, label

def init_weights(module):
    if isinstance(module, (nn.Linear, nn.Embedding)):
        module.weight.data.normal_(mean=0.0, std=0.02)
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)

    if isinstance(module, nn.Linear) and module.bias is not None:
        module.bias.data.zero_()

class MultiModal(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.bert = BertModel.from_pretrained(args.bert_dir, cache_dir=args.bert_cache)

        bert_config = BertConfig(
            vocab_size=21128,
            hidden_size=768,
            num_hidden_layers=6,
            num_attention_heads=12,
            intermediate_size=768 * 4,
            max_position_embeddings=256,
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
        )
        
        self.lm = MaskLM(tokenizer_path=args.bert_dir)
        self.sv = ShuffleVideo()
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=args.frame_embedding_size, nhead=12)
        self.nextvlad = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.nextvlad.apply(init_weights)
        
        self.layernorm_text = nn.LayerNorm([256, 768])
        self.layernorm_image = nn.LayerNorm([32, 768])

        self.cross_modal_text_transform = nn.Linear(768, 768)
        self.cross_modal_image_transform = nn.Linear(768, 768)
        self.token_type_embeddings = nn.Embedding(2, 768)

        self.avgpool = nn.AdaptiveAvgPool1d(1)

        self.cross_modal_image_layers = nn.ModuleList([BertCrossLayer(bert_config) for _ in range(6)])
        self.cross_modal_text_layers = nn.ModuleList([BertCrossLayer(bert_config) for _ in range(6)])
        self.cross_modal_image_pooler = heads.Pooler(768)
        self.cross_modal_text_pooler = heads.Pooler(768)

        bert_output_size = 768

        self.classifier = nn.Sequential(
            nn.Linear(bert_output_size*2, bert_output_size),
            nn.LayerNorm(bert_output_size),
            nn.GELU(),
            nn.Linear(bert_output_size, len(CATEGORY_ID_LIST)),
        )
        self.classifier.apply(init_weights)

        self.mlm_score = heads.MLMHead(bert_config)
        self.mlm_score.apply(init_weights)

        self.itm_score = heads.ITMHead(bert_output_size*2)
        self.itm_score.apply(init_weights)


    def forward(self, inputs, inference=False, do_mlm=False, do_itm=False):
        
        if do_mlm:
            input_ids, lm_label = self.lm.torch_mask_tokens(inputs['title_input'].cpu())
            input_ids = input_ids.to(inputs['title_input'].device)
            lm_label = lm_label.to(inputs['title_input'].device)
        else:
            input_ids = inputs['title_input']
        
        if do_itm:
            image_embeds, video_text_match_label = self.sv.torch_shuf_video(inputs['frame_input'].cpu())
            image_embeds = image_embeds.to(inputs['frame_input'].device)
            video_text_match_label = video_text_match_label.to(image_embeds.device)
        else:
            image_embeds = inputs['frame_input']
                    
        
        text_embeds = self.bert(input_ids, inputs['title_mask'])
        text_embeds = text_embeds['last_hidden_state']

        # layernorm
        text_embeds = self.layernorm_text(text_embeds)
        image_embeds = self.layernorm_image(image_embeds)
        
        src_key_padding_mask = inputs['frame_mask']==0
        image_embeds = self.nextvlad(image_embeds.permute(1, 0, 2), src_key_padding_mask=src_key_padding_mask).permute(1, 0, 2)        
        
        device = text_embeds.device
        input_shape = inputs['title_mask'].size()
        extend_text_masks = self.bert.get_extended_attention_mask(inputs['title_mask'], input_shape, device)

        text_embeds = self.cross_modal_text_transform(text_embeds)
        image_embeds = self.cross_modal_image_transform(image_embeds)
        extend_image_masks = self.bert.get_extended_attention_mask(inputs['frame_mask'], inputs['frame_mask'].size(), device)

        image_token_type_idx = 1
        text_embeds, image_embeds = (
            text_embeds + self.token_type_embeddings(torch.zeros_like(inputs['title_mask'])),
            image_embeds
            + self.token_type_embeddings(
                torch.full_like(inputs['frame_mask'], image_token_type_idx)
            ),
        )

        x, y = text_embeds, image_embeds
        for text_layer, image_layer in zip(self.cross_modal_text_layers, self.cross_modal_image_layers):
            x1 = text_layer(x, y, extend_text_masks, extend_image_masks)
            y1 = image_layer(y, x, extend_image_masks, extend_text_masks)
            x, y = x1[0], y1[0]

        text_feats, image_feats = x, y
        cls_feats_text = self.cross_modal_text_pooler(x)

        avg_image_feats = self.avgpool(image_feats.transpose(1, 2)).view(image_feats.size(0), 1, -1)
        cls_feats_image = self.cross_modal_image_pooler(avg_image_feats)

        final_embedding = torch.cat([cls_feats_text, cls_feats_image], dim=-1)

        prediction = self.classifier(final_embedding)


        
        
        if do_mlm:
            mlm_logits = self.mlm_score(text_feats)
            mlm_loss = F.cross_entropy(
                        mlm_logits.view(-1, 21128),
                        lm_label.view(-1),
                        ignore_index=-100,
                    )

        if do_itm:
            itm_logits = self.itm_score(final_embedding)
            itm_loss = nn.BCEWithLogitsLoss()(itm_logits.view(-1), video_text_match_label.view(-1))
        loss = mlm_loss + itm_loss
        return loss, mlm_loss, itm_loss

        

    @staticmethod
    def cal_loss(prediction, label):
        label = label.squeeze(dim=1)
        # loss = F.cross_entropy(prediction, label)
        loss = LabelSmoothingCrossEntropy(reduction='sum')(prediction, label)
        with torch.no_grad():
            pred_label_id = torch.argmax(prediction, dim=1)
            accuracy = (label == pred_label_id).float().sum() / label.shape[0]
        return loss, accuracy, pred_label_id, label

    @staticmethod
    def cal_loss_aux(prediction, label, prediction_aux, label_aux):
        label = label.squeeze(dim=1)
        label_aux = label_aux.squeeze(dim=1)
        # loss = F.cross_entropy(prediction, label)
        loss = (LabelSmoothingCrossEntropy(reduction='sum')(prediction, label) + LabelSmoothingCrossEntropy(reduction='sum')(prediction_aux, label_aux))/2
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

    def forward(self, inputs, mask):
        # todo mask
        # torch.Size([64, 32, 768])
        inputs = self.expansion_linear(inputs) # torch.Size([64, 32, 1536])
        attention = self.group_attention(inputs) # torch.Size([64, 32, 8])
        attention = torch.sigmoid(attention) # torch.Size([64, 32, 8])
        ## add mask
        # attention = tf.multiply(attention, tf.expand_dims(mask, -1))
        attention = torch.mul(attention, mask.unsqueeze(dim=-1))
        ##
        attention = attention.reshape([-1, inputs.size(1) * self.groups, 1]) # torch.Size([64, 256, 1])
        reshaped_input = inputs.reshape([-1, self.expansion_size * self.feature_size]) # torch.Size([2048, 1536])
        activation = self.cluster_linear(reshaped_input) # torch.Size([2048, 512])
        activation = activation.reshape([-1, inputs.size(1) * self.groups, self.cluster_size]) # torch.Size([64, 256, 64])
        activation = torch.softmax(activation, dim=-1) # torch.Size([64, 256, 64])
        activation = activation * attention # torch.Size([64, 256, 64])
        a_sum = activation.sum(-2, keepdim=True) # torch.Size([64, 1, 64])
        a = a_sum * self.cluster_weight  # torch.Size([64, 192, 64])
        activation = activation.permute(0, 2, 1).contiguous()  # torch.Size([64, 64, 256])
        reshaped_input = inputs.reshape([-1, inputs.shape[1] * self.groups, self.new_feature_size]) # torch.Size([64, 256, 192])
        vlad = torch.matmul(activation, reshaped_input) # torch.Size([64, 64, 192])
        vlad = vlad.permute(0, 2, 1).contiguous() # torch.Size([64, 192, 64])
        vlad = F.normalize(vlad - a, p=2, dim=1) # torch.Size([64, 192, 64])
        vlad = vlad.reshape([-1, self.cluster_size * self.new_feature_size]) # torch.Size([64, 12288])
        vlad = self.dropout(vlad)
        vlad = self.fc(vlad)  # torch.Size([64, 1024])
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
