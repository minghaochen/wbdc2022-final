import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, VisualBertModel, AutoConfig, RobertaConfig, BertConfig

from category_id_map import CATEGORY_ID_LIST
from label_smoothing import LabelSmoothingCrossEntropy
from coatt_module import DCNLayer
import heads
from bert_model import BertCrossLayer, BertAttention
from swin import swin_tiny
import timm


def get_extended_attention_mask(
    attention_mask, input_shape, device, dtype=None
):
    """
    Makes broadcastable attention and causal masks so that future and masked tokens are ignored.
    Arguments:
        attention_mask (`torch.Tensor`):
            Mask with ones indicating tokens to attend to, zeros for tokens to ignore.
        input_shape (`Tuple[int]`):
            The shape of the input to the model.
    Returns:
        `torch.Tensor` The extended attention mask, with a the same dtype as `attention_mask.dtype`.
    """
    if dtype is None:
        dtype = torch.float32

    # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
    # ourselves in which case we just need to make it broadcastable to all heads.
    if attention_mask.dim() == 3:
        extended_attention_mask = attention_mask[:, None, :, :]
    elif attention_mask.dim() == 2:
        # Provided a padding mask of dimensions [batch_size, seq_length]
        # - if the model is a decoder, apply a causal mask in addition to the padding mask
        # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
        extended_attention_mask = attention_mask[:, None, None, :]
    else:
        raise ValueError(
            f"Wrong shape for input_ids (shape {input_shape}) or attention_mask (shape {attention_mask.shape})"
        )

    extended_attention_mask = extended_attention_mask.to(dtype=dtype)  # fp16 compatibility
    extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(dtype).min
    return extended_attention_mask


def init_weights(module):
    if isinstance(module, (nn.Linear, nn.Embedding)):
        module.weight.data.normal_(mean=0.0, std=0.02)
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)

    if isinstance(module, nn.Linear) and module.bias is not None:
        module.bias.data.zero_()

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
    
class MultiModal(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.bert = BertModel.from_pretrained(args.bert_dir, cache_dir=args.bert_cache)
        # self.bert = BertModel.from_pretrained(args.bert_dir, cache_dir=args.bert_cache).embeddings
        #bert 复赛考虑只用6层
        # bert_config = BertConfig(
        #     vocab_size=21128,
        #     num_hidden_layers=1,
        # )
        # bert_config = BertConfig.from_pretrained(args.bert_dir)
        # bert_config.num_hidden_layers = 0
        # self.bert = BertModel.from_pretrained(args.bert_dir,
        #                                      config = bert_config)
        print(self.bert)
        
        
        # swin tiny
        self.visual_backbone = swin_tiny(args.swin_pretrained_path)
        # 换backbone
        # self.visual_backbone = timm.create_model('swin_base_patch4_window7_224_in22k', pretrained=False, num_classes=0)
        # checkpoint = torch.load('./opensource_models/swin_base_patch4_window7_224_22k.pth', map_location='cpu')
        # self.visual_backbone.load_state_dict(checkpoint, strict=False)
        # self.visual_backbone_proj = nn.Linear(1024,768)
        # self.visual_backbone_proj.apply(init_weights)
        
        
        
        # visual_backbone = timm.create_model('tf_efficientnetv2_s_in21ft1k', pretrained=False, num_classes=0)
        # checkpoint = torch.load('/home/tione/notebook/opensource_models/effecientnet/tf_efficientnetv2_s_21ft1k-d7dafa41.pth', map_location='cpu')
        #
        # visual_backbone = timm.create_model('efficientnet_b3_pruned', pretrained=False, num_classes=0)
        # checkpoint = torch.load('opensource_models/effecientnet/effnetb3_pruned_5abcc29f.pth', map_location='cpu')
        # visual_backbone.load_state_dict(checkpoint, strict=False)
        # self.visual_backbone = visual_backbone
        # self.visual_proj = nn.Linear
        
        # self.nextvlad = NeXtVLAD(args.frame_embedding_size, args.vlad_cluster_size,
        #                          output_size=args.vlad_hidden_size, dropout=args.dropout)

        # encoder_layer = nn.TransformerEncoderLayer(d_model=args.frame_embedding_size, nhead=12)
        # self.nextvlad = nn.TransformerEncoder(encoder_layer, num_layers=2)
        # # self.nextvlad = nn.TransformerEncoderLayer(d_model=args.frame_embedding_size, nhead=8)
        # self.nextvlad.apply(init_weights)

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
        
        # self.layernorm_text = nn.LayerNorm([256, 768])
        # self.layernorm_image = nn.LayerNorm([32, 768])


        # self.cross_modal_text_transform = nn.Linear(768, 768)
        # self.cross_modal_text_transform.apply(init_weights)
        # self.cross_modal_image_transform = nn.Linear(768, 768)
        # effecientnet
        # self.cross_modal_image_transform = nn.Linear(1536, 768)
        
        # self.cross_modal_image_transform.apply(init_weights)
        self.token_type_embeddings = nn.Embedding(2, 768)
        self.token_type_embeddings.apply(init_weights)
        
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.avgpool.apply(init_weights)
        
        # self.cross_modal_image_layers = nn.ModuleList([BertCrossLayer(bert_config) for _ in range(6)])
        # self.cross_modal_text_layers = nn.ModuleList([BertCrossLayer(bert_config) for _ in range(6)])
        # 复赛少点层
        self.cross_modal_image_layers = nn.ModuleList([BertCrossLayer(bert_config) for _ in range(1)])
        self.cross_modal_text_layers = nn.ModuleList([BertCrossLayer(bert_config) for _ in range(1)])
        
        
        # self.cross_modal_image_pooler = heads.Pooler(768)
        # self.cross_modal_text_pooler = heads.Pooler(768)
        self.cross_modal_image_layers.apply(init_weights)
        self.cross_modal_text_layers.apply(init_weights)
        # self.cross_modal_image_pooler.apply(init_weights)
        # self.cross_modal_text_pooler.apply(init_weights)
        
        self.cross_modal_text_pooler = MeanPooling()


        # self.frame_map = nn.Sequential(
        #     nn.Linear(args.frame_embedding_size, 512),
        #     nn.GELU()
        # )
        # self.frame_map.apply(init_weights)

        # config = AutoConfig.from_pretrained('uclanlp/visualbert-vqa-coco-pre')
        # config.update(
        #     {
        #         "num_hidden_layers": 12,
        #         # "visual_embedding_dim": 768,
        #     }
        # )
        # self.vlbert = VisualBertModel.from_pretrained('uclanlp/visualbert-vqa-coco-pre',config=config)
        # self.vlbert = VisualBertModel.from_pretrained(args.bert_dir)


        # self.enhance = SENet(channels=args.vlad_hidden_size, ratio=args.se_ratio)
        # self.enhance.apply(init_weights)
        bert_output_size = 768
        # self.fusion = ConcatDenseSE(args.vlad_hidden_size + bert_output_size, args.fc_size, args.se_ratio, args.dropout)
        # self.fusion.apply(init_weights)
        # self.coatt = DCNLayer(dim1=bert_output_size, dim2=args.frame_embedding_size, num_attn=12, num_none=3, num_seq=5, dropout=0.1)
        # self.coatt.apply(init_weights)
        # self.classifier = nn.Linear(args.fc_size, len(CATEGORY_ID_LIST))
        # self.classifier.apply(init_weights)

        self.classifier2 = nn.Sequential(
            # nn.BatchNorm1d(bert_output_size),
#             nn.Linear(bert_output_size*2, bert_output_size),
#             nn.LayerNorm(bert_output_size),
#             nn.GELU(),
            nn.Linear(bert_output_size*2 + 128, len(CATEGORY_ID_LIST)),
        )
        # self.classifier =nn.Linear(bert_output_size*2, len(CATEGORY_ID_LIST))
        self.classifier2.apply(init_weights)
        
        # origin 
        # self.dropout = nn.Dropout(0.1)
        # self.dropout1 = nn.Dropout(0.1)
        # self.dropout2 = nn.Dropout(0.2)
        # self.dropout3 = nn.Dropout(0.3)
        # self.dropout4 = nn.Dropout(0.4)
        # self.dropout5 = nn.Dropout(0.5)
        # multi sample from paper
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)
        self.dropout3 = nn.Dropout(0.5)
        self.dropout4 = nn.Dropout(0.5)
        self.dropout5 = nn.Dropout(0.5)
        
        
        self.class_text = nn.Linear(bert_output_size+128, len(CATEGORY_ID_LIST))
        self.class_video = nn.Linear(bert_output_size+128, len(CATEGORY_ID_LIST))
        self.class_text.apply(init_weights)
        self.class_video.apply(init_weights)
        
        self.avgpool2 = nn.AdaptiveAvgPool1d(1)
        self.avgpool2.apply(init_weights)
        self.cross_modal_text_pooler2 = MeanPooling()
        

        # self.classifier_aux = nn.Linear(args.fc_size, 23)
        # self.classifier = nn.Linear(23, len(CATEGORY_ID_LIST))

        # # fusion
        # self.Linear_dataproj = nn.Linear(bert_output_size, bert_output_size*5)
        # self.Linear_imgproj = nn.Linear(args.vlad_hidden_size, args.vlad_hidden_size*5)
        # self.Linear_predict = nn.Linear(bert_output_size, args.fc_size)
        # self.drop = nn.Dropout(0.1)
        # self.bert_output_size = bert_output_size

        # creation
        # self.loss = LabelSmoothingCrossEntropy(reduction='sum')
        
        self.label_emb_text = torch.nn.Parameter(
            torch.nn.init.normal_(torch.rand(23, 128), std=0.01))
        self.label_emb_video = torch.nn.Parameter(
            torch.nn.init.normal_(torch.rand(23, 128), std=0.01))
        
        self.label_emb = torch.nn.Parameter(
            torch.nn.init.normal_(torch.rand(23, 128), std=0.01))
        self.class_text_level_1 = nn.Linear(bert_output_size, 23)
        self.class_video_level_1 = nn.Linear(bert_output_size, 23)
        self.class_text_level_1.apply(init_weights)
        self.class_video_level_1.apply(init_weights)
        self.class_text_softmax = torch.nn.Softmax(dim=1)
        self.class_videl_softmax = torch.nn.Softmax(dim=1)
        
        self.final_class_level_1 = nn.Linear(bert_output_size*2, 23)
        self.final_class_level_1.apply(init_weights)
        self.final_level_1_softmax = torch.nn.Softmax(dim=1)
        

    def forward(self, inputs, inference=False):
        # print(inputs["title_input"].shape)
        # print(inputs["title_mask"].shape)
        # print(inputs["frame_input"].shape)
        # print(inputs["frame_mask"].shape)
        
        # # swin base
        # B, N, C, H, W = inputs['frame_input'].shape
        # output_shape = (B, N, -1)
        # inputs['frame_input'] = self.visual_backbone(inputs['frame_input'].view(B * N, C, H, W)).view(*output_shape)
        # inputs['frame_input'] = self.visual_backbone_proj(inputs['frame_input'])
        
        # swin tiny
        inputs['frame_input'] = self.visual_backbone(inputs['frame_input'])
        

        text_embeds = self.bert(inputs['title_input'], inputs['title_mask'])
        text_embeds = text_embeds['last_hidden_state']
        # only use embedding layer
        # text_embeds = self.bert(inputs['title_input'])
        # print(text_embeds.shape)
        
        # level1 pred label embed
        text_level_1_input = self.cross_modal_text_pooler2(text_embeds, inputs['title_mask'])
        subtask1_pred_level_1 = self.class_text_level_1(text_level_1_input)
        label_emb_text_level_1 = torch.mm(self.class_text_softmax(subtask1_pred_level_1), self.label_emb_text)
        text_level_2_input = torch.cat([text_level_1_input, label_emb_text_level_1], dim=-1)
        subtask1_pred = self.class_text(text_level_2_input)
        
        video_level_1_input = self.avgpool2(inputs['frame_input'].transpose(1, 2)).squeeze(2)
        subtask2_pred_level_1 = self.class_video_level_1(video_level_1_input)
        label_emb_video_level_1 = torch.mm(self.class_videl_softmax(subtask2_pred_level_1), self.label_emb_video)
        video_level_2_input = torch.cat([video_level_1_input, label_emb_video_level_1], dim=-1)
        subtask2_pred = self.class_video(video_level_2_input)
        
        
        # subtask1_pred = self.class_text()
        # subtask2_pred = self.class_video(self.avgpool2(inputs['frame_input'].transpose(1, 2)).squeeze(2))
        
        # print(subtask1_pred.shape)
        # print(subtask2_pred.shape)
        # a += 1 
        
        
        # layer
        # text_embeds = self.layernorm_text(text_embeds)
        image_embeds = inputs['frame_input']
        # image_embeds = self.layernorm_image(image_embeds)
        
        # src_key_padding_mask = inputs['frame_mask']==0
        # image_embeds = self.nextvlad(image_embeds.permute(1, 0, 2), src_key_padding_mask=src_key_padding_mask).permute(1, 0, 2)               
        

        device = text_embeds.device
        input_shape = inputs['title_mask'].size()
        extend_text_masks = self.bert.get_extended_attention_mask(inputs['title_mask'], input_shape, device)
        # extend_text_masks = get_extended_attention_mask(inputs['title_mask'], input_shape, device)

        # text_embeds = self.cross_modal_text_transform(text_embeds)
        # image_embeds = self.cross_modal_image_transform(image_embeds)
        extend_image_masks = self.bert.get_extended_attention_mask(inputs['frame_mask'], inputs['frame_mask'].size(), device)
        # extend_image_masks = get_extended_attention_mask(inputs['frame_mask'], inputs['frame_mask'].size(), device)

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
        
        # cls_feats_text = self.cross_modal_text_pooler(x)
        # avg_image_feats = self.avgpool(image_feats.transpose(1, 2)).view(image_feats.size(0), 1, -1)
        # cls_feats_image = self.cross_modal_image_pooler(avg_image_feats)
        
        # zhijie mean
        cls_feats_text = self.cross_modal_text_pooler(x, inputs['title_mask'])
        cls_feats_image = self.avgpool(image_feats.transpose(1, 2)).squeeze(2)
        # print(cls_feats_text.shape)
        # print(cls_feats_image.shape)
        
        # origin
        # final_embedding = torch.cat([cls_feats_text, cls_feats_image], dim=-1)
        
        final_level_1_input = torch.cat([cls_feats_text, cls_feats_image], dim=-1)
        final_pred_level_1 = self.final_class_level_1(final_level_1_input)
        label_emb_final_level_1 = torch.mm(self.final_level_1_softmax(final_pred_level_1), self.label_emb)
        final_embedding = torch.cat([cls_feats_text, cls_feats_image, label_emb_final_level_1], dim=-1)

        
        
        
        

        # anchor_len = inputs['title_mask'].unsqueeze(2)
        # bert_embedding = bert_embedding * anchor_len
        # bert_embedding = torch.sum(bert_embedding, 1) / torch.sum(anchor_len, dim=1)

        # vision_embedding = self.nextvlad(inputs['frame_input'], inputs['frame_mask'])
        # vision_embedding = self.nextvlad(inputs['frame_input'].permute(1, 0, 2)).permute(1, 0, 2)
        # anchor_len = inputs['frame_mask'].unsqueeze(2)
        # vision_embedding = vision_embedding * anchor_len
        # vision_embedding = torch.sum(vision_embedding, 1) / torch.sum(anchor_len, dim=1)
        # vision_embedding = self.enhance(vision_embedding)


        # vlbert
        # bert_embedding = self.bert(input_ids=inputs['title_input'])
        # token_type_ids = torch.zeros(bert_embedding.shape[:-1], dtype=torch.long).cuda()
        # visual_embeds = self.frame_map(inputs['frame_input'])
        # visual_token_type_ids = torch.ones(visual_embeds.shape[:-1], dtype=torch.long).cuda()
        #
        #
        # final_embedding = self.vlbert(inputs_embeds=bert_embedding,
        #                               attention_mask=inputs['title_mask'],
        #                               token_type_ids=token_type_ids,
        #                               visual_embeds=visual_embeds,
        #                               visual_token_type_ids=visual_token_type_ids,
        #                               visual_attention_mask=inputs['frame_mask']
        #                               )
        #
        # final_embedding = final_embedding['last_hidden_state']
        # mask = torch.cat((inputs['title_mask'], inputs['frame_mask']),dim=1)
        # anchor_len = mask.unsqueeze(2)
        # final_embedding = final_embedding * anchor_len
        # final_embedding = torch.sum(final_embedding, 1) / torch.sum(anchor_len, dim=1)





        # coattention
        # vision_embedding = inputs['frame_input']
        # bert_embedding, vision_embedding = self.coatt(bert_embedding, vision_embedding,
        #                                               mask1=inputs['title_mask'], mask2=inputs['frame_mask'])
        # anchor_len = inputs['title_mask'].unsqueeze(2)
        # bert_embedding = bert_embedding * anchor_len
        # bert_embedding = torch.sum(bert_embedding, 1) / torch.sum(anchor_len, dim=1)
        # anchor_len = inputs['frame_mask'].unsqueeze(2)
        # vision_embedding = vision_embedding * anchor_len
        # vision_embedding = torch.sum(vision_embedding, 1) / torch.sum(anchor_len, dim=1)

        ## fusion
        # data_out = self.Linear_dataproj(bert_embedding)  # data_out (batch, 5000)
        # img_feature = self.Linear_imgproj(vision_embedding)  # img_feature (batch, 5000)
        # iq = torch.mul(data_out, img_feature)
        # iq = self.drop(iq)
        # iq = iq.view(-1, 1, self.bert_output_size, 5)
        # iq = torch.squeeze(torch.sum(iq, 3))  # sum pool
        # iq = torch.sqrt(F.relu(iq)) - torch.sqrt(F.relu(-iq))  # signed sqrt
        # iq = F.normalize(iq)
        # final_embedding = self.Linear_predict(iq)  # (64,3000)


        # ORIGIN
#         prediction = self.classifier(final_embedding)
        # multi sample
        # final_embedding = self.dropout(final_embedding)
        # prediction1 = self.classifier2(self.dropout1(final_embedding))
        # prediction2 = self.classifier2(self.dropout2(final_embedding))
        # prediction3 = self.classifier2(self.dropout3(final_embedding))
        # prediction4 = self.classifier2(self.dropout4(final_embedding))
        # prediction5 = self.classifier2(self.dropout5(final_embedding))
        # prediction = (prediction1 + prediction2 + prediction3 + prediction4 + prediction5) / 5
        # from paper
        prediction1 = self.classifier2(self.dropout1(final_embedding))
        prediction2 = self.classifier2(self.dropout2(final_embedding))
        prediction3 = self.classifier2(self.dropout3(final_embedding))
        prediction4 = self.classifier2(self.dropout4(final_embedding))
        prediction5 = self.classifier2(self.dropout5(final_embedding))
        prediction = prediction1
        # prediction = (prediction1 + prediction2 + prediction3 + prediction4 + prediction5) / 5


        # prediction_aux = self.classifier_aux(final_embedding)
        # prediction = self.classifier(prediction_aux)

        if inference:
            return torch.argmax(prediction, dim=1), prediction
        else:
            return self.cal_loss(prediction, inputs['label'], subtask1_pred, subtask2_pred,
                                prediction1,prediction2,prediction3,prediction4,prediction5,
                                subtask1_pred_level_1, subtask2_pred_level_1, final_pred_level_1,
                                inputs['label_aux'])
            # return self.cal_loss(prediction, inputs['label'])
            # return self.cal_loss_aux(prediction, inputs['label'], prediction_aux, inputs['label_aux'])

    @staticmethod
    def cal_loss(prediction, label, subtask1_pred, subtask2_pred,
                prediction1,prediction2,prediction3,prediction4,prediction5,
                subtask1_pred_level_1, subtask2_pred_level_1, final_pred_level_1, label_aux):
        label = label.squeeze(dim=1)
        # loss = F.cross_entropy(prediction, label)
        # LabelSmoothingCrossEntropy(reduction='sum')
        loss_main1 = F.cross_entropy(prediction1, label)
        loss_main2 = F.cross_entropy(prediction2, label)
        loss_main3 = F.cross_entropy(prediction3, label)
        loss_main4 = F.cross_entropy(prediction4, label)
        loss_main5 = F.cross_entropy(prediction5, label)
        loss_main = (loss_main1 + loss_main2 + loss_main3 + loss_main4 + loss_main5)/5
        
        loss_sub1 = F.cross_entropy(subtask1_pred, label)
        loss_sub2 = F.cross_entropy(subtask2_pred, label)
        
        label_aux = label_aux.squeeze(dim=1)
        loss_main_level_1 = F.cross_entropy(final_pred_level_1, label_aux)
        loss_sub1_level_1 = F.cross_entropy(subtask1_pred_level_1, label_aux)
        loss_sub2_level_1 = F.cross_entropy(subtask2_pred_level_1, label_aux)
        
        loss = loss_main+0.25*loss_sub1+0.25*loss_sub2 + 0.5*(loss_main_level_1 + 0.25*loss_sub1_level_1 + 0.25*loss_sub2_level_1)# TODO: KLD
        with torch.no_grad():
            pred_label_id = torch.argmax(prediction, dim=1)
            accuracy = (label == pred_label_id).float().sum() / label.shape[0]
        return loss, accuracy, pred_label_id, label, prediction

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
