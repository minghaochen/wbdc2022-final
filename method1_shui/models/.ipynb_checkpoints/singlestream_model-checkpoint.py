import torch
import torch.nn as nn
import torch.nn.functional as F
from category_id_map import CATEGORY_ID_LIST
from .swin import swin_tiny

from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertEmbeddings, BertEncoder

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

class SingleStream_model(nn.Module):
    def __init__(self, args):
        super().__init__()
        
        self.bert = Bert_encoder.from_pretrained(args.bert_dir, cache_dir=args.bert_cache)
        self.my_classifer = myClassifer()
        # self.my_classifer = MeanPooling()
        self.drop = nn.Dropout(p=0.2)
        self.visual_backbone = swin_tiny(args.swin_pretrained_path)
        
        # self.dropout = nn.Dropout(0.1)
        # self.dropout1 = nn.Dropout(0.1)
        # self.dropout2 = nn.Dropout(0.2)
        # self.dropout3 = nn.Dropout(0.3)
        # self.dropout4 = nn.Dropout(0.4)
        # self.dropout5 = nn.Dropout(0.5)
        # 线性层
        self.classify_dense = nn.Linear(768, len(CATEGORY_ID_LIST))
        
        # self.my_classifer.apply(init_weights)
        self.classify_dense.apply(init_weights)

    def forward(self, inputs, inference=False):
        
        inputs['frame_input'] = self.visual_backbone(inputs['frame_input'])
        encoder_outputs, mask = self.bert(inputs)
        output = self.my_classifer(encoder_outputs['hidden_states'])
        
        # output = self.my_classifer(encoder_outputs['hidden_states'])
        # prediction1 = self.classify_dense(self.dropout1(output))
        # prediction2 = self.classify_dense(self.dropout2(output))
        # prediction3 = self.classify_dense(self.dropout3(output))
        # prediction4 = self.classify_dense(self.dropout4(output))
        # prediction5 = self.classify_dense(self.dropout5(output))
        # prediction = (prediction1 + prediction2 + prediction3 + prediction4 + prediction5) / 5
        
        # prediction = self.drop(output)
        prediction = self.classify_dense(output)

        if inference:
            return prediction
        else:
            loss, accuracy, pred_label_id, label = self.cal_loss(prediction, inputs['label'])
            return loss, accuracy, pred_label_id, label, prediction

    @staticmethod
    def cal_loss(prediction, label):
        label = label.squeeze(dim=1)
        # loss = LabelSmoothingCrossEntropy(epsilon=0.1)(prediction, label)
        loss = F.cross_entropy(prediction, label)  # label smooth
        # loss = F.cross_entropy(prediction, label, label_smoothing=0.1)  # label smooth
        with torch.no_grad():
            pred_label_id = torch.argmax(prediction, dim=1)
            accuracy = (label == pred_label_id).float().sum() / label.shape[0]
        return loss, accuracy, pred_label_id, label


class Bert_encoder(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.embeddings = BertEmbeddings(config)
        self.video_dense = nn.Linear(768, 768)
        self.encoder = BertEncoder(config)

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(self, inputs):
        frame_input, frame_mask, frame_token_type_ids = inputs['frame_input'], inputs['frame_mask'], inputs[
            'frame_token_type_ids']
        text_input, text_mask, text_token_type_ids = inputs['text_input'], inputs['text_mask'], inputs[
            'text_token_type_ids']

        text_emb = self.embeddings(input_ids=text_input)
        # text input is [CLS][SEP]text[SEP]
        cls_emb = text_emb[:, 0:1, :]
        text_emb = text_emb[:, 1:, :]

        cls_mask = text_mask[:, 0:1]
        text_mask = text_mask[:, 1:]

        frame_input = self.video_dense(frame_input)
        frame_emb = self.embeddings(inputs_embeds=frame_input)

        # [CLS] Video [SEP] Text [SEP]
        embedding_output = torch.cat([cls_emb, frame_emb, text_emb], 1)

        mask = torch.cat([cls_mask, frame_mask, text_mask], 1)
        extended_mask = mask[:, None, None, :]
        extended_mask = (1.0 - extended_mask) * -10000.0

        encoder_outputs = self.encoder(embedding_output, attention_mask=extended_mask, output_hidden_states=True)
        return encoder_outputs, mask
    
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