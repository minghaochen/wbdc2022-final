#%%writefile qqmodel/qq_uni_model.py
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.append("..")

from models.masklm import MaskLM, MaskVideo, ShuffleVideo
from transformers.models.bert.modeling_bert import BertConfig, BertOnlyMLMHead
from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertEmbeddings, BertEncoder
from .swin import swin_tiny

def init_params(module_lst):
    for module in module_lst:
        for param in module.parameters():
            if param.dim() > 1:
                torch.nn.init.xavier_uniform_(param)
    return

class QQUniModel(nn.Module):
    def __init__(self, args, model_path, task=['mlm', 'mfm'], init_from_pretrain=True):
        super().__init__()
        
        uni_bert_cfg = BertConfig.from_pretrained(f'{model_path}/config.json')    
        self.task = set(task)
        self.visual_backbone = swin_tiny(args.swin_pretrained_path)

        if 'mlm' in task:
            self.lm = MaskLM(tokenizer_path=model_path)
            self.vocab_size = uni_bert_cfg.vocab_size
        
        if 'mfm' in task:
            self.vm = MaskVideo()
            self.roberta_mvm_lm_header = VisualOnlyMLMHead(uni_bert_cfg) 
            
        if 'itm' in task:
            self.sv = ShuffleVideo()
            self.newfc_itm = torch.nn.Linear(uni_bert_cfg.hidden_size, 1) 

        if init_from_pretrain:
            self.roberta = UniBertForMaskedLM.from_pretrained(model_path, config=uni_bert_cfg)
        else:
            self.roberta = UniBertForMaskedLM(uni_bert_cfg)

    def forward(self, inputs, task=None):
        
        inputs['frame_input'] = self.visual_backbone(inputs['frame_input'])
        
        video_feature, video_mask = inputs['frame_input'], inputs['frame_mask']
        text_input_ids, text_mask = inputs['text_input'], inputs['text_mask']
        
        loss, pred = 0, None
        
        if task is None:
            sample_task = self.task
        elif type(task) == str:
            sample_task = [task]
        elif type(task) == list:
            sample_task = task
        
        # perprocess
        return_mlm = False
        if 'mlm' in sample_task:
            input_ids, lm_label = self.lm.torch_mask_tokens(text_input_ids.cpu())
            text_input_ids = input_ids.to(text_input_ids.device)
            lm_label = lm_label[:, 1:].to(text_input_ids.device) # [SEP] 卡 MASK 大师 [SEP]
            return_mlm = True
            
        if 'mfm' in sample_task:
            vm_input = video_feature # b 32 768
            input_feature, video_label = self.vm.torch_mask_frames(video_feature, video_mask)
            video_feature = input_feature.to(video_feature.device)
            video_label = video_label.to(video_feature.device)
            
        if 'itm' in sample_task:
            input_feature, video_text_match_label = self.sv.torch_shuf_video(video_feature.cpu())
            video_feature = input_feature.to(video_feature.device)
            video_text_match_label = video_text_match_label.to(video_feature.device)
            
        # concat features
        encoder_outputs, lm_prediction_scores, mask = self.roberta(video_feature, video_mask, text_input_ids, text_mask, return_mlm=return_mlm)
        features = encoder_outputs['last_hidden_state']
        
        mlm_loss, itm_loss = -1, -1
        # compute loss
        if 'mlm' in sample_task:
            pred = lm_prediction_scores.contiguous().view(-1, self.vocab_size)
            mlm_loss = nn.CrossEntropyLoss()(pred, lm_label.contiguous().view(-1))
            loss += mlm_loss / len(sample_task)
            # loss += mlm_loss / 1.25 / len(sample_task)
            
        if 'mfm' in sample_task:
            vm_output = self.roberta_mvm_lm_header(features[:, 1:video_feature.size()[1] + 1, :])
            mfm_loss = self.calculate_mfm_loss(vm_output, vm_input, 
                                                     video_mask, video_label, normalize=False)
            loss += mfm_loss / len(sample_task)
            # loss += mfm_loss  / 3 / len(sample_task)
            
        if 'itm' in sample_task:
            pred = self.newfc_itm(features[:, 0, :])
            itm_loss = nn.BCEWithLogitsLoss(reduce=True, reduction='mean')(pred.view(-1), video_text_match_label.view(-1))
            loss += itm_loss / len(sample_task)
            # loss += itm_loss / 100 / len(sample_task)

        return (pred, loss, mlm_loss, itm_loss, encoder_outputs)
    
    def calculate_mfm_loss(self, video_feature_output, video_feature_input, 
                           video_mask, video_labels_index, normalize=False, temp=0.1):
        if normalize:
            video_feature_output = torch.nn.functional.normalize(video_feature_output, p=2, dim=2)
            video_feature_input = torch.nn.functional.normalize(video_feature_input, p=2, dim=2)

        afm_scores_tr = video_feature_output.view(-1, video_feature_output.shape[-1]) # b*n, 768

        video_tr = video_feature_input.permute(2, 0, 1) # 768, b, n
        video_tr = video_tr.view(video_tr.shape[0], -1)
#         print(afm_scores_tr.shape, video_tr.shape)
        logits_matrix = torch.mm(afm_scores_tr, video_tr)
        if normalize:
            logits_matrix = logits_matrix / temp

        video_mask_float = video_mask.to(dtype=torch.float)
        mask_matrix = torch.mm(video_mask_float.view(-1, 1), video_mask_float.view(1, -1))
        masked_logits = logits_matrix + (1. - mask_matrix) * -1e8

        logpt = F.log_softmax(masked_logits, dim=-1)
        logpt = torch.diag(logpt)
        nce_loss = -logpt

        video_labels_index_mask = (video_labels_index != -100)
        nce_loss = nce_loss.masked_select(video_labels_index_mask.view(-1))
        nce_loss = nce_loss.mean()
        return nce_loss

def gelu(x):
    """Implementation of the gelu activation function.
    For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
    0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

def swish(x):
    return x * torch.sigmoid(x)

ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": swish}

class VisualPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states

class VisualLMPredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transform = VisualPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(config.hidden_size, 768, bias=False)
        self.bias = nn.Parameter(torch.zeros(768))

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states

class VisualOnlyMLMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = VisualLMPredictionHead(config)

    def forward(self, sequence_output):
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores
    
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
            return encoder_outputs, self.cls(encoder_outputs['last_hidden_state'])[:, video_size[1] + 1: , :], mask
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

class UniBert3(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.embeddings = VideoTextEmbedding(config)
        
        # self.video_dense = nn.Linear(768, 768)
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
        # video_feature = self.video_dense(video_feature)
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