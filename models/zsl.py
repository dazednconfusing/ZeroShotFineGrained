from typing import Any

import torch
import torch.nn.init as init
from torch import distributed as dist
from torch import nn
from transformers import AutoModel

from models.lstm import LSTM
from models.zest import Attention

transformer_map = {
    "bert": "bert-base-uncased",
    "sbert_cls": "sentence-transformers/bert-base-nli-cls-token",
    "sbert_mean": "sentence-transformers/bert-base-nli-mean-tokens",
    "xlnet": "xlnet-base-cased",
    "distilbert": "distilbert-base-uncased",
    "roberta": "roberta-base",
}


class PointwiseFFN(nn.Module):
    def __init__(self, opt, in_dim=768):
        super(PointwiseFFN, self).__init__()
        self.soft_max = nn.Softmax(dim=-1)
        self.clusters = opt.nclusters
        self.dropout = nn.Dropout(opt.ff_dropout) if opt.ff_dropout else nn.Identity()
        self.bn = nn.BatchNorm1d(in_dim) if opt.ff_bn else nn.Identity()
        self.act = nn.ReLU() if opt.ff_hidden_dim else nn.Identity()
        self.ff_hidden_dim = opt.ff_hidden_dim
        if opt.ff_hidden_dim:
            self.ffn = nn.Linear(in_dim, opt.ff_hidden_dim)
            self.ffn2 = nn.Linear(opt.ff_hidden_dim, opt.nclusters)
        else:
            self.ffn = nn.Linear(in_dim, opt.nclusters)

    def forward(self, x):
        classes, words, embed_dim = x.size()
        x = x.reshape((classes * words, embed_dim))
        x = self.dropout(x)
        x = self.bn(x)
        weights = self.ffn(x)
        if self.ff_hidden_dim:
            weights = self.act(weights)
            weights = self.ffn2(weights)
        weights = weights.reshape((classes, words, self.clusters)).transpose(1, 2)
        weights = self.soft_max(weights)

        # weights is now (classes, clusters, words)
        x = x.reshape(
            (classes, words, embed_dim)
        )  # .unsqueeze(1).expand(-1,self.clusters,words, embed_dim)

        # x is now(classes, words, embed_dim)
        clusters = torch.bmm(weights, x).squeeze()
        return clusters.reshape((classes, self.clusters * embed_dim))


# class GloballyCluster(nn.Module):
#     def __init__(self, opt, in_dim=768):
#         super(GloballyCluster, self).__init__()
#         self.soft_max = nn.Softmax(dim=0)
#         self.clusters = opt.nclusters
#         self.dropout = nn.Dropout(opt.ff_dropout) if opt.ff_dropout else nn.Identity()
#         self.bn = nn.BatchNorm1d(in_dim) if opt.ff_bn else nn.Identity()
#         self.act = nn.ReLU() if opt.act else nn.Identity()
#         self.ff_hidden_dim = opt.ff_hidden_dim
#         if opt.ff_hidden_dim:
#             self.ffn = nn.Linear(in_dim, opt.ff_hidden_dim)
#             self.ffn2 = nn.Linear(opt.ff_hidden_dim, opt.nclusters)
#         else:
#             self.ffn = nn.Linear(in_dim, opt.nclusters)

#     def forward(self, x):
#         classes, words, embed_dim = x.size()
#         x = x.reshape((classes * words, embed_dim))
#         x = self.dropout(x)
#         x = self.bn(x)
#         weights = self.ffn(x)
#         if self.ff_hidden_dim:
#             weights = self.act(weights)
#             weights = self.ffn2(weights)
#         weights = self.soft_max(weights)
#         weights = weights.reshape((classes, words, self.clusters)).transpose(1, 2)

#         # weights is now (classes, clusters, words)
#         x = x.reshape(
#             (classes, words, embed_dim)
#         )  # .unsqueeze(1).expand(-1,self.clusters,words, embed_dim)

#         # x is now(classes, words, embed_dim)
#         clusters = torch.bmm(weights, x).squeeze()
#         return clusters.reshape((classes, self.clusters * embed_dim))


class Transformer(nn.Module):
    def __init__(self, opt, in_dim=768):
        super(Transformer, self).__init__()
        self.transformer = nn.TransformerEncoderLayer(
            in_dim,
            opt.trans_nhead,
            dim_feedforward=opt.trans_hidden_dim,
            dropout=opt.trans_dropout,
            batch_first=True,
        )
        self.soft_max = nn.Softmax(dim=-2)
        # self.dropout = nn.Dropout(opt.ff_dropout) if opt.trans_dropout else nn.Identity()
        # self.bn = nn.BatchNorm1d(in_dim) if opt.trans_bn else nn.Identity()

    def forward(self, x):
        x = self.transformer(x).mean(dim=-1)

        classes, words, embed_dim = x.size()
        # x = x.reshape((classes * words, embed_dim))
        # x = self.dropout(x)
        # x = self.bn(x)
        weights = self.transformer(x)
        # weights = weights.reshape((classes, words, self.clusters)).transpose(1, 2)
        weights = self.soft_max(weights)

        # weights is now (classes, clusters, words)
        x = x.reshape(
            (classes, words, embed_dim)
        )  # .unsqueeze(1).expand(-1,self.clusters,words, embed_dim)

        # x is now(classes, words, embed_dim)
        clusters = torch.bmm(weights, x).squeeze()
        return clusters.reshape((classes, self.clusters * embed_dim))


class ZSL(nn.Module):
    def __init__(self, opt, embedder=None, pooler=None):
        super(ZSL, self).__init__()
        unfrozen_layers = opt.nunfrozen
        model = opt.txt_feat
        # embed_pooling = opt.token_pooling
        # print(model, embed_pooling, text_output_dim)
        # self.embed_pooling = embed_pooling
        # self.proto = opt.proto
        if opt.nunfrozen > 0:
            if embedder is None:
                self.embedder: Any = AutoModel.from_pretrained(transformer_map[model])
            else:
                self.embedder = embedder
            # if opt.pretrained_path:
            #     self.embedder.load_state_dict(torch.load(opt.pretrained_path))

            if model == "distilbert":
                nfrozen = 6 - unfrozen_layers
            else:
                nfrozen = 12 - unfrozen_layers

            for name, param in self.embedder.named_parameters():
                if "layer" in name:
                    s = name.split("layer.")[1]
                    layer_num = s.split(".")[0]
                    if int(layer_num) < nfrozen:
                        param.requires_grad = False
                    elif opt.gpu == 0 or not dist.is_initialized():
                        print(name)
                elif "embeddings" in name:
                    if opt.nunfrozen > 5:
                        if opt.gpu == 0 or not dist.is_initialized():
                            print(name)
                    else:
                        param.requires_grad = False
            for l in self.embedder.transformer.layer:
                l.ffn.dropout = torch.nn.Dropout(opt.bert_dropout)

            if opt.gpu == 0 or not dist.is_initialized():
                print(self.embedder)

        self.pooling = opt.pooling
        self.last_two = opt.last_two and not opt.last_two_mean
        self.last_two_mean = opt.last_two_mean

        text_dim = 768
        if self.last_two:
            text_dim *= 2

        if self.pooling == "mean_cls":
            text_dim *= 2
        elif self.pooling == "lstm":
            self.pooler = LSTM(opt, embedding_dim=text_dim)
            text_dim = self.pooler.num_directions * self.pooler.hidden_dim
        elif self.pooling == "local_cluster":
            self.pooler = PointwiseFFN(opt, text_dim)

            text_dim *= opt.nclusters
        elif self.pooling == "concat":
            text_dim = 175 * text_dim
        # elif self.pooling == "global_cluster":
        #     self.pooler = GloballyCluster(opt, text_dim)
        #     text_dim *= opt.nclusters

        if pooler is not None:
            self.pooler = pooler
        if opt.tfidf:
            if opt.no_bert:
                text_dim = opt.txt_dim
            else:
                text_dim += opt.txt_dim

       
        self.attention = Attention(opt=opt, text_dim=text_dim)

        self.txt_dropout = (
            nn.Dropout(opt.txt_dropout) if opt.txt_dropout else nn.Identity()
        )
        self.tf_dropout = (
            nn.Dropout(opt.tfidf_dropout) if opt.tfidf_dropout else nn.Identity()
        )

    def forward(self, img_ft, txt_ft, tf=None):
        if txt_ft is not None:
            if hasattr(self, "embedder"):
                txt_ft = self.embedder(txt_ft, output_hidden_states=True)
                if self.last_two:
                    txt_ft = torch.cat(
                        [txt_ft.hidden_states[-1], txt_ft.hidden_states[-2]], dim=-1
                    )
                elif self.last_two_mean:
                    txt_ft = (
                        0.5 * txt_ft.hidden_states[-1] + 0.5 * txt_ft.hidden_states[-2]
                    )
                else:
                    txt_ft = txt_ft.last_hidden_state

            if self.pooling == "mean":
                txt_ft = txt_ft.mean(dim=1)
            elif self.pooling == "sum":
                txt_ft = txt_ft.sum(dim=1)
            elif self.pooling == "cls":
                txt_ft = txt_ft[:, 0, :]
            elif self.pooling == "mean_cls":
                txt_ft = torch.cat([txt_ft[:, 0, :], txt_ft.mean(dim=1)], dim=-1)
            elif self.pooling == "concat":
                txt_ft = txt_ft.reshape((txt_ft.size(0), -1))
            else:
                txt_ft = self.pooler(txt_ft)

            txt_ft = self.txt_dropout(txt_ft)
            if tf is not None:
                tf = self.tf_dropout(tf)
                txt_ft = torch.cat([txt_ft, tf], dim=-1)
        else:
            txt_ft = tf

        return self.attention(img_ft, txt_ft)


# def weights_init(m):
#     classname = m.__class__.__name__
#     if "Linear" in classname:
#         init.xavier_normal_(m.weight.data)


# class PointwiseFFN(nn.Module):
#     def __init__(self, in_dim=768, out_dim=1):
#         super(PointwiseFFN, self).__init__()
#         self.ffn = nn.Linear(in_dim, out_dim)
#         self.soft_max = nn.Softmax(dim=-1)

#     def forward(self, x):
#         classes, sentences, embed_dim = x.size()
#         x = x.reshape((classes * sentences, embed_dim))
#         weights = self.ffn(x)
#         weights = weights.reshape((classes, sentences))
#         weights = self.soft_max(weights).unsqueeze(1)
#         x = x.reshape((classes, sentences, embed_dim))
#         return torch.bmm(weights, x).squeeze()


# class MeanPool(nn.Module):
#     def __init__(self, proto=False):
#         super(MeanPool, self).__init__()
#         self.proto = proto

#     def forward(self, xin):
#         if self.proto:
#             x = xin
#             return torch.sum(x, dim=1).squeeze()
#         else:
#             x, x_lengths = xin
#             classes, _, embed_dim = x.size()
#             x_lengths = x_lengths.unsqueeze(1)
#             x = torch.sum(x, dim=1).squeeze(dim=1) / x_lengths.expand(
#                 (classes, embed_dim)
#             )
#             return x


# class IdentityEncoder(nn.Module):
#     def forward(self, x, *args, **kwargs):
#         return x


# class TransformerDecoderLayer(nn.Module):
#     def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
#         super().__init__()
#         self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
#         self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
#         # Implementation of Feedforward model
#         self.linear1 = nn.Linear(d_model, dim_feedforward)
#         self.dropout = nn.Dropout(dropout)
#         self.linear2 = nn.Linear(dim_feedforward, d_model)

#         self.norm1 = nn.LayerNorm(d_model)
#         self.norm2 = nn.LayerNorm(d_model)
#         self.norm3 = nn.LayerNorm(d_model)
#         self.dropout1 = nn.Dropout(dropout)
#         self.dropout2 = nn.Dropout(dropout)
#         self.dropout3 = nn.Dropout(dropout)

#         self.activation = nn.ReLU()

#     def forward(self, tgt, memory):
#         q = k = tgt
#         tgt2 = self.self_attn(q, k, value=tgt)[0]
#         tgt = tgt + self.dropout1(tgt2)
#         tgt = self.norm1(tgt)
#         tgt2 = self.multihead_attn(query=tgt, key=memory, value=memory)[0]
#         tgt = tgt + self.dropout2(tgt2)
#         tgt = self.norm2(tgt)
#         tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
#         tgt = tgt + self.dropout3(tgt2)
#         tgt = self.norm3(tgt)
#         return tgt


# class Proto(nn.Module):
#     def __init__(self, opt):
#         super(Proto, self).__init__()

#         self.proto_count = opt.num_proto
#         self.emb = nn.Embedding(self.proto_count, opt.text_dim)
#         if opt.encode:
#             self.transformer = nn.Transformer(
#                 opt.text_dim,
#                 opt.nhead,
#                 num_decoder_layers=opt.num_decoder_layers,
#                 dim_feedforward=opt.hidden_dim,
#             )
#         elif opt.custom_decoder:
#             self.transformer = TransformerDecoderLayer(
#                 opt.text_dim, opt.nhead, opt.hidden_dim
#             )
#         elif opt.complete_transformer:
#             self.transformer = nn.Transformer(
#                 opt.text_dim,
#                 opt.nhead,
#                 num_decoder_layers=opt.num_decoder_layers,
#                 num_encoder_layers=0,
#                 dim_feedforward=opt.hidden_dim,
#                 custom_encoder=IdentityEncoder(),
#             )
#         else:
#             dl = nn.TransformerDecoderLayer(opt.text_dim, opt.nhead, opt.hidden_dim)
#             self.transformer = nn.TransformerDecoder(
#                 dl, num_layers=opt.num_decoder_layers
#             )

#         self.complete_transformer = opt.encode or opt.complete_transformer
#         self.text_dim = opt.text_dim

#     def forward(self, x):
#         if len(x.size()) == 2:
#             if self.text_dim == 1:
#                 x = x.unsqueeze(2)
#             else:
#                 x = x.unsqueeze(1)
#             # x = x.unsqueeze(2)

#         batch, _, embed_dim = x.size()
#         protos = self.emb.weight.unsqueeze(1).expand(self.proto_count, batch, embed_dim)

#         if self.complete_transformer:
#             out = self.transformer(x.transpose(0, 1), protos)
#         else:
#             out = self.transformer(protos, x.transpose(0, 1))
#         return out.transpose(0, 1).squeeze(-1)
#         # return torch.sum(out.transpose(0, 1).squeeze(-1), dim=1).squeeze()


# class ZSL(nn.Module):
#     def __init__(self, opt):
#         super(ZSL, self).__init__()
#         unfrozen_layers = opt.unfrozen_layers
#         model = opt.txt_feat
#         text_dim = opt.text_dim
#         pooling = opt.pooling
#         embed_pooling = opt.token_pooling
#         lstm_hidden_dim = opt.lstm_hidden_dim
#         linear = opt.linear
#         text_output_dim = opt.text_output_dim
#         batch_norm = opt.batch_norm
#         dropout = opt.dropout
#         # print(model, embed_pooling, text_output_dim)
#         self.embed = transformer_map.get(model) is not None
#         self.embed_pooling = embed_pooling
#         self.unfrozen_layers = unfrozen_layers
#         self.proto = opt.proto

#         if self.embed and unfrozen_layers:
#             self.embedder: Any = AutoModel.from_pretrained(transformer_map[model])

#             if model == "distilbert":
#                 nfrozen = 6 - unfrozen_layers
#             else:

#                 nfrozen = 12 - unfrozen_layers

#             for name, param in self.embedder.named_parameters():
#                 if "layer" in name:
#                     s = name.split("layer.")[1]
#                     layer_num = s.split(".")[0]
#                     if int(layer_num) < nfrozen:
#                         param.requires_grad = False
#                     else:
#                         print(name)

#             self.embedder = nn.DataParallel(self.embedder.to(opt.device), opt.devices)

#         if self.proto:
#             self.proto = Proto(opt)

#         # Pooling
#         pooling = []
#         if sentence_pooling == "lstm" and self.embed:
#             lstm = LSTM(
#                 embedding_dim=text_dim, hidden_dim=lstm_hidden_dim, proto=self.proto
#             )
#             pooling.append(lstm)
#             text_dim = lstm.num_directions * lstm_hidden_dim

#             if batch_norm:
#                 bn1 = nn.BatchNorm1d(text_dim)
#                 pooling.append(bn1)

#             relu = nn.ReLU()
#             pooling.append(relu)

#         elif sentence_pooling == "pointwise_ffn" and self.embed:
#             ffn = PointwiseFFN(text_dim, 1)
#             pooling.append(ffn)

#         elif sentence_pooling == "mean":
#             # Will perform Sum if Proto
#             mean_pool = MeanPool(proto=self.proto)
#             pooling.append(mean_pool)

#         if dropout and self.embed:
#             d1 = nn.Dropout(dropout)
#             pooling.append(d1)

#         if linear > 0:
#             for _ in range(linear):
#                 # print('text dim and output: ', text_dim, text_output_dim)
#                 linear = nn.Linear(text_dim, text_output_dim)
#                 pooling.append(linear)

#                 if batch_norm:
#                     bn2 = nn.BatchNorm1d(text_output_dim)
#                     pooling.append(bn2)
#                 relu = nn.ReLU()
#                 pooling.append(relu)

#                 if dropout:
#                     d2 = nn.Dropout(dropout)
#                     pooling.append(d2)
#                 text_dim = text_output_dim

#         self.sentence_pooling = nn.Sequential(*pooling)

#     def forward(self, text_feat, text_len=None):
#         if self.embed and self.unfrozen_layers:
#             classes, sentences, tokens = text_feat.size()
#             out = self.embedder(text_feat.view(classes * sentences, tokens))

#             embedded = out.last_hidden_state.view((classes, sentences, tokens, 768))

#             # Embed Pooling
#             if self.embed_pooling == "mean":
#                 # Mean Pooling
#                 text_feat = torch.sum(embedded, dim=2) / torch.sum(
#                     text_feat > 0, dim=2
#                 ).unsqueeze(-1).expand((classes, sentences, 768))
#             else:
#                 # CLS Pooling
#                 text_feat = embedded[:, :, 0, :]
#             text_feat = text_feat.squeeze()

#         # Proto
#         if self.proto:
#             text_feat = self.proto(text_feat)
#             text_feat = self.sentence_pooling(text_feat)
#         elif self.embed and text_len is not None:
#             text_feat = self.sentence_pooling((text_feat, text_len))

#         return text_feat
