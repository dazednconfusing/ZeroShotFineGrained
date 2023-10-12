from typing import Any

import torch
import torch.nn.init as init
from torch import distributed as dist
from torch import nn
from transformers import AutoModel

from models.lstm import LSTM

transformer_map = {
    "bert": "bert-base-uncased",
    "sbert_cls": "sentence-transformers/bert-base-nli-cls-token",
    "sbert_mean": "sentence-transformers/bert-base-nli-mean-tokens",
    "xlnet": "xlnet-base-cased",
    "distilbert": "distilbert-base-uncased",
    "roberta": "roberta-base",
}


def weights_init(m):
    classname = m.__class__.__name__
    if "Linear" in classname:
        init.xavier_normal_(m.weight.data)


class FFEncoder(nn.Module):
    def __init__(self, hidden=512, bn=False, do=0):
        super(FFEncoder, self).__init__()
        self.linear1 = nn.Linear(512, hidden)
        self.linear2 = nn.Linear(hidden, 768)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(do) if do else nn.Identity()
        self.bn1 = nn.BatchNorm1d(hidden, affine=False) if bn else nn.Identity()
        self.bn2 = nn.BatchNorm1d(768, affine=False) if bn else nn.Identity()

    def forward(self, x):
        x = x.reshape(-1, 512)
        x = self.relu(self.bn1(self.dropout(self.linear1(x))))
        x = self.relu(self.bn2(self.dropout(self.linear2(x))))
        return x.reshape((-1, 7, 768))


class FFPooler(nn.Module):
    def __init__(self, hidden=256, bn=False, do=0):
        super(FFPooler, self).__init__()
        self.linear1 = nn.Linear(768, hidden)
        self.linear2 = nn.Linear(hidden, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(do) if do else nn.Identity()
        self.bn = nn.BatchNorm1d(hidden, affine=False) if bn else nn.Identity()

    def forward(self, x):
        x = x.reshape(-1, 512)
        x = self.relu(self.bn(self.dropout(self.linear1(x))))
        x = self.linear2(x)
        return x.reshape((-1, 7, 768))


class VZSL(nn.Module):
    def __init__(self, opt):
        super(VZSL, self).__init__()
        model = opt.txt_feat
        self.embedder_model = model
        unfrozen_layers = opt.nunfrozen

        self.embedder: Any = AutoModel.from_pretrained(transformer_map[model])

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

        self.embedder = self.embedder.to(opt.device)
        # self.img2bert = nn.Linear(512, 768)
        self.embed_text = self.embedder.get_input_embeddings()

        # self.linear2 = nn.Linear(1024, 1)
        # self.relu = nn.ReLU()
        # self.sigmoid = nn.Sigmoid()
        # self.bn = nn.BatchNorm1d()

        if opt.img_encoding == "transformer":
            self.img_encoder = nn.TransformerEncoderLayer(768, 2, 1024)
        elif opt.img_encoding == "ff":
            self.img_encoder = FFEncoder(bn=opt.batch_norm, do=opt.dropout / 3)
        else:
            self.img_encoder = nn.Identity()

        self.img_encoding = opt.img_encoding

        self.dropout = nn.Dropout(opt.dropout) if opt.dropout else nn.Identity()
        self.bn = nn.BatchNorm1d(768, affine=False) if opt.batch_norm else nn.Identity()

        if opt.pooling == "ff":
            self.pooler = FFPooler(bn=opt.bn, do=opt.dropout / 3)
        else:
            self.pooler = lambda x: torch.sum(x, dim=-1)

        # self.bn2 = nn.BatchNorm1d(1024, affine=False)

        # # Pooling
        # pooling = []
        # if sentence_pooling == "lstm" and self.embed:
        #     lstm = LSTM(
        #         embedding_dim=text_dim, hidden_dim=lstm_hidden_dim, proto=self.proto
        #     )
        #     pooling.append(lstm)
        #     text_dim = lstm.num_directions * lstm_hidden_dim

        #     if batch_norm:
        #         bn1 = nn.BatchNorm1d(text_dim)
        #         pooling.append(bn1)

        #     relu = nn.ReLU()
        #     pooling.append(relu)

        # elif sentence_pooling == "pointwise_ffn" and self.embed:
        #     ffn = PointwiseFFN(text_dim, 1)
        #     pooling.append(ffn)

        # elif sentence_pooling == "mean":
        #     # Will perform Sum if Proto
        #     mean_pool = MeanPool(proto=self.proto)
        #     pooling.append(mean_pool)

        # if dropout and self.embed:
        #     d1 = nn.Dropout(dropout)
        #     pooling.append(d1)

        # if linear > 0:
        #     for _ in range(linear):
        #         # print('text dim and output: ', text_dim, text_output_dim)
        #         linear = nn.Linear(text_dim, text_output_dim)
        #         pooling.append(linear)

        #         if batch_norm:
        #             bn2 = nn.BatchNorm1d(text_output_dim)
        #             pooling.append(bn2)
        #         relu = nn.ReLU()
        #         pooling.append(relu)

        #         if dropout:
        #             d2 = nn.Dropout(dropout)
        #             pooling.append(d2)
        #         text_dim = text_output_dim

        # self.sentence_pooling = nn.Sequential(*pooling)

    def forward(self, img_ft, txt_ft, text_len=None):
        if self.img_encoding == "transformer":
            img_ft = img_ft.transpose(0, 1)

        embed_img = self.img_encoder(self.dropout(img_ft))

        if self.img_encoding == "transformer":
            embed_img = img_ft.transpose(0, 1)
        embed_txt = self.embed_text(txt_ft)
        embeds = torch.cat([embed_txt, embed_img], dim=1)

        if self.embedder_model != "distilbert":
            token_type_ids = torch.cat(
                [torch.zeros_like(embed_txt), torch.ones_like(embed_img)], dim=1
            ).long()
            out = self.embedder(inputs_embeds=embeds, token_type_ids=token_type_ids)
        else:
            out = self.embedder(inputs_embeds=embeds)

        embedded = out.last_hidden_state[:, 0, :]

        alignment = self.pooler(embedded)

        return alignment.squeeze()
