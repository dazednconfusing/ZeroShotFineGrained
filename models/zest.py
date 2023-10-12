import torch
import torch.nn as nn


class Attention(nn.Module):
    def __init__(
        self,
        opt,
        text_dim=7621,
        img_dim=3584,
    ):
        super(Attention, self).__init__()

        # self.linear_out = nn.Linear(dimensions * 2, dimensions, bias=False)
        # self.softmax = nn.Softmax(dim=-1)
        # self.tanh = nn.Tanh()
        self.img2txt = opt.img2txt
        if opt.img2txt:
            self.project = nn.Linear(img_dim, text_dim, bias=False)
        else:
            self.project = nn.Linear(text_dim, img_dim, bias=False)
        self.img_dropout = (
            nn.Dropout(opt.img_dropout) if opt.img_dropout else nn.Identity()
        )
        # self.txt_dropout = (
        #     nn.Dropout(opt.txt_dropout) if opt.txt_dropout else nn.Identity()
        # )
        self.img_bn = nn.BatchNorm1d(img_dim) if opt.img_bn else nn.Identity()
        self.txt_bn = nn.BatchNorm1d(text_dim) if opt.txt_bn else nn.Identity()

        # self.softmax = nn.Softmax(dim=-1)

    def forward(self, img_ft, txt_ft):
        # n_txts = txt_ft.size(0)
        # bs = img_ft.size(0)
        img_ft = self.img_dropout(img_ft)
        # txt_ft = self.txt_dropout(txt_ft)

        img_ft = self.img_bn(img_ft)
        txt_ft = self.txt_bn(txt_ft)
        # Project txt_ft to the img_ft space

        if self.img2txt:
            img_ft = self.project(img_ft)
        else:
            txt_ft = self.project(txt_ft)

        txt_ft = txt_ft.unsqueeze(0)

        # if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
        #     print("proj size: ", proj.size())

        # Copy the text features for each img_ft in the batch
        # Now proj.shape = (bs, img_ft_dim, n_txts)
        txt_ft = txt_ft.expand(img_ft.shape[0], -1, -1).transpose(1, 2)

        # Now img_ft.shape = (bs, 1, img_ft_dim)
        img_ft = img_ft.unsqueeze(1)

        # if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
        #     print("proj, img size: ", proj.size(), img_ft.size())
        attention_scores = torch.bmm(img_ft, txt_ft).squeeze()
        return attention_scores

        # return attention_weights, attention_scores
        # context = context.expand(img_ft.shape[0], -1, -1)
        # img_ft = img_ft.unsqueeze(1)
        # batch_size, output_len, _ = img_ft.size()
        # query_len = context.size(1)

        # attention_scores = torch.bmm(img_ft, context.transpose(1, 2).contiguous())
        # attention_scores = attention_scores.view(batch_size * output_len, query_len)
        # attention_weights = self.softmax(attention_scores)
        # attention_weights = attention_weights.view(batch_size, output_len, query_len)

        # return attention_weights, attention_scores
