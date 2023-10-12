
from typing import List, Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn

# This model is from Facebook Detr
# Ref: https://github.com/facebookresearch/detr/


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)


class ApplyProto(nn.Module):

    def __init__(self, feat_dim, proto_count, nhead=1, dim_feedforward=100, dropout=0.1, activation="relu",
                 normalize_before=False, combine="sum"):
        super().__init__()
        self.decoder_layer = TransformerDecoderLayer(
            feat_dim, nhead, dim_feedforward, dropout, activation, normalize_before)
        self.conversion = None
        self.combine = combine
        if self.combine == "linear":
            self.conversion = nn.Linear(proto_count*feat_dim, feat_dim)

    # feat shape: bsxdim, proto_vecs: bs x pcount x pdim

    def forward(self, proto_vecs, feat):
        p = proto_vecs.permute(1, 0, 2)
        f = feat
        if len(feat.shape) == 2:
            f = feat.unsqueeze(0)
        else:
            f = feat.permute(1, 0, 2)
        out = self.decoder_layer.forward(p, f)
        out = out.permute(1, 0, 2)
        bs = out.shape[0]
        if self.combine == "sum":
            return torch.sum(out, dim=1).reshape(bs, -1)
        elif self.combine == "linear":
            return self.conversion.forward(out.reshape(bs, -1))
        else:
            return out


class ProtoModule(nn.Module):
    def __init__(self, proto_count, proto_feat):
        super().__init__()
        self.proto_count = proto_count
        self.proto_feat = proto_feat
        self.proto = nn.Embedding(proto_count, proto_feat)

    # return bs x n x dim
    def get_protos(self, bs):
        if bs is not None:
            return self.proto.weight.unsqueeze(0).repeat(bs, 1, 1)
        else:
            return self.proto.weight


# Code to test Detr decoder for proto vectors.
def main():

    pm = ProtoModule(30, 1024)
    ap = ApplyProto(1024, 30, combine="sum")

    img_feat = torch.randn((10, 1024))
    query_feat = pm.get_protos(10)

    print(query_feat.shape)

    out = ap.forward(query_feat, img_feat)

    print(out.shape)

    pass


if __name__ == "__main__":
    main()
