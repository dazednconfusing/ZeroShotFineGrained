import torch
import torch.nn as nn
from models.proto_module import ApplyProto, ProtoModule
from models.TextToImageSpace import TextToImage
from utils.helper_functions import get_norm_function


def get_text_to_image(opt):
    return TextToImage(
        opt.img_dim,
        opt.txt_dim,
        opt.txt_to_img_hidden1,
        get_norm_function(opt.txt_to_img_norm),
    )


def get_proto_module(opt):
    return ProtoModule(opt.proto_count, opt.txt_dim)


def get_apply_proto_for_text(opt):
    return ApplyProto(
        opt.txt_dim,
        opt.proto_count,
        combine=opt.apply_proto_combine,
        dim_feedforward=opt.apply_proto_dim_feed_forward,
    )


def get_apply_proto_for_img(opt):
    return ApplyProto(
        opt.img_dim,
        opt.proto_count,
        combine=opt.apply_proto_combine,
        dim_feedforward=opt.apply_proto_dim_feed_forward,
    )


class TransformFeaturesSimple(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.proto_module = get_proto_module(opt)

    def forward(self, img_feat, txt_feat):
        bs, seq_length, em_dim = txt_feat.shape
        return img_feat, txt_feat.reshape(1, bs, -1)


class TransformFeatures(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.proto_module = get_proto_module(opt)
        self.ap_txt = get_apply_proto_for_text(opt)
        self.ap_img = get_apply_proto_for_img(opt)
        self.txt_to_img = get_text_to_image(opt)

    # txt feat can be image specific or the entire description like in Zest.
    # txt feat should be (batch, sequence length, embedd dim)
    def forward(self, img_feat, txt_feat):
        bs, seq_length, em_dim = txt_feat.shape
        txt_feat_flatten = txt_feat.reshape(bs * seq_length, em_dim)

        txt_proto_vecs = self.proto_module.get_protos(txt_feat_flatten.shape[0])
        txt_after_proto = self.ap_txt(txt_proto_vecs, txt_feat_flatten)
        txt_after_proto = txt_after_proto.reshape(1, bs, -1)

        return img_feat, txt_after_proto


class TransformFeaturesSimpleProto(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.proto_module = get_proto_module(opt)

    def forward(self, img_feat, txt_feat):
        bs, seq_length, em_dim = txt_feat.shape
        txt_feat_flatten = txt_feat.reshape(bs * seq_length, em_dim)

        txt_proto_vecs = self.proto_module.get_protos(1).squeeze(0)
        proto_count = txt_proto_vecs.shape[0]

        attention_weights = torch.matmul(
            txt_feat_flatten, torch.transpose(txt_proto_vecs, 0, 1)
        )

        attention_weights = attention_weights.reshape(-1, 1)

        proto_repeat = txt_proto_vecs.repeat(bs, 1)

        weighted = attention_weights * proto_repeat

        weighted = weighted.reshape(bs, proto_count, -1).permute(0, 2, 1)

        ot = torch.sum(weighted, dim=2)

        output = ot.reshape(1, bs, -1)
        return img_feat, output
