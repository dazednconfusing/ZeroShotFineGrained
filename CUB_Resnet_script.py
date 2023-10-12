import sys

sys.path.append("/cbica/home/thodupv/zero/CIS620")

import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import torch
from torch import nn
from models.TextToImageSpace import TextToImage
import torch.nn.functional as F
from models.proto_module import ProtoModule, ApplyProto
import itertools

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import scipy.io as sio
import matplotlib.pyplot as plt

from models.zest import Attention as ZestNet

from datasets.feature_dataset import FeatureDataset
from datasets.resnet_dataset import ResnetDataset
from dataloaders.CUBLoader import CUBFeatDataSet, NABFeatDataSet


class Attention(nn.Module):
    def __init__(self, dimensions=3584, text_dim=7621):
        super(Attention, self).__init__()

        self.linear_out = nn.Linear(dimensions * 2, dimensions, bias=False)
        self.softmax = nn.Softmax(dim=-1)
        self.tanh = nn.Tanh()
        self.main = nn.Linear(text_dim, dimensions, bias=False)

    def forward(self, query, text_feat):
        context = self.main(text_feat)
        context = context.expand(query.shape[0], context.shape[1],
                                 context.shape[2])
        query = query.unsqueeze(1)
        batch_size, output_len, dimensions = query.size()
        query_len = context.size(1)

        attention_scores = torch.bmm(
            query, context.transpose(1, 2).contiguous())
        attention_scores = attention_scores.view(
            batch_size * output_len, query_len)
        attention_weights = self.softmax(attention_scores)
        attention_weights = attention_weights.view(
            batch_size, output_len, query_len)

        return attention_weights, attention_scores


def get_norm_function(name):
    if name == "relu":
        return F.relu
    if name == "sigmoid":
        return F.sigmoid
    if name == "tanh":
        return F.tanh


class Options():
    def __init__(self, img_dim=None, txt_dim=None, proto_count=None, txt_to_img_hidden1=None,
                 txt_to_img_norm=None, apply_proto_dim_feed_forward=None, apply_proto_combine=None):
        self.img_dim = img_dim
        self.txt_dim = txt_dim
        self.proto_count = proto_count
        self.txt_to_img_hidden1 = txt_to_img_hidden1
        self.txt_to_img_norm = txt_to_img_norm
        self.apply_proto_dim_feed_forward = apply_proto_dim_feed_forward
        self.apply_proto_combine = apply_proto_combine


def get_text_to_image(opt):
    return TextToImage(opt.img_dim, opt.txt_dim, opt.txt_to_img_hidden1, get_norm_function(opt.txt_to_img_norm))


def get_proto_module(opt):
    return ProtoModule(opt.proto_count, opt.txt_dim)


def get_apply_proto_for_text(opt):
    return ApplyProto(opt.txt_dim, opt.proto_count, combine=opt.apply_proto_combine,
                      dim_feedforward=opt.apply_proto_dim_feed_forward)


def get_apply_proto_for_img(opt):
    return ApplyProto(opt.img_dim, opt.proto_count, combine=opt.apply_proto_combine,
                      dim_feedforward=opt.apply_proto_dim_feed_forward)


class TransformFeaturesSimple(nn.Module):

    def __init__(self, opt):
        super().__init__()
        self.proto_module = get_proto_module(opt)

    def forward(self, img_feat, txt_feat):
        bs, seq_length, em_dim = txt_feat.shape
        #         txt_feat_flatten = txt_feat.reshape(bs*seq_length, em_dim)

        #         txt_proto_vecs = self.proto_module.get_protos(1).squeeze(0)

        #         output = torch.zeros((bs*seq_length, em_dim)).float().to(device)

        #         for i in range(txt_feat_flatten.shape[0]):
        #             semi_out = torch.zeros((1, em_dim)).float().to(device)
        #             for j in range(txt_proto_vecs.shape[0]):
        #                 semi_out = semi_out + torch.dot(txt_feat_flatten[i, :], txt_proto_vecs[j, :]) * txt_proto_vecs[j:j+1, :]
        #             output[i, :] = semi_out[0, :]

        #         output = output.reshape(1, bs, -1)
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

        #         ibs = img_feat.shape[0]
        #         img_proto_vecs = self.txt_to_img(self.proto_module.get_protos(None)).unsqueeze(0).repeat(ibs, 1, 1)
        #         img_after_proto = self.ap_img(img_proto_vecs, img_feat)
        return img_feat, txt_after_proto


def train_zest_style():
    opt = Options(img_dim=2048, txt_dim=7551, proto_count=30, txt_to_img_hidden1=4096,
                  txt_to_img_norm="relu", apply_proto_dim_feed_forward=2048, apply_proto_combine="sum")

    tf = TransformFeaturesSimple(opt).to(device)
#     train_dataset = NABFeatDataSet(folder_path='/cbica/home/thodupv/zero/CIS620/data/NABird', split='easy',
#                                    model='vanilla', train=True)
#     train_dataset = CUBFeatDataSet(folder_path='data/CUB2011', split='easy', model='vanilla', train=True)
    
    train_dataset = ResnetDataset(split='easy', normalize=True, train=True)

    tain_txt_feat_all = train_dataset.train_text_feature
    tain_txt_feat_all = torch.from_numpy(tain_txt_feat_all).float().unsqueeze(1)
    train_loader = DataLoader(train_dataset, batch_size=50, shuffle=True)

#     test_dataset = NABFeatDataSet(folder_path='/cbica/home/thodupv/zero/CIS620/data/NABird', split='easy',
#                                   model='vanilla', train=False)
    test_dataset = ResnetDataset(split='easy',  normalize=True, train=False)
#     test_dataset = CUBFeatDataSet(folder_path='data/CUB2011', split='easy', model='vanilla', train=False)

    txt_feat_all = test_dataset.test_text_feature
    text_inp = torch.from_numpy(txt_feat_all).float().unsqueeze(1)
    test_labels = test_dataset.test_cid
    test_loader = DataLoader(test_dataset, batch_size=50, shuffle=False)

    tain_txt_feat_all = tain_txt_feat_all.to(device)
    text_inp = text_inp.to(device)

    net = Attention(dimensions=2048, text_dim=7551).to(device)
    # net.apply(init_weights_xavier)

#     optimizer = torch.optim.Adam(net.parameters(), lr=0.0005, betas=(0.5, 0.9), weight_decay=0.003)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.0005, betas=(0.5, 0.9), weight_decay=0.)
#     optimizer = torch.optim.Adam(itertools.chain(net.parameters(), tf.parameters()), lr=0.0005, betas=(0.5, 0.9), weight_decay=0.0001)

    epochs = 600
    zsl_acc = []
    criterion = torch.nn.CrossEntropyLoss()
    for epoch in range(epochs):
        net.train()
        tf.train()
        correct = 0
        with tqdm(train_loader, unit='batch') as tepoch:
            for img_feat, orig_lab, _, _ in tepoch:
#                 print(tain_txt_feat_all.shape, img_feat.shape)
                img_feat = img_feat.to(device)
                orig_lab = orig_lab.to(device)
                img_feat, temp_feat_all = tf(img_feat, tain_txt_feat_all)
                attention_weights, attention_scores = net(img_feat, temp_feat_all)
                topv, topi = attention_weights.squeeze().data.topk(1)
                correct += torch.sum(topi.squeeze() == orig_lab).cpu().tolist()
                optimizer.zero_grad()
                loss = criterion(attention_weights.squeeze(), orig_lab.long())
                loss.backward()
                optimizer.step()
                tepoch.set_description('EPOCH: %i' % epoch)
                tepoch.set_postfix(loss=loss.item())
                # print(90)

        acc = correct * 1.0 / len(train_loader.dataset)
        print('Train Accuracy: {}\n'.format(acc))

        net.eval()
        tf.eval()
        correct = 0
        with tqdm(test_loader, unit='batch') as tepoch:
            for img_feat, orig_lab, _, _ in tepoch:
                img_feat = img_feat.to(device)
                orig_lab = orig_lab.to(device)
                img_feat, temp_feat_all = tf(img_feat, text_inp)
                attention_weights, attention_scores = net(img_feat, temp_feat_all)
                topv, topi = attention_weights.squeeze().data.topk(1)
                correct += torch.sum(topi.squeeze() == orig_lab).cpu().tolist()

        acc = correct * 1.0 / len(test_loader.dataset)
        print('Test Accuracy: {}\n'.format(acc))
        zsl_acc.append(acc)


def main():
    train_zest_style()


if __name__ == "__main__":
    main()
