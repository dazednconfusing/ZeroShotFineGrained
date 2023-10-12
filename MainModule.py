import os

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from dataloaders.CUBLoader import CUBFeatDataSet
from datasets.feature_dataset import FeatureDataset
from models.proto_module import ApplyProto, ProtoModule
from models.TextToImageSpace import TextToImage
from models.zest import Attention as ZestNet

# print('cwd: ', os.getcwd())

#####
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')


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
    return ApplyProto(opt.txt_dim, opt.proto_count, combine=opt.apply_proto_combine, dim_feedforward=opt.apply_proto_dim_feed_forward)


def get_apply_proto_for_img(opt):
    return ApplyProto(opt.img_dim, opt.proto_count, combine=opt.apply_proto_combine, dim_feedforward=opt.apply_proto_dim_feed_forward)


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
        txt_feat_flatten = txt_feat.reshape(bs*seq_length, em_dim)

        txt_proto_vecs = self.proto_module.get_protos(
            txt_feat_flatten.shape[0])
        txt_after_proto = self.ap_txt(txt_proto_vecs, txt_feat_flatten)
        txt_after_proto = txt_after_proto.reshape(bs, seq_length, -1)

        ibs = img_feat.shape[0]
        img_proto_vecs = self.txt_to_img(
            self.proto_module.get_protos(None)).unsqueeze(0).repeat(ibs, 1, 1)
        # print(img_proto_vecs.shape)
        img_after_proto = self.ap_img(img_proto_vecs, img_feat)
        # print(img_after_proto.shape)
        return img_after_proto, txt_after_proto


def weights_init(m):
    classname = m.__class__.__name__
    if 'Linear' in classname:
        init.xavier_normal_(m.weight.data)


def train_zest_style():
    opt = Options(img_dim=3584, txt_dim=7551, proto_count=5, txt_to_img_hidden1=409,
                  txt_to_img_norm="relu", apply_proto_dim_feed_forward=204, apply_proto_combine="sum")

    tf = TransformFeatures(opt).to(device)
    train_dataset = CUBFeatDataSet(folder_path='data/CUB2011', split='easy',
                                   model='vanilla', train=True)
    train_txt_feat_all = train_dataset.train_text_feature
    train_txt_feat_all = torch.from_numpy(
        train_txt_feat_all).float().unsqueeze(0).to(device)
    train_loader = DataLoader(train_dataset, batch_size=100)

    test_dataset = CUBFeatDataSet(folder_path='data/CUB2011', split='easy',
                                  model='vanilla', train=False)
    I_all, label_all, orig_label_all, txt_feat_all = test_dataset.get_all_data()
    text_inp = torch.from_numpy(txt_feat_all).float().unsqueeze(0).to(device)
    test_labels = test_dataset.test_cid
    test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False)

    net = ZestNet(dimensions=3584, text_dim=7551).to(device)
    net.apply(weights_init)

    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-2)

    epochs = 60
    zsl_acc = []
    criterion = torch.nn.CrossEntropyLoss()
    for epoch in range(epochs):
        net.train()
        with tqdm(train_loader, unit='batch') as tepoch:
            for img_feat, orig_lab, _, txt_feat in tepoch:
                orig_lab = Variable(orig_lab.long()).to(device)
                img_feat = Variable(img_feat).to(device)
                img_feat, train_txt_feat_all = tf(img_feat, train_txt_feat_all)
                attention_weights, attention_scores = net(
                    img_feat, train_txt_feat_all)
                loss = criterion(attention_weights.squeeze(), orig_lab)
                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()
                tepoch.set_postfix(loss=loss.item())
                # print(90)

        net.eval()
        correct = 0
        with tqdm(test_loader, unit='batch') as tepoch:
            for img_feat, orig_lab, _, txt_feat in tepoch:
                attention_weights, attention_scores = net(img_feat, text_inp)
                topv, topi = attention_weights.squeeze().data.topk(1)
                correct += torch.sum(topi.squeeze() == orig_lab).cpu().tolist()

        acc = correct * 1.0 / len(test_loader.dataset)
        print('Accuracy: {}\n'.format(acc))
        zsl_acc.append(acc)


def main():
    train_zest_style()


if __name__ == "__main__":
    main()
