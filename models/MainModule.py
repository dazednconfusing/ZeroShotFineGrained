
import torch
from torch import nn
from models.TextToImageSpace import TextToImage
import torch.nn.functional as F
from proto_module import ProtoModule, ApplyProto

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import scipy.io as sio
import matplotlib.pyplot as plt



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


from models.zest import Attention as ZestNet


from datasets.feature_dataset import FeatureDataset
from dataloaders.CUBLoader import CUBFeatDataSet


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
    if name  == "relu":
        return F.relu
    if name == "sigmoid":
        return F.sigmoid
    if name == "tanh":
        return F.tanh


class Options():
    def __init__(self, img_dim=None, txt_dim=None, proto_count=None, txt_to_img_hidden1=None,
                 txt_to_img_norm=None, apply_proto_dim_feed_forward=None, apply_proto_combine = None):
        self.img_dim = img_dim
        self.txt_dim = txt_dim
        self.proto_count = proto_count
        self.txt_to_img_hidden1 =  txt_to_img_hidden1
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
        self.opt =  opt
        self.proto_module = get_proto_module(opt)
        self.ap_txt = get_apply_proto_for_text(opt)
        self.ap_img = get_apply_proto_for_img(opt)
        self.txt_to_img = get_text_to_image(opt)


    # txt feat can be image specific or the entire description like in Zest.
    # txt feat should be (batch, sequence length, embedd dim)
    def forward(self, img_feat, txt_feat):

        bs, seq_length, em_dim = txt_feat.shape
        txt_feat_flatten = txt_feat#.reshape(bs*seq_length, em_dim)

        txt_proto_vecs = self.proto_module.get_protos(txt_feat_flatten.shape[0])
        txt_after_proto = self.ap_txt(txt_proto_vecs, txt_feat_flatten)
        txt_after_proto = txt_after_proto.reshape(1, bs, -1)
        print("txt_after_proto", txt_after_proto.shape)

        # ibs = img_feat.shape[0]
        # img_proto_vecs = self.txt_to_img(self.proto_module.get_protos(None)).unsqueeze(0).repeat(ibs, 1, 1)
        # print(img_proto_vecs.shape)
        # img_after_proto = self.ap_img(img_proto_vecs, img_feat)
        # print(img_after_proto.shape)
        return img_feat, txt_after_proto


def train_zest_style():
    opt = Options(img_dim=3584, txt_dim=7551, proto_count=5, txt_to_img_hidden1=409,
                 txt_to_img_norm="relu", apply_proto_dim_feed_forward=204, apply_proto_combine = "sum")

    tf = TransformFeatures(opt)
    train_dataset = CUBFeatDataSet(folder_path='/Users/nikhilt/Desktop/Sp21/proj/CIS620/data/CUB2011', split='easy',
                                   model='vanilla', train=True)
    tain_txt_feat_all = train_dataset.train_text_feature
    tain_txt_feat_all = torch.from_numpy(tain_txt_feat_all).float().unsqueeze(1)
    train_loader = DataLoader(train_dataset, batch_size=2)

    test_dataset = CUBFeatDataSet(folder_path='/Users/nikhilt/Desktop/Sp21/proj/CIS620/data/CUB2011', split='easy',
                                  model='vanilla', train=False)
    I_all, label_all, orig_label_all, txt_feat_all = test_dataset.get_all_data()
    text_inp = torch.from_numpy(txt_feat_all).float().unsqueeze(1)
    test_labels = test_dataset.test_cid
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    tain_txt_feat_all = tain_txt_feat_all.to(device)
    text_inp = text_inp.to(device)

    net = ZestNet(dimensions=3584, text_dim=7551)
    # net.apply(init_weights_xavier)

    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-2)

    epochs = 60
    zsl_acc = []
    criterion = torch.nn.CrossEntropyLoss()
    for epoch in range(epochs):
        net.train()
        with tqdm(train_loader, unit='batch') as tepoch:
            for img_feat, orig_lab, _, txt_feat in tepoch:
                print(tain_txt_feat_all.shape)
                img_feat, temp_feat_all = tf(img_feat, tain_txt_feat_all)
                attention_weights, attention_scores = net(img_feat, temp_feat_all)
                optimizer.zero_grad()
                loss = criterion(attention_weights.squeeze(), orig_lab.long())
                loss.backward()
                optimizer.step()
                tepoch.set_postfix(loss=loss.item())

        # net.eval()
        # correct = 0
        # with tqdm(test_loader, unit='batch') as tepoch:
        #     for img_feat, orig_lab, _, txt_feat in tepoch:
        #         attention_weights, attention_scores = net(img_feat, text_inp)
        #         topv, topi = attention_weights.squeeze().data.topk(1)
        #         correct += torch.sum(topi.squeeze() == orig_lab).cpu().tolist()
        #
        # acc = correct * 1.0 / len(test_loader.dataset)
        # print('Accuracy: {}\n'.format(acc))
        # zsl_acc.append(acc)

def main():
    train_zest_style()

if __name__ == "__main__":
    main()




