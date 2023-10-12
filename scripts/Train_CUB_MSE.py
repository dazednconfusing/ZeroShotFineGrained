import os
import sys

sys.path.append(os.getcwd())

import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import argparse
import itertools

import numpy as np
import torch
import utils.knn as knn
from dataloaders.CUBLoader import CUBFeatDataSet
from models.baseline import BaselineNet as MSEBaseLine
from models.options import Options
from models.zest import Attention
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from Script_Helper import *


def train_zest_style(batch_size=100, gpu='0'):
    device = torch.device("cuda:"+gpu if torch.cuda.is_available() else "cpu")
    print(device)
    opt = Options(img_dim=3584, txt_dim=7551, proto_count=30, txt_to_img_hidden1=2048,
                  txt_to_img_norm="relu", apply_proto_dim_feed_forward=1024, apply_proto_combine="sum")

    tf = TransformFeatures(opt).to(device)
    train_dataset = CUBFeatDataSet(folder_path='data/CUB2011', split='easy',
                                   model='vanilla', train=True, size_limit=30)

    # tain_txt_feat_all = train_dataset.bert_train_feat.float()
    tain_txt_feat_all = train_dataset.train_text_feature
    tain_txt_feat_all = torch.from_numpy(tain_txt_feat_all).float().unsqueeze(1)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    I_all_train = torch.from_numpy(train_dataset.pfc_feat_data_train).float()
    train_labels = train_dataset.train_cid
    orig_label_train_all = train_dataset.train_cid[train_dataset.labels_train]

    test_dataset = CUBFeatDataSet(folder_path='data/CUB2011', split='easy',
                                  model='vanilla', train=False, size_limit=30)

    I_all, label_all, orig_label_all, txt_feat_all = test_dataset.get_all_data()
    # text_inp = test_dataset.bert_test_feat.float()
    text_inp = torch.from_numpy(txt_feat_all).float().unsqueeze(1)
    test_labels = test_dataset.test_cid
    test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False)

    tain_txt_feat_all = tain_txt_feat_all.to(device)
    text_inp = text_inp.to(device)

    train_dataset_size = len(train_dataset)
    test_dataset_size = len(test_dataset)

    # net = Attention(dimensions=3584, text_dim=7551).to(device)
    net = MSEBaseLine(7551, 4096, 3584).to(device)
    # net.apply(init_weights_xavier)

    optimizer = torch.optim.Adam(itertools.chain(net.parameters(), tf.parameters()), lr=0.0005, betas=(0.5, 0.9),
                                 weight_decay=0.0001)

    epochs = 600
    zsl_acc = []
    epoch_losses = []
    criterion = torch.nn.MSELoss()
    for epoch in range(epochs):
        net.train()
        tf.train()
        correct = 0
        epoch_loss = 0.0
        with tqdm(train_loader, unit='batch') as tepoch:
            for img_feat, orig_lab, _, txt_feat in tepoch:
                img_feat = img_feat.to(device)
                orig_lab = orig_lab.to(device)
                txt_feat = txt_feat.unsqueeze(1).to(device)
                img_feat, temp_feat_all = tf(img_feat, txt_feat)
                ot = net(temp_feat_all.squeeze(0))
                optimizer.zero_grad()
                loss = criterion(ot, img_feat)
                loss.backward()
                optimizer.step()
                loss_d = {'Loss': loss.item()}
                tepoch.set_description('EPOCH: %i' % epoch)
                tepoch.set_postfix(loss_d)

                epoch_loss += ot.shape[0] * loss.item()

        epoch_loss /= train_dataset_size
        epoch_losses.append(epoch_loss)

        print('Epoch Loss: {}\n'.format(epoch_loss))

        net.eval()
        tf.eval()
        correct = 0
        with tqdm(test_loader, unit='batch') as tepoch:
            _, text_temp_inp = tf(None, text_inp)
            #           text_temp_inp = text_inp.to(device)
            text_pred = net(text_temp_inp.squeeze(0))
            outpred = [0] * test_dataset_size
            for i in range(test_dataset_size):
                outputLabel = knn.kNNClassify(I_all[i, :], text_pred.cpu().data.numpy(), test_labels, 1)
                outpred[i] = outputLabel
            outpred = np.array(outpred)
            acc = np.equal(outpred, orig_label_all).mean()
            print('Test Accuracy: {}\n'.format(acc))
            zsl_acc.append(acc)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--gpu', type=str, default='0')
    parser.add_argument('-b', '--bs', type=int, default=100)
    args = parser.parse_args()
    train_zest_style(args.bs, args.gpu)


if __name__ == "__main__":
    main()

