import sys
# Change the path to folder
import os
sys.path.append(os.getcwd())

import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import utils.knn as knn
from datasets.resnet_dataset import ResnetDataset
import itertools

from Script_Helper import *
from models.options import Options
from models.baseline import BaselineNet as MSEBaseLine

def train_zest_style():
    opt = Options(img_dim=2048, txt_dim=7551, proto_count=30, txt_to_img_hidden1=2048,
                  txt_to_img_norm="relu", apply_proto_dim_feed_forward=1024, apply_proto_combine="sum")

    tf = TransformFeatures(opt).to(device)
    #     train_dataset = CUBFeatDataSet(folder_path='/cbica/home/thodupv/zero/CIS620/data/CUB2011', split='easy',
    #                                    model='vanilla', train=True, size_limit=30)

    train_dataset = ResnetDataset(split='hard', train=True)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, pin_memory=True)

    # tain_txt_feat_all = train_dataset.bert_train_feat.float()
    I_all_train, label_all_train, orig_label_all_train, txt_feat_all_train = train_dataset.get_all_data()
    print(I_all_train.shape)
    text_inp_train = torch.from_numpy(txt_feat_all_train).float().unsqueeze(1)
    train_labels = train_dataset.train_cid

    train_dataset_size = len(train_dataset)

    #     test_dataset = CUBFeatDataSet(folder_path='data/CUB2011_2', split='hard', model='vanilla', train=False)
    test_dataset = ResnetDataset(split='hard', train=False)
    test_dataset_size = len(test_dataset)
    I_all, label_all, orig_label_all, txt_feat_all = test_dataset.get_all_data()
    text_inp = torch.from_numpy(txt_feat_all).float().unsqueeze(1)
    test_labels = test_dataset.test_cid

    tain_txt_feat_all = text_inp_train.to(device)
    text_inp = text_inp.to(device)

    train_dataset_size = len(train_dataset)
    test_dataset_size = len(test_dataset)

    # net = Attention(dimensions=3584, text_dim=7551).to(device)
    net = MSEBaseLine(7551, 4096, 2048).to(device)
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
                #                 print(ot.shape, img_feat.shape)
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
        if True:
            _, text_temp_inp = tf(None, text_inp)
            text_pred = net(text_temp_inp.squeeze(0))
            outpred = [0] * test_dataset_size
            for i in range(test_dataset_size):
                outputLabel = knn.kNNClassify(I_all[i, :], text_pred.cpu().data.numpy(), test_labels, 1)
                outpred[i] = outputLabel
            outpred = np.array(outpred)
            acc = np.equal(outpred, orig_label_all).mean()
            print('Test Accuracy: {}\n'.format(acc))
            zsl_acc.append(acc)


#         net.eval()
#         tf.eval()
#         correct = 0
#         with tqdm(test_loader, unit='batch') as tepoch:
#             _, text_temp_inp = tf(None, text_inp)
# #           text_temp_inp = text_inp.to(device)
#             text_pred = net(text_temp_inp.squeeze(0))
#             outpred = [0] * test_dataset_size
#             for i in range(test_dataset_size):
#                 outputLabel = knn.kNNClassify(I_all[i, :], text_pred.cpu().data.numpy(), test_labels, 1)
#                 outpred[i] = outputLabel
#             outpred = np.array(outpred)
#             acc = np.equal(outpred, orig_label_all).mean()
#             print('Test Accuracy: {}\n'.format(acc))
#             zsl_acc.append(acc)


## Train data size
#         net.eval()
#         tf.eval()
#         correct = 0
#         ck = nn.MSELoss()
#         with tqdm(test_loader, unit='batch') as tepoch:
#             _, text_temp_inp = tf(None, tain_txt_feat_all)
# #           text_temp_inp = text_inp.to(device)
#             text_pred = net(text_temp_inp.squeeze(0))
# #             print(text_pred.shape)
#             outpred = [0] * train_dataset_size
#             for i in range(train_dataset_size):
#                 outputLabel = knn.kNNClassify(I_all_train[i, :].data.numpy(), text_pred.cpu().data.numpy(), train_labels, 1)
#                 outpred[i] = outputLabel
#                 vec = I_all_train[i:i+1, :]
#             #print(ck(text_pred[0, :],  I_all_train[0, :].to(device)))
#             #print(outpred)
#             outpred = np.array(outpred)
#             acc = np.equal(outpred, orig_label_train_all[:1]).mean()
#             print('Train Accuracy: {}\n'.format(acc))
#             #zsl_acc.append(acc)

def main():
    train_zest_style()
    pass


if __name__ == "__main__":
    main()

