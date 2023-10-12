import sys
import os
sys.path.append(os.getcwd())
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import itertools
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from datasets.resnet_dataset import ResnetDataset

from Script_Helper import *
from models.options import Options
from models.zest import Attention

def train_zest_style():
    opt = Options(img_dim=2048, txt_dim=7551, proto_count=60, txt_to_img_hidden1=4096,
                  txt_to_img_norm="relu", apply_proto_dim_feed_forward=2048, apply_proto_combine="sum")

    tf = TransformFeatures(opt).to(device)
    #     train_dataset = NABFeatDataSet(folder_path='/cbica/home/thodupv/zero/CIS620/data/NABird', split='easy',
    #                                    model='vanilla', train=True)

    train_dataset = ResnetDataset(split='easy', train=True)

    tain_txt_feat_all = train_dataset.train_text_feature
    tain_txt_feat_all = torch.from_numpy(tain_txt_feat_all).float().unsqueeze(1)
    train_loader = DataLoader(train_dataset, batch_size=100, shuffle=False)

    #     test_dataset = NABFeatDataSet(folder_path='/cbica/home/thodupv/zero/CIS620/data/NABird', split='easy',
    #                                   model='vanilla', train=False)
    test_dataset = ResnetDataset(split='easy', train=False)

    txt_feat_all = test_dataset.test_text_feature
    text_inp = torch.from_numpy(txt_feat_all).float().unsqueeze(1)
    test_labels = test_dataset.test_cid
    test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False)

    tain_txt_feat_all = tain_txt_feat_all.to(device)
    text_inp = text_inp.to(device)

    net = Attention(dimensions=2048, text_dim=7551).to(device)
    # net.apply(init_weights_xavier)

    #     optimizer = torch.optim.Adam(net.parameters(), lr=0.0005, betas=(0.5, 0.9), weight_decay=0.0001)
    optimizer = torch.optim.Adam(itertools.chain(net.parameters(), tf.parameters()), lr=0.0005, betas=(0.5, 0.9),
                                 weight_decay=0.09)

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
                # print(attention_weights.shape)
                # print(temp_feat_all)
                # print(attention_weights)
                # print(loss, correct * 1.0 / img_feat.shape[0])
                # break
                # tepoch.set_postfix(loss=loss.item())
                # print(90)

        acc = correct * 1.0 / len(train_loader.dataset)
        print('Accuracy: {}\n'.format(acc))

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
        print('Accuracy: {}\n'.format(acc))
        zsl_acc.append(acc)


def main():
    train_zest_style()


if __name__ == "__main__":
    main()
