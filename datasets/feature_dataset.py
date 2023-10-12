

import torch
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import click
import scipy.io as sio
import pickle


class FeatureDataset(Dataset):

    def __init__(self, mapping):

        # load features
        img_feat_path, img_col = mapping['image_feature']

        self.image_features = torch.from_numpy(np.array(sio.loadmat(img_feat_path)[img_col]).astype(np.float32))

        txt_feat_path, txt_col = mapping['text_feature']
        self.text_features = torch.from_numpy(np.array(sio.loadmat(txt_feat_path)[txt_col]).astype(np.float32))

        print(self.image_features.shape)
        print(self.text_features.shape)

    def __len__(self):
        return self.image_features.shape[0]

    def __getitem__(self, idx):
        img_feature, txt_feature = self.image_features[idx, :], self.text_features[idx, :]
        return img_feature, txt_feature


if __name__ == '__main__':
    run_test = True
    if run_test:

        # baseline run test
        train_feature_path = 'data/CUB_baseline/train_cub_googlenet_bn.mat'
        train_text_path = 'data/CUB_baseline/train_attr.mat'
        mapping = {'image_feature': ('data/CUB_baseline/train_cub_googlenet_bn.mat', 'train_cub_googlenet_bn') , 'text_feature': ('data/CUB_baseline/train_attr.mat','train_attr')}
        dataset = FeatureDataset(mapping)
        img_f, txt_f = dataset[0]
        print(img_f.shape)
        print(txt_f.shape)
