
import os
import pickle

import click
import numpy as np
import scipy.io as sio
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, utils


class CUBDataset(Dataset):

    def __init__(self, feat_path, label_path, normalize=True):
        self.feat_path = feat_path
        self.label_path = label_path

        # load features
        self.image_features = sio.loadmat(
            feat_path)['pfc_feat'].astype(np.float32)

        # load labels
        with open(self.label_path, 'rb') as f:
            self.labels = pickle.load(f, encoding='latin1')

        mean = self.image_features.mean()
        var = self.image_features.var()

        self.image_features = (self.image_features - mean)/var
        print(self.image_features.shape)
        print(self.labels.shape)
        # todo: add centroid of image features
        # todo: include text embeddings

    def __len__(self):
        return self.image_features.shape[0]

    def __getitem__(self, idx):
        feature, label = self.image_features[idx, :], self.labels[idx]
        return feature, label


if __name__ == '__main__':
    run_test = True
    if run_test:
        train_feature_path = 'data/CUB2011/pfc_feat_train.mat'
        train_label_path = 'data/CUB2011/labels_train.pkl'
        dataset = CUBDataset(train_feature_path, train_label_path)
        feature, label = dataset[0]
        print(feature.shape)
        print(label)
        print(len(dataset))
