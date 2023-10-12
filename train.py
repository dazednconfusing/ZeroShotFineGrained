
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import numpy as np

from datasets.cub_dataset import CUBDataset

if __name__ == '__main__':

    train_feature_path = 'data/CUB2011/pfc_feat_train.mat'
    train_label_path = 'data/CUB2011/labels_train.pkl'
    train_dataset = CUBDataset(train_feature_path, train_label_path)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)  # num_workers

    for i, (input, labels) in enumerate(train_loader):
        print(input.shape)
        print(labels.shape)

