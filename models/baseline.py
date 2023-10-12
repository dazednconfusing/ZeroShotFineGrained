

# Learning a Deep Embedding Model for Zero-Shot Learning

import sys
import os

sys.path.append(os.getcwd())

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import scipy.io as sio
import matplotlib.pyplot as plt

from datasets.feature_dataset import FeatureDataset
from datasets.resnet_dataset import ResnetDataset
from dataloaders.CUBLoader import CUBFeatDataSet
import utils.knn as knn

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def init_weights_xavier(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


class BaselineNet(nn.Module):

    def __init__(self, input_dim, hidden_dim_1, output_dim):
        super(BaselineNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim_1)
        self.fc2 = nn.Linear(hidden_dim_1, output_dim)


        # initialize weights
        # torch.nn.init.normal_(self.fc1.weight, 0, 0.02)
        # torch.nn.init.normal_(self.fc2.weight, 0, 0.02)
        #
        # self.fc1.bias.data.fill_(0)
        # self.fc2.bias.data.fill_(0)
        # todo: can create overall initialize function to experiment with differnet initializations

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # x = self.fc2(x)

        return x


def train_tf_idf():
    train_dataset = CUBFeatDataSet(folder_path='data/CUB2011', split='easy', model='vanilla', train=True)
#     train_dataset = ResnetDataset(split='easy', train=True)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, pin_memory=True)
    
    I_all_train, label_all_train, orig_label_all_train, txt_feat_all_train = train_dataset.get_all_data()
    print(I_all_train.shape)
    text_inp_train = torch.from_numpy(txt_feat_all_train).float()
    train_labels = train_dataset.train_cid 

    train_dataset_size = len(train_dataset)

    test_dataset = CUBFeatDataSet(folder_path='data/CUB2011', split='easy', model='vanilla', train=False)
#     test_dataset = ResnetDataset(split='easy', train=False)
    test_dataset_size = len(test_dataset)
    I_all, label_all, orig_label_all, txt_feat_all = test_dataset.get_all_data()
    text_inp = torch.from_numpy(txt_feat_all).float()
    test_labels = test_dataset.test_cid
    
    
#     net = BaselineNet(7551, 4096, 2048).to(DEVICE) # for resnet
    net = BaselineNet(7551, 4096, 3584).to(DEVICE)
    net.apply(init_weights_xavier)

       # 0.0005
    optimizer = torch.optim.Adam(net.parameters(), lr=0.0005, betas=(0.5, 0.9), weight_decay=0.0001)
#     optimizer = torch.optim.Adam(net.parameters(), lr=0.001, betas=(0.5, 0.9), weight_decay=0.01)
    criterion = nn.MSELoss()
    
    plot = False

    epochs = 600
    zsl_acc = []
    epoch_losses = []
    mse_losses = []
    ce_losses = []
    for epoch in range(epochs):
        net.train()

        epoch_loss = 0.0
        with tqdm(train_loader, unit='batch') as tepoch:
            for img_feat, label, orig_label, txt_feat in tepoch:
                img_feat = img_feat.to(DEVICE)
                txt_feat = txt_feat.to(DEVICE)
                output = net(txt_feat)
                
                optimizer.zero_grad()
                loss = criterion(output, img_feat)
                loss.backward()
                optimizer.step()

                loss_d = {'Loss': loss.item()}
                tepoch.set_description('EPOCH: %i' % epoch)
                tepoch.set_postfix(loss_d)

                epoch_loss += output.shape[0] * loss.item()


        epoch_loss /= train_dataset_size
        epoch_losses.append(epoch_loss)
        
        print('Epoch Loss: {}\n'.format(epoch_loss))
        
        eval_epoch = 5
        save_epoch = 20
        if not epoch % eval_epoch:
            net.eval()
            print('Evaluation\n')
            # Train Evaluation

            text_inp_train = text_inp_train.to(DEVICE)
            text_pred_train = net(text_inp_train)
            outpred_train = [0] * train_dataset_size
            for i in range(train_dataset_size):
                outputLabel = knn.kNNClassify(I_all_train[i, :], text_pred_train.cpu().data.numpy(), train_labels, 1)
                outpred_train[i] = outputLabel
            outpred_train = np.array(outpred_train)
            acc_train = np.equal(outpred_train, orig_label_all_train).mean()
            print('Train Accuracy: {}\n'.format(acc_train))

            # Test valuation

            text_inp = text_inp.to(DEVICE)
            text_pred = net(text_inp)
            outpred = [0] * test_dataset_size
            for i in range(test_dataset_size):
                outputLabel = knn.kNNClassify(I_all[i, :], text_pred.cpu().data.numpy(), test_labels, 1)
                outpred[i] = outputLabel
            outpred = np.array(outpred)
            acc = np.equal(outpred, orig_label_all).mean()
            print('Test Accuracy: {}\n'.format(acc))
            zsl_acc.append(acc)

            # save model
            if plot and not epoch % save_epoch:
                model_path = 'saved_models/model_easy_resnet{}'.format(epoch) + '.pt'
                torch.save({'epoch': epoch,
                           'model_state_dict': net.state_dict(),
                           'optimizer_state_dict': optimizer.state_dict(),
                           'loss': loss}, model_path)

                path_1 = 'plots/Baseline(Resnet,TFIDF_easy)_Loss{}'.format(epoch) + '.png'
                path_2 = 'plots/Baseline(Resnet,TFIDF_easy)_accuracy{}'.format(epoch) + '.png'
                

                plt.plot(np.arange(len(epoch_losses)), epoch_losses, label='Total Loss')
                plt.title('VPDE, TFIDF Simple')
                plt.xlabel('Number of Epoch')
                plt.ylabel('Loss')
                plt.legend()
                plt.savefig(path_1)
                plt.clf()

                plt.plot(np.arange(len(zsl_acc))*eval_epoch, zsl_acc)
                plt.title('VPDE, TFIDF Simple')
                plt.xlabel('Number of Epoch')
                plt.ylabel('Accuracy')
                plt.savefig(path_2)
                plt.clf()

def train():

    # train data
    train_img_path = 'data/CUB_baseline/train_cub_googlenet_bn.mat'
    train_text_path = 'data/CUB_baseline/train_attr.mat'
    mapping = {'image_feature': (train_img_path, 'train_cub_googlenet_bn') , 'text_feature': (train_text_path, 'train_attr')}
    train_dataset = FeatureDataset(mapping)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # test data

    test_img_path = 'data/CUB_baseline/test_cub_googlenet_bn.mat'
    test_img_feat = np.array(sio.loadmat(test_img_path)['test_cub_googlenet_bn'])

    test_text_path = 'data/CUB_baseline/test_proto.mat'
    test_text_feat = np.array(sio.loadmat(test_text_path)['test_proto'])

    test_text_inp = torch.from_numpy(test_text_feat).float() # input to net


    test_label_path = 'data/CUB_baseline/test_labels_cub.mat'
    test_x2label = np.squeeze(np.array(sio.loadmat(test_label_path)['test_labels_cub']))

    test_x2label = test_x2label.astype("float32")

    test_classes_path = 'data/CUB_baseline/testclasses_id.mat'
    test_att2label = np.squeeze(np.array(sio.loadmat(test_classes_path)['testclasses_id']))


    print(test_att2label.shape)  # 50 labels
    print(np.unique(test_att2label))
    print(test_x2label.shape)
    print(np.unique(test_x2label))

    net = BaselineNet(312, 700, 1024)

    optimizer = torch.optim.Adam(net.parameters(), lr=1e-5, weight_decay=1e-2)
    criterion = nn.MSELoss()

    epochs = 60
    zsl_acc = []
    for epoch in range(epochs):
        net.train()

        with tqdm(train_loader, unit='batch') as tepoch:
            for img_feat, txt_feat in tepoch:
                output = net(txt_feat)
                optimizer.zero_grad()
                loss = criterion(output, img_feat)
                loss.backward()
                optimizer.step()
                tepoch.set_postfix(loss=loss.item())

        # evaluation step
        net.eval()

        att_pred = net(test_text_inp)
        outpred = [0] * 2933
        for i in range(2933):
            outputLabel = knn.kNNClassify(test_img_feat[i, :], att_pred.cpu().data.numpy(), test_att2label, 1)
            outpred[i] = outputLabel
        outpred = np.array(outpred)
        acc = np.equal(outpred, test_x2label).mean()
        print('Accuracy: {}\n'.format(acc))
        zsl_acc.append(acc)

    # save plot of accuracies
    plt.plot(list(range(epochs)), zsl_acc)
    plt.title('Baseline CUB Attribute, Image Feature Vectors')
    plt.xlabel('Number of Epoch')
    plt.ylabel('ZSL Accuracy')
    plt.savefig('plots/Baseline_CUB.png')


if __name__ == '__main__':
    run_train = True
    run_fake = False
    if run_train:
        # train()
        train_tf_idf()

    if run_fake:
        fake_data = torch.randn(32, 312) # batch_size X attribute_vector
        net = BaselineNet(312, 700, 1024)
        output = net(fake_data)
        print(output.shape) # 32 X 1024

    # todo: create an overall test function
    # todo: write separate train and evaluate function for each dataset function to make train loop modular

    # Reasons why tf-df is not wokring
    # 1. Img_features of baseline are only positive, not true for VPDE-net which is why ReLU function might not work
    # 2. Loss is low giving smaller gradients, hence no learning; loss for baseline (AwA) starts much higher
    # 3. tf-idf vectors have much smaller values than attribute vectors

    '''Fixes that did not work 
    1. Removing ReLU activation from last layer 
        Epochs: 20
        Loss: 0.0278
        Acc: 0.02045 (converged)
    2. Removing normalization from CUBLoader and using normal Baseline model
        Learning rate: 1e-4
            Loss: 56.2 (converged)
            Acc: 0.032
        Learning rate: 1e-3
            Loss:
            Acc:
    '''
