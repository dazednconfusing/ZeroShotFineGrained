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
print(DEVICE)

def init_weights_xavier(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


class Baseline2Net(nn.Module):

    def __init__(self, input_dim, hidden_dim_1, output_dim):
        super(Baseline2Net, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim_1)
        self.fc2 = nn.Linear(hidden_dim_1, output_dim)
        self.fc3 = nn.Linear(output_dim, 150)


        # initialize weights
        # torch.nn.init.normal_(self.fc1.weight, 0, 0.02)
        # torch.nn.init.normal_(self.fc2.weight, 0, 0.02)
        #
        # self.fc1.bias.data.fill_(0)
        # self.fc2.bias.data.fill_(0)
        # todo: can create overall initialize function to experiment with differnet initializations

    def forward(self, x):

        x = F.relu(self.fc1(x))
        feat_proj = F.relu(self.fc2(x))
        out = self.fc3(feat_proj)

        return feat_proj, out


def train_tf_idf():
    #     train_dataset = CUBFeatDataSet(folder_path='data/CUB2011', split='easy', model='vanilla', train=True)
    train_dataset = ResnetDataset(split='easy', train=True)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, pin_memory=True)
    
    I_all_train, label_all_train, orig_label_all_train, txt_feat_all_train = train_dataset.get_all_data()
    text_inp_train = torch.from_numpy(txt_feat_all_train).float()
    train_labels = train_dataset.train_cid 

    train_dataset_size = len(train_dataset)

#     test_dataset = CUBFeatDataSet(folder_path='data/CUB2011', split='easy', model='vanilla', train=False)
    test_dataset = ResnetDataset(split='easy', train=False)
    test_dataset_size = len(test_dataset)
    I_all, label_all, orig_label_all, txt_feat_all = test_dataset.get_all_data()
    text_inp = torch.from_numpy(txt_feat_all).float()
    test_labels = test_dataset.test_cid
    
    plot = False
    
#     net = Baseline2Net(7551, 4096, 3584).to(DEVICE)
    net = Baseline2Net(7551, 4096, 2048).to(DEVICE)
    net.apply(init_weights_xavier)
   

    optimizer = torch.optim.Adam(net.parameters(), lr=0.0005, betas=(0.5, 0.9), weight_decay=0.0001)
    criterion_1 = nn.MSELoss()
    criterion_2 = nn.CrossEntropyLoss()

    epochs = 600
    zsl_acc = []
    epoch_losses = []
    mse_losses = []
    ce_losses = []
    for epoch in range(epochs):
        net.train()

        epoch_loss = 0.0
        mse_loss = 0.0
        ce_loss = 0.0
        with tqdm(train_loader, unit='batch') as tepoch:
            for img_feat, label, orig_label, txt_feat in tepoch:
                img_feat = img_feat.to(DEVICE)
                txt_feat = txt_feat.to(DEVICE)
                label = label.to(DEVICE)
                feat, output = net(txt_feat)
                optimizer.zero_grad()
                loss_1 = criterion_1(feat, img_feat)
                loss_2 = criterion_2(output, label.long())
                loss = loss_1 + loss_2
                loss.backward()
                optimizer.step()

                loss_d = {'MSE Loss': loss_1.item(), 'CE Loss': loss_2.item(), 'Total Loss': loss.item()}
                tepoch.set_description('EPOCH: %i' % epoch)
                tepoch.set_postfix(loss_d)

                epoch_loss += output.shape[0] * loss.item()
                mse_loss += output.shape[0] * loss_1.item()
                ce_loss += output.shape[0] * loss_2.item()

        epoch_loss /= train_dataset_size
        mse_loss /= train_dataset_size
        ce_loss /= train_dataset_size
        
        epoch_losses.append(epoch_loss)
        mse_losses.append(mse_loss)
        ce_losses.append(ce_loss)
        
        print('Epoch Loss: {}\n'.format(epoch_loss))
        
        
        if not epoch % 10:
            net.eval()

            print('Evaluation\n')
                # Train Evaluation

            text_inp_train = text_inp_train.to(DEVICE)
            text_pred_train, _ = net(text_inp_train)
            outpred_train = [0] * train_dataset_size
            for i in range(train_dataset_size):
                outputLabel = knn.kNNClassify(I_all_train[i, :], text_pred_train.cpu().data.numpy(), train_labels, 1)
                outpred_train[i] = outputLabel

            outpred_train = np.array(outpred_train)
            acc_train = np.equal(outpred_train, orig_label_all_train).mean()
            print('Train Accuracy: {}\n'.format(acc_train))

            text_inp = text_inp.to(DEVICE)
            text_pred, _ = net(text_inp)
            outpred = [0] * test_dataset_size
            for i in range(test_dataset_size):
                outputLabel = knn.kNNClassify(I_all[i, :], text_pred.cpu().data.numpy(), test_labels, 1)
                outpred[i] = outputLabel
            outpred = np.array(outpred)
            acc = np.equal(outpred, orig_label_all).mean()
            print('Test Accuracy: {}\n'.format(acc))
            zsl_acc.append(acc)

            # save model
            if plot and not epoch % 50:
                model_path = 'saved_models/model_{}'.format(epoch) + '.pt'
                torch.save({'epoch': epoch,
                           'model_state_dict': net.state_dict(),
                           'optimizer_state_dict': optimizer.state_dict(),
                           'loss': loss}, model_path)

                path_1 = 'plots/Weight_Loss_{}'.format(epoch) + '.png'
                path_2 = 'plots/Weighted_accuracy_{}'.format(epoch) + '.png'

                plt.plot(list(range(epoch+1)), epoch_losses, label='Total Loss')
                plt.plot(list(range(epoch+1)), mse_losses, label='MSE Loss')
                plt.plot(list(range(epoch+1)), ce_losses, label='CE Loss')
                plt.title('VPDE, TFIDF, BSE+CE Unweighted')
                plt.xlabel('Number of Epoch')
                plt.ylabel('Loss')
                plt.legend()
                plt.savefig(path_1)
                plt.clf()

                plt.plot(list(range(epoch+1)), zsl_acc)
                plt.title('VPDE, TFIDF, BSE+CE Unweighted')
                plt.xlabel('Number of Epoch')
                plt.ylabel('Accuracy')
                plt.savefig(path_2)
                plt.clf()
    


if __name__ == '__main__':
    run_train = True
    if run_train:
        train_tf_idf()
