
import torch
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import click
import scipy.io as sio
import pickle

from sklearn.cluster import DBSCAN
from sklearn import metrics
import hdbscan


class ResnetDataset(Dataset):

    def __init__(self, feature_path='data/CUB2011/res101.mat', split='easy', model='vanilla', normalize=False, train=True):
        
        self.train = train
        
        if split == 'easy':
            train_test_split_dir = 'data/CUB2011/train_test_split_easy.mat'
        else:
            train_test_split_dir = 'data/CUB2011/train_test_split_hard.mat'
            
        train_test_split = sio.loadmat(train_test_split_dir)
        self.train_cid = train_test_split['train_cid'].squeeze() - 1 # labels in training
        self.test_cid = train_test_split['test_cid'].squeeze() - 1 # labels in testing
        matcontent = sio.loadmat(feature_path)
        self.resnet_features = matcontent['features'].T.astype(np.float32)
        
    

        labels = matcontent['labels'].astype(np.uint8).squeeze() - 1
        g_to_c = create_mapping()
        all_labels = np.zeros_like(labels)
        for i, label in enumerate(labels):
            all_labels[i] = g_to_c[label]
            
        mask_train = np.isin(all_labels, self.train_cid)
        mask_test = np.isin(all_labels, self.test_cid)
        
        self.resnet_features_train = self.resnet_features[mask_train]
        self.resnet_features_test = self.resnet_features[mask_test]
        
        self.labels_train = all_labels[mask_train]
        self.labels_test = all_labels[mask_test]
        
        if normalize:
            mean = self.resnet_features_train.mean()
            var = self.resnet_features_train.var()
            self.resnet_features_train = (self.resnet_features_train - mean)/var
            self.resnet_features_test = (self.resnet_features_test - mean)/var
        

#         if self.train:
#             mask = np.isin(all_labels, self.train_cid)
#             self.resnet_features = self.resnet_features[mask]
#             self.labels = all_labels[mask]
#         else:
#             mask = np.isin(all_labels, self.test_cid)
#             self.resnet_features = self.resnet_features[mask]
#             self.labels = all_labels[mask]
        
        txt_feat_path = 'data/CUB2011/CUB_Porter_7551D_TFIDF_new_original.mat'
        txt_feat_path_original = r"data/CUB2011/CUB_Porter_7551D_TFIDF_new_original.mat"
        similarity_flag = 0 if model == 'vanilla' else 1
        self.train_text_feature, self.test_text_feature = get_text_feature(txt_feat_path, train_test_split_dir, txt_feat_path_original, similarity_flag)
        self.text_dim = self.train_text_feature.shape[1]


    def __len__(self):
        if self.train:
            return len(self.labels_train)
        else:
            return len(self.labels_test)

    def __getitem__(self, idx):
#         orig_label = self.labels[idx]
        if self.train:
            orig_label = self.labels_train[idx]
            curr_label = np.where(self.train_cid == orig_label)[0] 
            return self.resnet_features_train[idx], curr_label.squeeze(), orig_label, self.train_text_feature[curr_label, :].squeeze()
        else:
            orig_label = self.labels_test[idx]
            curr_label = np.where(self.test_cid == orig_label)[0]
            return self.resnet_features_test[idx], curr_label.squeeze(), orig_label, self.test_text_feature[curr_label, :].squeeze()
    
    def get_all_data(self):
        if self.train: 
            curr_labels = np.searchsorted(self.train_cid, self.labels_train)
            return self.resnet_features_train, curr_labels, self.labels_train, self.train_text_feature
        else:
            curr_labels = np.searchsorted(self.test_cid, self.labels_test)
            return self.resnet_features_test, curr_labels, self.labels_test, self.test_text_feature
        
    

def get_text_feature(dir, train_test_split_dir,txt_feat_path_original, add_similarity_flag=True):
    train_test_split = sio.loadmat(train_test_split_dir)
    train_cid = train_test_split['train_cid'].squeeze()-1
    test_cid = train_test_split['test_cid'].squeeze()-1

    text_feature = sio.loadmat(dir)['PredicateMatrix']
    text_feature_original=sio.loadmat(txt_feat_path_original)['PredicateMatrix']

    text_feature_new, intersections=add_similarity(text_feature,train_cid,test_cid,text_feature_original)
    if intersections.shape[0]/test_cid.shape[0]>0.15 and add_similarity_flag:
        print ('added similarity features')
        text_feature=text_feature_new

    train_text_feature = text_feature[train_cid]  # 0-based index
    test_text_feature = text_feature[test_cid]  # 0-based index

    return train_text_feature.astype(np.float32), test_text_feature.astype(np.float32)


def family_addition_features(text_features, path_text):
    files = os.listdir(path_text)
    files_xls = [f for f in files if f[-3:] == 'txt']
    files_xls.sort(key=lambda x: int(x.split(".")[0]))

    dic_family = {}
    list_family = []
    for counter, file_name in enumerate(files_xls):
        family = file_name.split("_")[-1]
        if family not in dic_family:
            dic_family[family] = len(dic_family)
        list_family.append(dic_family[family])


    family_fetures=np.zeros((len(list_family),len(dic_family)))
    for l_i,label in enumerate(list_family):
        zero_current=np.zeros(len(dic_family))
        zero_current[label]=1
        family_fetures[l_i]=zero_current


    return np.concatenate((family_fetures,text_features),axis=1)



def evaluate_cluster(clustering, path_dir_text):
    from sklearn.metrics import accuracy_score
    import operator

    cluster_labels=[]
    counter_max_label=max(clustering.labels_)+1
    for i in clustering.labels_:
        if i==-1:
            label=counter_max_label
            counter_max_label+=1
        else:
            label=i
        cluster_labels.append(label)

    files = os.listdir(path_dir_text)
    files_xls = [f for f in files if f[-3:] == 'txt']

    files_xls.sort(key=lambda x: int(x.split(".")[0]))

    ground_truth=[]
    prdeiction_label={}
    counter_family=-1
    dic_family={}
    list_family=[]
    for counter,file_name in enumerate(files_xls):
        # print (counter)
        family=file_name.split("_")[-1]
        if family not in dic_family:
            counter_family+=1
            dic_family[family]=counter_family
        list_family.append(dic_family[family])

        if family not in prdeiction_label:
            prdeiction_label[family]={}
        if cluster_labels[counter] not in prdeiction_label[family]:
            prdeiction_label[family][cluster_labels[counter]]=0

        prdeiction_label[family][cluster_labels[counter]] +=1

    for counter,file_name in enumerate(files_xls):
        family=file_name.split("_")[-1]


        cluster_id=max(prdeiction_label[family].items(), key=operator.itemgetter(1))[0]


        ground_truth.append(cluster_id)


    accuracy_score=accuracy_score(cluster_labels,ground_truth)
    print ("cluaster accuracy: ",accuracy_score )
    print ("cluster cluster:", metrics.adjusted_rand_score(list_family, clustering.labels_))
    print ("cluster adjusted_mutual_info_score:",metrics.adjusted_mutual_info_score(list_family, clustering.labels_))



def add_similarity(text_feat,train_id, test_id,original_tetx):
    eps = 0.65
    clustering = hdbscan.HDBSCAN(min_cluster_size=2).fit(original_tetx)

    clustering_2 = DBSCAN(eps=eps, min_samples=2, metric="cosine").fit(original_tetx)

    ids_train = clustering.labels_[train_id]
    ids_test = clustering.labels_[test_id]
    (clusters_train, counts_train) = np.unique(ids_train, return_counts=True)
    (cluster_test, counts_test) = np.unique(ids_test, return_counts=True)
    intersections=np.intersect1d(cluster_test, clusters_train)

    n_clusters=np.unique(clustering.labels_, return_counts=False).shape[0]
    n_clusters_2=np.unique(clustering_2.labels_, return_counts=False).shape[0]


    family = np.zeros((text_feat.shape[0], n_clusters))
    family_2 = np.zeros((text_feat.shape[0], n_clusters_2))
    for i in range(text_feat.shape[0]):
        np_zero = np.zeros(n_clusters)
        text_cluster=clustering.labels_[i]
        if text_cluster!=-1:
            np_zero[text_cluster] = 1
        family[i] = np_zero

        np_zero_2 = np.zeros(n_clusters_2)
        text_cluster_2=clustering_2.labels_[i]
        if text_cluster_2!=-1:
            np_zero_2[text_cluster_2] = 1
        family_2[i] = np_zero_2

    text_feat = np.concatenate((family_2, family, text_feat), axis=1)
    return text_feat,intersections


def create_mapping():
    with open('data/CUB2011/general.txt') as f:
        g_lines = [line.rstrip().split('.')[0] for line in f]

    with open('data/CUB2011/conventional.txt') as f:
         c_lines = [line.rstrip().split('.')[0] for line in f]

    g_to_c = {}
    for i, v in enumerate(g_lines):
        g_to_c[i] = int(v)-1

    # print(g_lines)
    # print(c_lines)
    # print(g_lines[21])
    # print(c_lines[int(g_lines[21])-1])

    return g_to_c


def test_data():
    feature_path='data/CUB2011/res101.mat'
    matcontent = sio.loadmat(feature_path)
#     resnet_features = matcontent['features'].T.astype(np.float32)
    labels = matcontent['labels'].astype(np.uint8).squeeze() - 1
    g_to_c = create_mapping()
    all_labels = np.zeros_like(labels)
    for i, label in enumerate(labels):
        all_labels[i] = g_to_c[label]

    values, counts = np.unique(all_labels, return_counts=True)
    
    pfc_label_path_train = 'data/CUB2011/labels_train.pkl'
    pfc_label_path_test = 'data/CUB2011/labels_test.pkl'
    with open(pfc_label_path_train, 'rb') as fout1, open(pfc_label_path_test, 'rb') as fout2:
        labels_train = pickle.load(fout1, encoding='latin1')
        labels_test = pickle.load(fout2, encoding='latin1')
    
    print(len(labels_train))
    train_test_split_dir = 'data/CUB2011/train_test_split_easy.mat'
    train_test_split = sio.loadmat(train_test_split_dir)
    train_cid = train_test_split['train_cid'].squeeze() - 1 # labels in training
    test_cid = train_test_split['test_cid'].squeeze() - 1 # labels in testing
    
    labels_train_orig = train_cid[labels_train]
    labels_test_orig = test_cid[labels_test]
    labels_2 = np.concatenate((labels_train_orig, labels_test_orig), axis=0)
    values_2, counts_2 = np.unique(labels_2, return_counts=True)
    for c1, c2 in zip(counts, counts_2):
        print(c2-c1)
    print(counts)
    print(counts_2)
    print((values_2 - values).sum())
    
if __name__ == '__main__':
    run_test = True
    if run_test:
        train_dataset = ResnetDataset()
        train_dataset.get_all_data()
        f, curr_l, orig_l, t = train_dataset[0]
        print(f.shape)
        print(t.shape)
        print(curr_l.shape)
        print(orig_l.shape)
        print(len(train_dataset))
        print(np.unique(f))
#         print('\nTestingDataset')
#         test_dataset = Resnet_Dataset(train=False)
#         f, curr_l, orig_l, t = test_dataset[0]
#         print(f.shape)
#         print(t.shape)
#         print(curr_l)
#         print(orig_l)
#         print(len(test_dataset))
        
#         # iterate through entire dataset
#         for i in range(len(train_dataset)):
#             d = train_dataset[i]
            
#         for i in range(len(test_dataset)):
#             d = test_dataset[i]
            
#         print('\nEnumeration Succesful')
