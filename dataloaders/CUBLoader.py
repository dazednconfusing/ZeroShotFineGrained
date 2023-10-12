import os
import pickle
import sys

import hdbscan
import numpy as np
import pandas as pd
import scipy.io as sio
import torch
from preprocessing.embedder import Embedder
from sklearn import metrics
from sklearn.cluster import DBSCAN
from torch.utils.data import Dataset
from utils.fileloaders import load_image

s = os.getcwd()


def limit_bert_feature(feature, size):
    if feature.shape[0] > size:
        return feature[:size, :]
    else:
        out = torch.zeros(size, feature.shape[1])
        out[: feature.shape[0], :] = feature
        return out


def read_captions(file_name):
    f = open(file_name, "r")
    data = [i.strip() for i in f.readlines()]
    return data


def get_caption_name(file_name):
    prefix = file_name[:-4] + ".txt"
    return prefix


def get_class_name(file_name):
    label = file_name.split("/")[0]
    label_id, label_name = label.split(".")
    return int(label_id) - 1, label_name


def read_image_indexes(file_name, data_dir, caption_dir):
    f = open(file_name, "r")
    data = f.readlines()
    data = [line.strip().split(" ") for line in data]
    # 0-index, path of file, label,
    data = [
        [int(line[0]) - 1, os.path.join(data_dir, line[1])]
        + list(get_class_name(line[1]))
        + [get_caption_name(line[1])]
        for line in data
    ]
    data = [
        DataRecord(
            i[0],
            i[1],
            i[2],
            i[3],
            captions=read_captions(os.path.join(caption_dir, i[4])),
        )
        for i in data
    ]
    return data


def read_label_xlsx(file):
    df = pd.read_excel(file)
    records = []
    for i in range(len(df)):
        dfr = df.loc[i]
        record = SentenceRecord(
            i,
            dfr["score"],
            dfr["for_summarization"],
            dfr["sentences"],
            dfr["tf_for_summarization"],
        )
        records.append(record)
    label = file.split("/")[-1].split(".")[0].split("_")[0]
    lrecord = LabelRecord(int(label), records)
    return lrecord


def read_label_info(dir_name):
    files = [(i, os.path.join(dir_name, i)) for i in os.listdir(dir_name)]
    files = [i for i in files if os.path.isfile(i[1])]
    records = [read_label_xlsx(i[1]) for i in files]
    records = {i.label: i for i in records}
    return records


class DataRecord:
    def __init__(self, index, file_path, label, label_info, captions=None):
        if captions is None:
            captions = []
        self.index = index
        self.file_path = file_path
        self.label = label
        self.label_info = label_info
        self.captions = captions


class LabelRecord:
    def __init__(self, label, records):
        self.label = label
        self.records = records


class SentenceRecord:
    def __init__(self, index, score, for_summ, text, tf):
        self.index = index
        self.score = score
        self.for_summ = for_summ
        self.text = text
        self.tf = tf


class CUBLabelSet(Dataset):
    def __init__(self, folder_path):
        super().__init__()
        self.folder_path = folder_path
        self.label_info = read_label_info(
            os.path.join(self.folder_path, "wikipedia_HUMAN")
        )

    def __len__(self):
        return len(self.label_info)

    # send label id (0-199)
    def __getitem__(self, index):
        return self.label_info[index]


class CUBDataSet(Dataset):
    def __init__(self, folder_path, image_transforms=None):
        super().__init__()
        self.folder_path = folder_path
        self.data_dir = os.path.join(folder_path, "CUB_200_2011")
        self.image_file = os.path.join(self.data_dir, "images.txt")
        self.image_dir = os.path.join(self.data_dir, "images")
        self.caption_dir = os.path.join(self.folder_path, "text_c10")

        self.data = read_image_indexes(
            self.image_file, self.image_dir, self.caption_dir
        )
        self.image_transforms = image_transforms
        self.label_data_set = CUBLabelSet(folder_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data_record = self.data[index]
        image = load_image(data_record.file_path)
        if self.image_transforms:
            image = self.image_transforms(image)
        return image, data_record, self.label_data_set[data_record.label]


class NABFeatDataSet(Dataset):
    def __init__(self, folder_path, split="easy", model="vanilla", train=True):
        self.train = train
        txt_feat_path_original = r"NAB_Porter_13217D_TFIDF_new_original.mat"

        if split == "easy":
            if model == "similarity_VRS":  # 'similarity_VRS'
                txt_feat_path = "NAB_EASY_SPLIT_VRS.mat"
            else:  # vanilla\similarity
                txt_feat_path = r"NAB_Porter_13217D_TFIDF_new_original.mat"

            train_test_split_dir = "train_test_split_NABird_easy.mat"
            pfc_label_path_train = "labels_train.pkl"
            pfc_label_path_test = "labels_test.pkl"
            pfc_feat_path_train = "pfc_feat_train_easy.mat"
            pfc_feat_path_test = "pfc_feat_test_easy.mat"
            train_cls_num = 323
            test_cls_num = 81
        else:
            if model == "similarity_VRS":  # 'similarity_VRS'
                txt_feat_path = "NAB_HARD_SPLIT_VRS.mat"
            else:  # vanilla\similarity
                txt_feat_path = r"NAB_Porter_13217D_TFIDF_new_original.mat"

            train_test_split_dir = "train_test_split_NABird_hard.mat"
            pfc_label_path_train = "labels_train_hard.pkl"
            pfc_label_path_test = "labels_test_hard.pkl"
            pfc_feat_path_train = "pfc_feat_train_hard.mat"
            pfc_feat_path_test = "pfc_feat_test_hard.mat"
            train_cls_num = 323
            test_cls_num = 81

        txt_feat_path = os.path.join(folder_path, txt_feat_path)
        train_test_split_dir = os.path.join(folder_path, train_test_split_dir)
        txt_feat_path_original = os.path.join(folder_path, txt_feat_path_original)
        pfc_feat_path_train = os.path.join(folder_path, pfc_feat_path_train)
        pfc_feat_path_test = os.path.join(folder_path, pfc_feat_path_test)
        pfc_label_path_train = os.path.join(folder_path, pfc_label_path_train)
        pfc_label_path_test = os.path.join(folder_path, pfc_label_path_test)

        self.pfc_feat_data_train = sio.loadmat(pfc_feat_path_train)["pfc_feat"].astype(
            np.float32
        )
        self.pfc_feat_data_test = sio.loadmat(pfc_feat_path_test)["pfc_feat"].astype(
            np.float32
        )

        self.train_cls_num = train_cls_num
        self.test_cls_num = test_cls_num
        self.feature_dim = self.pfc_feat_data_train.shape[1]
        # calculate the corresponding centroid.
        with open(pfc_label_path_train, "rb") as fout1, open(
            pfc_label_path_test, "rb"
        ) as fout2:
            if sys.version_info >= (3, 0):
                self.labels_train = pickle.load(fout1, encoding="latin1")
                self.labels_test = pickle.load(fout2, encoding="latin1")
            else:
                self.labels_train = pickle.load(fout1)
                self.labels_test = pickle.load(fout2)

        # Normalize feat_data to zero-centered
        #         mean = self.pfc_feat_data_train.mean()
        #         var = self.pfc_feat_data_train.var()
        #         self.pfc_feat_data_train = (self.pfc_feat_data_train - mean) / var
        #         self.pfc_feat_data_test = (self.pfc_feat_data_test - mean) / var

        train_test_split = sio.loadmat(train_test_split_dir)
        self.train_cid = train_test_split["train_cid"].squeeze() - 1
        self.test_cid = train_test_split["test_cid"].squeeze() - 1

        similarity_flag = 0 if model == "vanilla" else 1
        self.train_text_feature, self.test_text_feature = get_text_feature(
            txt_feat_path, train_test_split_dir, txt_feat_path_original, similarity_flag
        )
        self.text_dim = self.train_text_feature.shape[1]

    def __len__(self):
        if self.train:
            return len(self.pfc_feat_data_train)
        else:
            return len(self.pfc_feat_data_test)

    def __getitem__(self, index):
        "Generates one sample of data"
        # Select sample
        if self.train:
            img_feat = torch.from_numpy(self.pfc_feat_data_train[index])
            label = torch.tensor(self.labels_train[index], dtype=torch.float64)
            orig_label = torch.tensor(
                self.train_cid[self.labels_train[index]], dtype=torch.float64
            )
            return img_feat, label, orig_label
        else:
            img_feat = torch.from_numpy(self.pfc_feat_data_test[index])
            label = torch.tensor(self.labels_test[index], dtype=torch.float64)
            orig_label = torch.tensor(
                self.test_cid[self.labels_test[index]], dtype=torch.float64
            )
            return img_feat, label, orig_label


class CUBFeatDataSet(Dataset):
    def __init__(
        self, folder_path, split="easy", model="vanilla", train=True, size_limit=15
    ):
        self.train = train
        self.label_data_set = CUBLabelSet(folder_path)
        self.embedder = Embedder()
        bert_feat_train, bert_feat_test = None, None
        txt_feat_path_original = r"CUB_Porter_7551D_TFIDF_new_original.mat"

        if split == "easy":
            if model == "similarity_VRS":  # 'similarity_VRS'
                txt_feat_path = r"CUB_EASY_SPLIT_VRS.mat"
            else:  # vanilla\similarity
                txt_feat_path = r"CUB_Porter_7551D_TFIDF_new_original.mat"

            train_test_split_dir = "train_test_split_easy.mat"
            pfc_label_path_train = "labels_train.pkl"
            pfc_label_path_test = "labels_test.pkl"
            pfc_feat_path_train = "pfc_feat_train.mat"
            pfc_feat_path_test = "pfc_feat_test.mat"

            bert_feat_train = "bert_train_feat_easy.p"
            bert_feat_test = "bert_test_feat_easy.p"

            train_cls_num = 150
            test_cls_num = 50
        else:
            if model == "similarity_VRS":  # 'similarity_VRS'
                txt_feat_path = r"CUB_HARD_SPLIT_VRS.mat"
            else:  # vanilla\similarity
                txt_feat_path = r"CUB_Porter_7551D_TFIDF_new_original.mat"

            train_test_split_dir = "train_test_split_hard.mat"
            pfc_label_path_train = "labels_train_hard.pkl"
            pfc_label_path_test = "labels_test_hard.pkl"
            pfc_feat_path_train = "pfc_feat_train_hard.mat"
            pfc_feat_path_test = "pfc_feat_test_hard.mat"

            bert_feat_train = "bert_train_feat_hard.p"
            bert_feat_test = "bert_test_feat_hard.p"

            train_cls_num = 160
            test_cls_num = 40

        txt_feat_path = os.path.join(folder_path, txt_feat_path)
        train_test_split_dir = os.path.join(folder_path, train_test_split_dir)
        txt_feat_path_original = os.path.join(folder_path, txt_feat_path_original)
        pfc_feat_path_train = os.path.join(folder_path, pfc_feat_path_train)
        pfc_feat_path_test = os.path.join(folder_path, pfc_feat_path_test)
        pfc_label_path_train = os.path.join(folder_path, pfc_label_path_train)
        pfc_label_path_test = os.path.join(folder_path, pfc_label_path_test)
        bert_feat_train = os.path.join(folder_path, bert_feat_train)
        bert_feat_test = os.path.join(folder_path, bert_feat_test)

        self.pfc_feat_data_train = sio.loadmat(pfc_feat_path_train)["pfc_feat"].astype(
            np.float32
        )
        self.pfc_feat_data_test = sio.loadmat(pfc_feat_path_test)["pfc_feat"].astype(
            np.float32
        )

        with open(bert_feat_train, "rb") as fout1, open(bert_feat_test, "rb") as fout2:
            if sys.version_info >= (3, 0):
                self.bert_train_feat = pickle.load(fout1, encoding="latin1")
                self.bert_test_feat = pickle.load(fout2, encoding="latin1")
            else:
                self.bert_train_feat = pickle.load(fout1)
                self.bert_test_feat = pickle.load(fout2)

            self.bert_test_feat = torch.stack(
                [limit_bert_feature(i, size_limit) for i in self.bert_test_feat]
            )
            self.bert_train_feat = torch.stack(
                [limit_bert_feature(i, size_limit) for i in self.bert_train_feat]
            )

        self.train_cls_num = train_cls_num
        self.test_cls_num = test_cls_num
        self.feature_dim = self.pfc_feat_data_train.shape[1]

        with open(pfc_label_path_train, "rb") as fout1, open(
            pfc_label_path_test, "rb"
        ) as fout2:
            if sys.version_info >= (3, 0):
                self.labels_train = pickle.load(fout1, encoding="latin1")
                self.labels_test = pickle.load(fout2, encoding="latin1")
            else:
                self.labels_train = pickle.load(fout1)
                self.labels_test = pickle.load(fout2)

        # Normalize feat_data to zero-centered
        mean = self.pfc_feat_data_train.mean()
        var = self.pfc_feat_data_train.var()
        self.pfc_feat_data_train = (self.pfc_feat_data_train - mean) / var
        self.pfc_feat_data_test = (self.pfc_feat_data_test - mean) / var

        # if self.train == True: # todo: make sure this can be removed

        similarity_flag = 0 if model == "vanilla" else 1
        self.train_text_feature, self.test_text_feature = get_text_feature(
            txt_feat_path, train_test_split_dir, txt_feat_path_original, similarity_flag
        )
        self.text_dim = self.train_text_feature.shape[1]

        train_test_split = sio.loadmat(train_test_split_dir)
        self.train_cid = train_test_split["train_cid"].squeeze() - 1
        self.test_cid = train_test_split["test_cid"].squeeze() - 1

        # print(self.train_text_feature.shape)
        # print(self.train_cid) # 150 training ids
        # print(self.train_cid.shape)
        # print('labels_train\n')
        # print(self.labels_train)
        # print(np.unique(self.labels_train))
        # print(self.labels_train.shape)
        # print('End\n')

        self.all_orig_train_labels = self.train_cid
        self.all_orig_test_labels = self.test_cid

        # self.bert_train_feat = []
        # self.bert_test_feat = []

        # for i in range(self.all_orig_train_labels.shape[0]):
        #     label = self.all_orig_train_labels[i].item()
        #     recordSet = self.label_data_set[label]
        #     txt = torch.cat([torch.tensor(self.embedder.get_sentence_embeddings(
        #         entry.text)) for entry in recordSet.records], 0)
        #     self.bert_train_feat.append(txt)

        # for i in range(self.all_orig_test_labels.shape[0]):
        #     label = self.all_orig_test_labels[i].item()
        #     recordSet = self.label_data_set[label]
        #     txt = torch.cat([torch.tensor(self.embedder.get_sentence_embeddings(
        #         entry.text)) for entry in recordSet.records])
        #     self.bert_test_feat.append(txt)

        # pickle.dump(self.bert_train_feat, open("bert_train_feat_hard.p", "wb"))
        # pickle.dump(self.bert_test_feat, open("bert_test_feat_hard.p", "wb"))

    def get_all_data(self):
        if self.train:
            I_all = self.pfc_feat_data_train
            label_all = self.labels_train
            orig_label_all = self.train_cid[self.labels_train]
            return I_all, label_all, orig_label_all, self.train_text_feature
        else:
            I_all = self.pfc_feat_data_test
            label_all = self.labels_test
            orig_label_all = self.test_cid[self.labels_test]
            return I_all, label_all, orig_label_all, self.test_text_feature

    def __len__(self):
        if self.train:
            return len(self.pfc_feat_data_train)
        else:
            return len(self.pfc_feat_data_test)

    def __getitem__(self, index):
        "Generates one sample of data"
        # Select sample
        if self.train:
            img_feat = torch.from_numpy(self.pfc_feat_data_train[index])
            label = torch.tensor(self.labels_train[index], dtype=torch.float64)
            orig_label = self.train_cid[self.labels_train[index]]
            # return img_feat, label, orig_label, self.train_text_feature[int(label), :],
            #  self.label_data_set[orig_label]
            return img_feat, label, orig_label, self.train_text_feature[int(label), :]
        else:
            img_feat = torch.from_numpy(self.pfc_feat_data_test[index])
            label = torch.tensor(self.labels_test[index], dtype=torch.float64)
            orig_label = self.test_cid[self.labels_test[index]]
            # return img_feat, label, orig_label, self.test_text_feature[label, :], self.label_data_set[orig_label]
            return img_feat, label, orig_label, self.test_text_feature[int(label), :]


def get_text_feature(
    dir, train_test_split_dir, txt_feat_path_original, add_similarity_flag=True
):
    train_test_split = sio.loadmat(train_test_split_dir)
    train_cid = train_test_split["train_cid"].squeeze() - 1
    test_cid = train_test_split["test_cid"].squeeze() - 1

    text_feature = sio.loadmat(dir)["PredicateMatrix"]
    text_feature_original = sio.loadmat(txt_feat_path_original)["PredicateMatrix"]

    text_feature_new, intersections = add_similarity(
        text_feature, train_cid, test_cid, text_feature_original
    )
    if intersections.shape[0] / test_cid.shape[0] > 0.15 and add_similarity_flag:
        print("added similarity features")
        text_feature = text_feature_new

    train_text_feature = text_feature[train_cid]  # 0-based index
    test_text_feature = text_feature[test_cid]  # 0-based index

    return train_text_feature.astype(np.float32), test_text_feature.astype(np.float32)


def family_addition_features(text_features, path_text):
    files = os.listdir(path_text)
    files_xls = [f for f in files if f[-3:] == "txt"]
    files_xls.sort(key=lambda x: int(x.split(".")[0]))

    dic_family = {}
    list_family = []
    for _, file_name in enumerate(files_xls):
        family = file_name.split("_")[-1]
        if family not in dic_family:
            dic_family[family] = len(dic_family)
        list_family.append(dic_family[family])

    family_fetures = np.zeros((len(list_family), len(dic_family)))
    for l_i, label in enumerate(list_family):
        zero_current = np.zeros(len(dic_family))
        zero_current[label] = 1
        family_fetures[l_i] = zero_current

    return np.concatenate((family_fetures, text_features), axis=1)


def evaluate_cluster(clustering, path_dir_text):
    import operator

    from sklearn.metrics import accuracy_score

    cluster_labels = []
    counter_max_label = max(clustering.labels_) + 1
    for i in clustering.labels_:
        if i == -1:
            label = counter_max_label
            counter_max_label += 1
        else:
            label = i
        cluster_labels.append(label)

    files = os.listdir(path_dir_text)
    files_xls = [f for f in files if f[-3:] == "txt"]

    files_xls.sort(key=lambda x: int(x.split(".")[0]))

    ground_truth = []
    prdeiction_label = {}
    counter_family = -1
    dic_family = {}
    list_family = []
    for counter, file_name in enumerate(files_xls):
        # print (counter)
        family = file_name.split("_")[-1]
        if family not in dic_family:
            counter_family += 1
            dic_family[family] = counter_family
        list_family.append(dic_family[family])

        if family not in prdeiction_label:
            prdeiction_label[family] = {}
        if cluster_labels[counter] not in prdeiction_label[family]:
            prdeiction_label[family][cluster_labels[counter]] = 0

        prdeiction_label[family][cluster_labels[counter]] += 1

    for counter, file_name in enumerate(files_xls):
        family = file_name.split("_")[-1]

        cluster_id = max(prdeiction_label[family].items(), key=operator.itemgetter(1))[
            0
        ]

        ground_truth.append(cluster_id)

    accuracy_score = accuracy_score(cluster_labels, ground_truth)
    print("cluaster accuracy: ", accuracy_score)
    print(
        "cluster cluster:", metrics.adjusted_rand_score(list_family, clustering.labels_)
    )
    print(
        "cluster adjusted_mutual_info_score:",
        metrics.adjusted_mutual_info_score(list_family, clustering.labels_),
    )


def add_similarity(text_feat, train_id, test_id, original_tetx):
    eps = 0.65
    clustering = hdbscan.HDBSCAN(min_cluster_size=2).fit(original_tetx)

    clustering_2 = DBSCAN(eps=eps, min_samples=2, metric="cosine").fit(original_tetx)

    ids_train = clustering.labels_[train_id]
    ids_test = clustering.labels_[test_id]
    (clusters_train, counts_train) = np.unique(ids_train, return_counts=True)
    (cluster_test, counts_test) = np.unique(ids_test, return_counts=True)
    intersections = np.intersect1d(cluster_test, clusters_train)

    n_clusters = np.unique(clustering.labels_, return_counts=False).shape[0]
    n_clusters_2 = np.unique(clustering_2.labels_, return_counts=False).shape[0]

    family = np.zeros((text_feat.shape[0], n_clusters))
    family_2 = np.zeros((text_feat.shape[0], n_clusters_2))
    for i in range(text_feat.shape[0]):
        np_zero = np.zeros(n_clusters)
        text_cluster = clustering.labels_[i]
        if text_cluster != -1:
            np_zero[text_cluster] = 1
        family[i] = np_zero

        np_zero_2 = np.zeros(n_clusters_2)
        text_cluster_2 = clustering_2.labels_[i]
        if text_cluster_2 != -1:
            np_zero_2[text_cluster_2] = 1
        family_2[i] = np_zero_2

    text_feat = np.concatenate((family_2, family, text_feat), axis=1)
    return text_feat, intersections
