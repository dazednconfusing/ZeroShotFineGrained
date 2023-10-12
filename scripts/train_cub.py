# Change the path to folder
import torch
from dataloaders.CUBLoader import CUBFeatDataSet
from models.options import Options
from models.zest import Attention
from torch.utils.data import DataLoader
from tqdm import tqdm

from scripts.helper import TransformFeatures


def train_zest_style(loss="ace", split="easy", batch_size=100, gpu="0"):
    device = torch.device("cuda:" + gpu if torch.cuda.is_available() else "cpu")

    if loss == "ace":
        dim1 = 4096
        dim2 = 2048

    else:
        dim1 = 2048
        dim2 = 1024

    opt = Options(
        img_dim=3584,
        txt_dim=7551,
        proto_count=30,
        txt_to_img_hidden1=dim1,
        txt_to_img_norm="relu",
        apply_proto_dim_feed_forward=dim2,
        apply_proto_combine="sum",
    )

    tf = TransformFeatures(opt).to(device)
    train_dataset = CUBFeatDataSet(
        folder_path="data/CUB2011", split=split, model="vanilla", train=True
    )

    train_txt_feat_all = train_dataset.train_text_feature
    train_txt_feat_all = torch.from_numpy(train_txt_feat_all).float().unsqueeze(1)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = CUBFeatDataSet(
        folder_path="data/CUB2011", split=split, model="vanilla", train=False
    )

    _, _, _, txt_feat_all = test_dataset.get_all_data()
    text_inp = torch.from_numpy(txt_feat_all).float().unsqueeze(1)
    # test_labels = test_dataset.test_cid
    test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False)

    train_txt_feat_all = train_txt_feat_all.to(device)
    text_inp = text_inp.to(device)

    net = Attention(dimensions=3584, text_dim=7551).to(device)
    # net.apply(init_weights_xavier)

    optimizer = torch.optim.Adam(
        net.parameters(), lr=0.0005, betas=(0.5, 0.9), weight_decay=0.0001
    )

    epochs = 600
    zsl_acc = []
    criterion = torch.nn.CrossEntropyLoss()
    for epoch in range(epochs):
        net.train()
        tf.train()
        correct = 0
        with tqdm(train_loader, unit="batch") as tepoch:
            for img_feat, orig_lab, _, txt_feat in tepoch:
                img_feat = img_feat.to(device)
                orig_lab = orig_lab.to(device)
                img_feat, temp_feat_all = tf(img_feat, train_txt_feat_all)
                attention_weights, attention_scores = net(img_feat, temp_feat_all)
                topv, topi = attention_weights.squeeze().data.topk(1)
                correct += torch.sum(topi.squeeze() == orig_lab).cpu().tolist()
                optimizer.zero_grad()
                loss = criterion(attention_weights.squeeze(), orig_lab.long())
                loss.backward()
                optimizer.step()

        acc = correct * 1.0 / len(train_dataset)
        print("Epoch: " + str(epoch) + ", Accuracy: {}\n".format(acc))

        net.eval()
        tf.eval()
        correct = 0
        with tqdm(test_loader, unit="batch") as tepoch:
            for img_feat, orig_lab, _, txt_feat in tepoch:
                img_feat = img_feat.to(device)
                orig_lab = orig_lab.to(device)
                img_feat, temp_feat_all = tf(img_feat, text_inp)
                attention_weights, attention_scores = net(img_feat, temp_feat_all)
                topv, topi = attention_weights.squeeze().data.topk(1)
                correct += torch.sum(topi.squeeze() == orig_lab).cpu().tolist()

        acc = correct * 1.0 / len(test_dataset)
        print("Test Accuracy: {}\n".format(acc))
        zsl_acc.append(acc)
