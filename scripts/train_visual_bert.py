import copy

import numpy as np
import scipy.integrate as integrate
import torch
import torch.optim as optim
from datasets.cub_dataset_align import DatasetCUB, get_img_txt
from models.vzsl import VZSL
from models.zest import Attention
from torch import distributed as dist
from torch.distributed.optim import ZeroRedundancyOptimizer
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils import data
from torch.utils.data import DataLoader, RandomSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from utils.distributed_sampler import DistributedEvalSampler

from scripts.pretrain import pretrain_text_embedder


def unique(x, dim=None):
    """Unique elements of x and indices of those unique elements
    https://github.com/pytorch/pytorch/issues/36748#issuecomment-619514810

    e.g.

    unique(tensor([
        [1, 2, 3],
        [1, 2, 4],
        [1, 2, 3],
        [1, 2, 5]
    ]), dim=0)
    => (tensor([[1, 2, 3],
                [1, 2, 4],
                [1, 2, 5]]),
        tensor([0, 1, 3]))
    """
    unique, inverse = torch.unique(x, sorted=True, return_inverse=True, dim=dim)
    perm = torch.arange(inverse.size(0), dtype=inverse.dtype, device=inverse.device)
    inverse, perm = inverse.flip([0]), perm.flip([0])
    return unique, inverse.new_empty(unique.size(0)).scatter_(0, inverse, perm)


def train(opt):
    params = {"batch_size": opt.batch_size}
    if dist.is_initialized():
        if opt.gpu == 0:
            img_train, txt_train, labels_train = get_img_txt(opt, train=True)
            img_test, txt_test, labels_test = get_img_txt(opt, train=False)

        dist.barrier()
        if opt.gpu != 0:
            img_train, txt_train, labels_train = get_img_txt(opt, train=True)
            img_test, txt_test, labels_test = get_img_txt(opt, train=False)
    else:
        img_train, txt_train, labels_train = get_img_txt(opt, train=True)
        img_test, txt_test, labels_test = get_img_txt(opt, train=False)

    # if opt.gpu == 0 or not dist.is_initialized():
    #     print("train sizes: ", img_train.size(), txt_train.size(), labels_train.size())

    training_set = TensorDataset(img_train, txt_train, labels_train)

    # if opt.gpu == 0 or not dist.is_initialized():
    #     print("test sizes: ", img_test.size(), txt_test.size(), labels_test.size())
    test_set = TensorDataset(img_test, labels_test)

    if opt.distributed:
        train_sampler = DistributedSampler(training_set)
        test_sampler = DistributedSampler(test_set)  # DistributedEvalSampler(test_set)
    else:
        train_sampler = RandomSampler(training_set)
        test_sampler = RandomSampler(test_set)
    # torch.backends.cudnn.benchmark = True

    training_generator = data.DataLoader(
        training_set,
        sampler=train_sampler,
        batch_size=opt.batch_size,
        pin_memory=True,
        num_workers=opt.num_workers,
    )

    test_generator = data.DataLoader(
        test_set,
        sampler=test_sampler,
        batch_size=opt.test_batch_size,
        pin_memory=True,
        num_workers=opt.num_workers,
    )

    txt_test = txt_test.detach()
    test_sz = txt_test.size(0)

    # if opt.pretrain_task == "both":
    #     opt.pretrain_task = "mlm"
    #     opt.unfrozen_layers = 12
    #     opt = pretrain_text_embedder(opt, text_feat)
    #     opt.pretrain_task = "cls"
    #     opt.unfrozen_layers = 6

    # opt = pretrain_text_embedder(opt, text_feat)
    # training_set = DatasetCUB(opt)
    # training_generator = data.DataLoader(training_set, **params)

    # test_set = DatasetCUB(opt, train=False)
    # test_generator = data.DataLoader(test_set, **params)

    # text_len = training_set.text_len
    # text_len_test = training_set.text_len_test

    # opt.feature_dim = training_set.feature_dim
    # opt.text_dim = training_set.text_dim

    # zsl = torch.nn.DataParallel(ZSL(opt).to(opt.device))
    vzsl = VZSL(opt).to(opt.gpu)

    params = []

    if opt.pooling == "ff":
        params.append({"params": vzsl.pooler.parameters(), "lr": opt.lr})
    if opt.img_encoding in ("transformer", "ff"):
        params.append({"params": vzsl.img_encoder.parameters(), "lr": opt.lr})

    base_lr = opt.lr
    for i in range(
        len(vzsl.embedder.transformer.layer) - 1,
        len(vzsl.embedder.transformer.layer) - 1 - opt.nunfrozen,
        -1,
    ):
        if opt.gpu == 0 or not dist.is_initialized():
            print("layer unfrozen: ", i)
        params.append(
            {
                "params": vzsl.embedder.transformer.layer[i].parameters(),
                "lr": base_lr,
            }
        )
        base_lr *= opt.lr_discount
    if opt.distributed:
        vzsl = DDP(
            vzsl,
            device_ids=[opt.gpu],
            output_device=opt.gpu,
            find_unused_parameters=True,
        )

        optimizer = ZeroRedundancyOptimizer(
            params,
            optim=torch.optim.Adam,
            lr=opt.lr,
        )
    else:
        optimizer = optim.Adam(
            params,
            lr=opt.lr,
            betas=(0.5, 0.9),
            weight_decay=opt.weight_decay,
        )
    criterion = torch.nn.CrossEntropyLoss()

    best = 0
    for it in range(opt.max_epoch):
        if opt.gpu == 0 or not dist.is_initialized():
            print("epoch: ", it)

        train_correct = 0
        total = 0
        pbar = tqdm(training_generator, unit="batch")
        nsteps = 0
        losses = []
        for batch in pbar:
            vzsl.train()
            img_ft, txt_ft, labels = [b.to(opt.device) for b in batch]
            # image_feat, y_true = images.to(opt.device), labels.to(opt.device)
            img_ft = img_ft.float()

            _, idxs = unique(labels)
            bs = idxs.size(0)
            labels = torch.arange(0, bs).long().to(opt.device)

            img_ft = (
                img_ft[idxs, :, :]
                .unsqueeze(1)
                .expand([bs, bs, 7, 768])
                .reshape((bs * bs, 7, 768))
            )

            txt_ft = (
                txt_ft[idxs, :].unsqueeze(0).expand([bs, bs, -1]).reshape((bs * bs, -1))
            )
            # print("img_ft size: ", img_ft.size(), " txt_ft.size: ", txt_ft.size())
            optimizer.zero_grad()
            out = vzsl(img_ft, txt_ft)
            # print("out size: ", out.size())
            out = out.reshape((bs, bs))
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            losses.append(loss.detach().cpu().item())
            nsteps += 1
            preds = out.argmax(dim=-1)
            correct = float(torch.sum(preds == labels).item())
            train_correct += correct
            total += bs
            pbar.set_postfix(
                {
                    "Tr Acc": f"{(100* correct / bs):.2f}%",
                    "Tr Loss": f"{loss.item():.4f}",
                }
            )

        train_accuracy = torch.tensor(train_correct / total).to(opt.gpu).detach()
        train_loss = torch.tensor(np.mean(losses)).to(opt.gpu).detach()
        if dist.is_initialized():
            dist.all_reduce(train_accuracy)
            dist.all_reduce(train_loss)
            train_accuracy /= dist.get_world_size()
            train_loss /= dist.get_world_size()
        if opt.gpu == 0 or not dist.is_initialized():
            print(f"train accuracy: {100 * train_accuracy.detach().cpu().item():.4f} %")
            print(f"train loss: {train_loss.detach().cpu().item():.4f} ")

        dist.barrier()
        if opt.gpu == 0 or not dist.is_initialized():
            print("eval time")
        vzsl.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            pbar_val = tqdm(test_generator, unit="batch")
            for batch in pbar_val:

                img_ft, lbls = [b.to(opt.device) for b in batch]
                bs = img_ft.size(0)

                img_ft = img_ft.detach()
                lbls = lbls.detach()
                img_ft = (
                    img_ft.float()
                    .unsqueeze(1)
                    .expand(bs, test_sz, 7, 768)
                    .reshape((test_sz * bs, 7, 768))
                )

                out = vzsl(
                    img_ft,
                    txt_test.unsqueeze(0)
                    .expand(bs, -1, -1)
                    .reshape((bs * test_sz, -1)),
                )
                out = out.reshape((bs, test_sz))
                preds = out.argmax(dim=1).detach()
                val_correct += float(torch.sum(preds == lbls).item())
                val_total += bs
                pbar_val.set_postfix(
                    {"Test Acc": f"{(100* val_correct / val_total):.2f}%"}
                )

            test_accuracy = torch.tensor(val_correct / val_total).to(opt.gpu).detach()
            if dist.is_initialized():
                dist.all_reduce(test_accuracy)
                test_accuracy /= dist.get_world_size()
            test_accuracy = test_accuracy.detach().cpu().item()

            if opt.gpu == 0 or not dist.is_initialized():
                print(f"test accuracy: {100 * test_accuracy:.4f} %")
                with open("vzsl.txt", "a") as f:
                    f.write(f"test accuracy: {100 * test_accuracy:.4f} %")

                if test_accuracy > best:
                    best = test_accuracy

                    if not dist.is_initialized():
                        torch.save(vzsl.state_dict(), "vzsl.model")
                    else:
                        torch.save(vzsl.module.state_dict(), "vzsl.model")

        # evaluate_gzsl(
        #     text_feat,
        #     text_len,
        #     text_feat_test,
        #     text_len_test,
        #     training_set.train_cls_num,
        #     training_generator,
        #     test_generator,
        #     zsl,
        #     net,
        #     opt.device,
        # )


def evaluate_gzsl(
    text_feat_train,
    text_len,
    text_feat_test,
    text_len_test,
    train_cls_num,
    training_generator,
    test_generator,
    zsl,
    net,
    device,
):
    if zsl.embed:
        n_classes = text_feat_train.shape[0] + text_feat_test.shape[0]
        max_sentences = max(text_feat_train.shape[1], text_feat_test.shape[1])
        text_dim = text_feat_train.shape[2]
        text_feat = torch.zeros((n_classes, max_sentences, text_dim)).to(device)
        text_feat[
            : text_feat_train.shape[0], : text_feat_train.shape[1], :
        ] = text_feat_train
        text_feat[
            text_feat_train.shape[0] :, : text_feat_test.shape[1], :
        ] = text_feat_test

    else:
        text_feat = torch.cat((text_feat_train, text_feat_test), dim=0)
    unseen_sim = np.zeros([0, text_feat.shape[0]])

    text_len = torch.cat((text_len, text_len_test), dim=0)
    seen_sim = np.zeros_like(unseen_sim)
    labels_unseen = []
    for _, batch in enumerate(test_generator):
        images, labels = batch
        labels += train_cls_num
        labels_unseen += labels.tolist()
        image_feat = images.to(device)

        text_feat_out = zsl(text_feat, text_len)
        attention_weights, _ = net(image_feat, text_feat_out.unsqueeze(0))
        unseen_sim = np.vstack(
            (unseen_sim, attention_weights.squeeze().data.cpu().numpy())
        )

    labels_seen = []
    for _, batch in enumerate(training_generator):
        images, labels = batch
        image_feat = images.to(device)
        labels_seen += labels.tolist()

        text_feat_out = zsl(text_feat, text_len)
        attention_weights, _ = net(image_feat, text_feat_out.unsqueeze(0))

        # sm = torch.nn.Softmax(dim=-1)
        # attention_weights = sm(attention_weights)
        seen_sim = np.vstack((seen_sim, attention_weights.squeeze().data.cpu().numpy()))

    acc_s_t_list, acc_u_t_list = list(), list()

    for gzsl_lambda in np.arange(-2, 2, 0.01):
        tmp_seen_sim = copy.deepcopy(seen_sim)
        tmp_seen_sim[:, train_cls_num:] += gzsl_lambda
        pred_lbl = np.argmax(tmp_seen_sim, axis=1)
        acc_s_t_list.append((pred_lbl == np.asarray(labels_seen)).mean())

        tmp_unseen_sim = copy.deepcopy(unseen_sim)
        tmp_unseen_sim[:, train_cls_num:] += gzsl_lambda
        pred_lbl = np.argmax(tmp_unseen_sim, axis=1)
        acc_u_t_list.append((pred_lbl == (np.asarray(labels_unseen))).mean())

    auc_score = integrate.trapz(y=acc_s_t_list, x=acc_u_t_list)

    print("AUC Score is {:.4}".format(auc_score))
