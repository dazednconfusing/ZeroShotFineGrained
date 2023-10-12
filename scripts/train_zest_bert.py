# Next: last two, pretrain, lstm


import copy
import random
from functools import partial

import numpy as np
import scipy.integrate as integrate
import torch
import torch.optim as optim
from datasets.cub_dataset_align import DatasetCUB, get_img_txt
from models.vzsl import VZSL
from models.zest import Attention
from models.zsl import ZSL, transformer_map
from preprocessing.embedder import Embedder
from torch import distributed as dist
from torch.distributed.optim import ZeroRedundancyOptimizer
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils import data
from torch.utils.data import DataLoader, RandomSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers.utils.dummy_pt_objects import AutoModel
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

    if opt.gpu == 0 or not dist.is_initialized():
        print(
            "\n\ntrain sizes: ",
            img_train.size(),
            txt_train.size(),
            labels_train.size(),
        )
        print("test sizes: ", img_test.size(), txt_test.size(), labels_test.size())
    if not opt.raw_img_feat:
        training_set = TensorDataset(img_train, labels_train)
        test_set = TensorDataset(img_test, labels_test)

    else:
        del img_train
        del img_test
        del labels_train
        del labels_test
        training_set = DatasetCUB(opt, train=True)

        if opt.tfidf:
            tf_train = torch.tensor(training_set.text_feature).to(opt.device)
            tf_test = torch.tensor(training_set.text_feature_test).to(opt.device)
            opt.txt_dim = tf_train.size(-1)
            if opt.gpu == 0 or not dist.is_initialized():
                print("\n\ntf sizes: ", tf_train.size(), tf_test.size())

        else:
            tf_train = None
            tf_test = None
        test_set = DatasetCUB(opt, train=False)
        if opt.gpu == 0 or not dist.is_initialized():
            print("\n\n raw images used")

    if opt.distributed:
        train_sampler = DistributedSampler(training_set, seed=dist.get_rank() * 15)
        test_sampler = DistributedSampler(
            test_set, seed=dist.get_rank() * 20
        )  # DistributedEvalSampler(test_set)
    else:
        train_sampler = RandomSampler(training_set)
        test_sampler = RandomSampler(test_set)
    # torch.backends.cudnn.benchmark = True

    training_generator = data.DataLoader(
        training_set, sampler=train_sampler, batch_size=opt.batch_size, pin_memory=True
    )

    test_generator = data.DataLoader(
        test_set, sampler=test_sampler, batch_size=opt.test_batch_size, pin_memory=True
    )

    txt_test = txt_test.detach()
    # txt_train = txt_train.detach().cpu()
    txt_sz = txt_train.size(0)
    txt_train = txt_train.to(opt.device)
    txt_test = txt_test.to(opt.device)

    if opt.zero_bert:
        txt_train = torch.zeros((tf_train.size(0), 2, 768))
        txt_test = torch.zeros((tf_test.size(0), 2, 768))
    elif opt.no_bert:
        txt_train = None
        txt_test = None

    if not opt.no_bert and not opt.zero_bert:
        opt, embedder, pooler = pretrain_text_embedder(
            opt, torch.cat([txt_train, txt_test], dim=0)
        )
    else:
        embedder = None
        pooler = None
    if dist.is_initialized():
        dist.barrier()

    zsl = ZSL(opt, embedder, pooler).to(opt.device)
    if opt.load_path:
        zsl.load_state_dict(torch.load(opt.load_path, map_location=opt.gpu))
        if opt.txt_feat == "distilbert":
            nfrozen = 6 - opt.nunfrozen
        else:
            nfrozen = 12 - opt.nunfrozen

        for name, param in zsl.embedder.named_parameters():
            if "layer" in name:
                s = name.split("layer.")[1]
                layer_num = s.split(".")[0]
                if int(layer_num) < nfrozen:
                    param.requires_grad = False
                elif opt.gpu == 0 or not dist.is_initialized():
                    print(name)
            elif "embeddings" in name:
                if opt.nunfrozen > 5:
                    if opt.gpu == 0 or not dist.is_initialized():
                        print(name)
                else:
                    param.requires_grad = False
        for l in zsl.embedder.transformer.layer:
            l.ffn.dropout = torch.nn.Dropout(opt.bert_dropout)

            if opt.gpu == 0 or not dist.is_initialized():
                print("\n\nLoaded:", zsl.embedder)

    params = [
        # {"params": zsl.attention.parameters(), "lr": opt.lr},
    ]

    base_lr = opt.lr

    if hasattr(zsl, "embedder"):
        layer_size = len(
            zsl.embedder.transformer.layer
            if opt.txt_feat == "distilbert"
            else zsl.embedder.encoder.layer
        )
        for i in range(
            layer_size - 1,
            layer_size - 1 - opt.nunfrozen,
            -1,
        ):
            params.append(
                {
                    "params": zsl.embedder.transformer.layer[i].parameters()
                    if opt.txt_feat == "distilbert"
                    else zsl.embedder.encoder.layer[i].parameters(),
                    "lr": base_lr,
                }
            )
            if not dist.is_initialized() or opt.gpu == 0:
                print("lr:", base_lr, "for layer: ", i)
            base_lr *= opt.lr_discount

        if opt.nunfrozen > 5:
            params.append(
                {
                    "params": zsl.embedder.embeddings.parameters(),
                    "lr": base_lr,
                }
            )

            if not dist.is_initialized() or opt.gpu == 0:

                print("lr:", base_lr, "for embedding layer: ")
        for n, p in zsl.embedder.named_parameters():
            if p.requires_grad and opt.gpu == 0:
                print(n, " requires grad")
        if hasattr(zsl, "pooler"):
            params.append(
                {
                    "params": zsl.pooler.parameters(),
                    "lr": opt.pooler_lr,
                }
            )

    if opt.distributed:
        find_unused = True  # opt.nunfrozen > 0 and opt.nunfrozen < 6
        zsl = DDP(
            zsl,
            device_ids=[opt.gpu],
            output_device=opt.gpu,
            find_unused_parameters=find_unused,
        )

        after_bert_params = list(zsl.module.attention.parameters())
        # if hasattr(zsl.module, "pooler"):
        #     after_bert_params += list(zsl.module.pooler.parameters())

        optimizer = ZeroRedundancyOptimizer(
            after_bert_params,
            optimizer_class=partial(torch.optim.Adam, betas=(0.5, 0.9)),
            lr=opt.after_bert_lr,
        )

        for p in params:
            optimizer.add_param_group(p)
    else:
        optimizer = optim.Adam(
            zsl.parameters(),
            lr=opt.after_bert_lr,
            betas=(0.5, 0.9),
            weight_decay=opt.weight_decay,
        )
    criterion = torch.nn.CrossEntropyLoss()
    best = 0

    lr_updates = 0
    for it in range(opt.max_epoch):
        if dist.is_initialized():
            train_sampler.set_epoch(it)
        if (
            lr_updates < len(opt.lr_update_epochs)
            and (it + 1) == opt.lr_update_epochs[lr_updates]
            and not opt.const_lr
        ):
            # opt.lr /= 5
            lr_updates += 1
            for g in optimizer.param_groups:
                g["lr"] = g["lr"] / 5
                lr = g["lr"]

            if opt.gpu == 0 or not dist.is_initialized():

                print("\n\n New LR: ", lr)
        # elif it + 1 > 100 and (it + 1) % 100 == 0 and not opt.const_lr:
        #     opt.lr /= 5
        #     for g in optimizer.param_groups:
        #         g["lr"] = g["lr"] / 5

        #     if opt.gpu == 0 or not dist.is_initialized():

        #         print("\n\n New LR: ", opt.lr)
        if opt.gpu == 0 or not dist.is_initialized():
            print("epoch: ", it)

        train_correct = 0
        total = 0
        pbar = tqdm(training_generator, unit="batch")
        nsteps = 0
        train_loss = 0
        for batch in pbar:
            if not opt.regularizer_off_epoch or opt.regularizer_off_epoch > it:
                zsl.train()
            else:
                zsl.eval()
                if opt.gpu == 0 or not dist.is_initialized():
                    print("\n\n***** EVAL OFF Aat EPOCH: ", it)
            img_ft, lbls = batch
            img_ft = img_ft.to(opt.device)  # half
            lbls = lbls.to(opt.device)
            bs = int(img_ft.size(0))

            # labels, u_idxs = unique(labels)
            # bs = u_idxs.size(0)
            # print("img_ft size: ", img_ft.size(), " txt_ft.size: ", txt_ft.size())

            # if opt.ncandidates < 200:
            #     new_lbls = torch.zeros_like(lbls)
            #     ncandidates = opt.ncandidates
            #     txt_idxs = np.zeros(ncandidates)
            #     choices = np.arange(0, txt_sz)
            #     probs = np.ones(txt_sz, dtype=float)

            #     lbl_map = {}
            #     # do_print = False
            #     for i, lbl in enumerate(lbls):
            #         if lbl_map.get(int(lbl)) is None:
            #             txt_idxs[i] = lbl
            #             lbl_map[int(lbl)] = i
            #             new_lbls[i] = i
            #             probs[lbl] = 0
            #         else:
            #             # do_print = True
            #             new_lbls[i] = lbl_map[int(lbl)]

            #     del lbls
            #     lbls = new_lbls
            #     del new_lbls

            #     # if opt.gpu == 0 and do_print:
            #     #     print("\n\nlbl:", lbls)
            #     #     print("\n\nnew lbls:", new_lbls)
            #     probs = probs / sum(probs)

            #     txt_idxs[len(lbl_map.keys()) :] = np.random.choice(
            #         choices, size=ncandidates - len(lbl_map.keys()), p=probs
            #     )
            # else:
            #     txt_idxs = np.arange(txt_sz)

            # Shuffle text idxs and new labels together
            # rand_idxs = torch.randperm(ncandidates)
            # text_idxs = text_idxs[rand_idxs]
            # new_lbls = new_lbls[rand_idxs]

            # if opt.gpu == 0 or not dist.is_initialized():
            #     print("idxs: ", idxs)
            #     print(
            #         "txt train size, first ft",
            #         txt_train.size(),
            #         txt_train[idxs[:2]][0],
            #         txt_train[labels[0]],
            #     )
            # idxs = torch.tensor(idxs).long().to(opt.device)

            optimizer.zero_grad()
            out = zsl(img_ft, txt_train, tf_train)
            # print("out size: ", out.size())
            loss = criterion(out, lbls)
            loss.backward()
            optimizer.step()
            train_loss += loss.detach().cpu().item()
            nsteps += 1
            preds = out.argmax(dim=-1).detach().cpu()
            correct = float(torch.sum(preds == lbls.detach().cpu()).item())
            train_correct += correct
            total += bs
            pbar.set_postfix(
                {
                    "Tr Acc": f"{(100* correct / bs):.2f}%",
                    "Tr Loss": f"{loss.item():.4f}",
                }
            )

        train_accuracy = torch.tensor(train_correct / total).to(opt.gpu).detach()
        train_loss = torch.tensor(train_loss / nsteps).to(opt.gpu).detach()
        if dist.is_initialized():
            dist.all_reduce(train_accuracy)
            dist.all_reduce(train_loss)
            train_accuracy /= dist.get_world_size()
            train_loss /= dist.get_world_size()
        if opt.gpu == 0 or not dist.is_initialized():
            print(f"train accuracy: {100 * train_accuracy.detach().cpu().item():.4f} %")
            print(f"train loss: {train_loss.detach().cpu().item():.4f} ")

        if dist.is_initialized():
            dist.barrier()
        if opt.gpu == 0 or not dist.is_initialized():
            print("eval time")
        zsl.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            pbar_val = tqdm(test_generator, unit="batch")
            for j, batch in enumerate(pbar_val):
                if dist.is_initialized():
                    test_sampler.set_epoch(j)

                img_ft, lbls = [b.to(opt.device) for b in batch]
                # img_ft = img_ft.half()
                out = zsl(img_ft, txt_test, tf_test)
                preds = out.argmax(dim=-1).detach()
                val_correct += float(torch.sum(preds == lbls).item())
                val_total += lbls.size(0)
                pbar_val.set_postfix(
                    {"Test Acc": f"{(100* val_correct / val_total):.2f}%"}
                )

            test_accuracy = torch.tensor(val_correct / val_total).to(opt.gpu).detach()
            if dist.is_initialized():
                dist.all_reduce(test_accuracy)
                test_accuracy /= dist.get_world_size()
            test_accuracy = float(test_accuracy.detach().cpu().item())
            if opt.gpu == 0 or not dist.is_initialized():
                print(f"test accuracy: {100 * test_accuracy:.4f} %")
                with open("zsl.txt", "a") as f:
                    f.write(f"test accuracy: {100 * test_accuracy:.4f} %")

                if test_accuracy > best:
                    best = test_accuracy

                    if not dist.is_initialized():
                        torch.save(zsl.state_dict(), "vzsl.model")
                    else:
                        torch.save(zsl.module.state_dict(), "vzsl.model")

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


# def train(opt):
#     params = {"batch_size": opt.batch_size, "shuffle": True}

#     # torch.backends.cudnn.benchmark = True
#     training_set = DatasetCUB(opt)
#     training_generator = data.DataLoader(training_set, **params)

#     test_set = DatasetCUB(opt, train=False)
#     test_generator = data.DataLoader(test_set, **params)
#     # print(len(training_set), len(test_set))
#     if opt.embed:
#         text_feat = training_set.text_embed
#         text_feat_test = training_set.text_embed_test

#         if opt.pretrain_task == "both":
#             opt.pretrain_task = "mlm"
#             opt.unfrozen_layers = 12
#             opt = pretrain_text_embedder(opt, text_feat)
#             opt.pretrain_task = "cls"
#             opt.unfrozen_layers = 6

#         opt = pretrain_text_embedder(opt, text_feat)
#         training_set = DatasetCUB(opt)
#         training_generator = data.DataLoader(training_set, **params)

#         test_set = DatasetCUB(opt, train=False)
#         test_generator = data.DataLoader(test_set, **params)

#         text_feat = training_set.text_embed
#         text_feat_test = training_set.text_embed_test

#     else:
#         text_feat = torch.tensor(training_set.text_feature).to(opt.device)

#         text_feat_test = torch.tensor(training_set.text_feature_test).to(opt.device)

#     text_len = training_set.text_len
#     text_len_test = training_set.text_len_test

#     opt.feature_dim = training_set.feature_dim
#     opt.text_dim = training_set.text_dim

#     # zsl = torch.nn.DataParallel(ZSL(opt).to(opt.device))
#     zsl = ZSL(opt).to(opt.device)

#     text_dim = opt.text_dim
#     if opt.sentence_pooling == "lstm":
#         text_dim = 2 * opt.lstm_hidden_dim
#     if opt.linear > 0:
#         text_dim = opt.text_output_dim

#     net = Attention(dimensions=3584, text_dim=text_dim).to(opt.device)

#     params = list(net.parameters())
#     if not opt.freeze_text_processing:
#         params += list(zsl.parameters())

#     optimizer = optim.Adam(
#         params, lr=opt.lr, betas=(0.5, 0.9), weight_decay=opt.weight_decay
#     )
#     criterion = torch.nn.CrossEntropyLoss()

#     for it in range(opt.max_epoch):
#         if (it + 1) % 100 == 0 and not opt.const_lr:
#             opt.lr /= 10
#             for g in optimizer.param_groups:
#                 g["lr"] = opt.lr

#             print("\n\n New LR: ", opt.lr)
#         print("epoch: ", it)

#         zsl.train()
#         train_correct = 0
#         total = 0
#         for batch in tqdm(training_generator, unit="batch"):

#             img_ft, labels = [b.to(opt.device) for b in batch]
#             # image_feat, y_true = images.to(opt.device), labels.to(opt.device)

#             text_feat_out = zsl(text_feat, text_len)
#             out = net(img_ft, text_feat_out.unsqueeze(0))
#             optimizer.zero_grad()
#             loss = criterion(out, labels.long())
#             loss.backward()
#             optimizer.step()
#             # print("shapes: ", text_feat_out.shape, text_dim)
#             # attention_weights, attention_scores = net(
#             #     image_feat, text_feat_out.unsqueeze(0)
#             # )
#             preds = out.argmax()
#             correct = torch.sum(preds == labels).item()
#             train_correct += correct
#             total += labels.size(0)

#             # _, topi = attention_scores.squeeze().data.topk(1)
#             # compare_pred_ground = (topi.squeeze() == y_true).to(opt.device)
#             # total += len(y_true)
#             # correct = np.count_nonzero(compare_pred_ground.cpu() == 1)
#             # train_correct += correct

#             # optimizer.zero_grad()
#             # loss = criterion(attention_weights.squeeze(), y_true.long())
#             # loss.backward()
#             # optimizer.step()

#         print("train accuracy:", 100 * train_correct / total)
#         zsl.eval()
#         correct = 0

#         with torch.no_grad():

#             for batch in tqdm(test_generator, unit="batch"):

#                 images, labels = batch
#                 image_feat, y_true = images.to(opt.device), labels.to(opt.device)
#                 text_feat_out = zsl(text_feat_test, text_len_test)
#                 attention_weights, attention_scores = net(
#                     image_feat, text_feat_out.unsqueeze(0)
#                 )
#                 topv, topi = attention_weights.squeeze().data.topk(1)
#                 correct += torch.sum(topi.squeeze() == y_true).cpu().tolist()

#             # print(test_set.pfc_feat_data_test.shape)
#             print(
#                 "test accuracy:", 100 * correct / test_set.pfc_feat_data_test.shape[0]
#             )

#         evaluate_gzsl(
#             text_feat,
#             text_len,
#             text_feat_test,
#             text_len_test,
#             training_set.train_cls_num,
#             training_generator,
#             test_generator,
#             zsl,
#             net,
#             opt.device,
#         )


# def evaluate_gzsl(
#     text_feat_train,
#     text_len,
#     text_feat_test,
#     text_len_test,
#     train_cls_num,
#     training_generator,
#     test_generator,
#     zsl,
#     net,
#     device,
# ):
#     if zsl.embed:
#         n_classes = text_feat_train.shape[0] + text_feat_test.shape[0]
#         max_sentences = max(text_feat_train.shape[1], text_feat_test.shape[1])
#         text_dim = text_feat_train.shape[2]
#         text_feat = torch.zeros((n_classes, max_sentences, text_dim)).to(device)
#         text_feat[
#             : text_feat_train.shape[0], : text_feat_train.shape[1], :
#         ] = text_feat_train
#         text_feat[
#             text_feat_train.shape[0] :, : text_feat_test.shape[1], :
#         ] = text_feat_test

#     else:
#         text_feat = torch.cat((text_feat_train, text_feat_test), dim=0)
#     unseen_sim = np.zeros([0, text_feat.shape[0]])

#     text_len = torch.cat((text_len, text_len_test), dim=0)
#     seen_sim = np.zeros_like(unseen_sim)
#     labels_unseen = []
#     for _, batch in enumerate(test_generator):
#         images, labels = batch
#         labels += train_cls_num
#         labels_unseen += labels.tolist()
#         image_feat = images.to(device)

#         text_feat_out = zsl(text_feat, text_len)
#         attention_weights, _ = net(image_feat, text_feat_out.unsqueeze(0))
#         unseen_sim = np.vstack(
#             (unseen_sim, attention_weights.squeeze().data.cpu().numpy())
#         )

#     labels_seen = []
#     for _, batch in enumerate(training_generator):
#         images, labels = batch
#         image_feat = images.to(device)
#         labels_seen += labels.tolist()

#         text_feat_out = zsl(text_feat, text_len)
#         attention_weights, _ = net(image_feat, text_feat_out.unsqueeze(0))

#         # sm = torch.nn.Softmax(dim=-1)
#         # attention_weights = sm(attention_weights)
#         seen_sim = np.vstack((seen_sim, attention_weights.squeeze().data.cpu().numpy()))

#     acc_s_t_list, acc_u_t_list = list(), list()

#     for gzsl_lambda in np.arange(-2, 2, 0.01):
#         tmp_seen_sim = copy.deepcopy(seen_sim)
#         tmp_seen_sim[:, train_cls_num:] += gzsl_lambda
#         pred_lbl = np.argmax(tmp_seen_sim, axis=1)
#         acc_s_t_list.append((pred_lbl == np.asarray(labels_seen)).mean())

#         tmp_unseen_sim = copy.deepcopy(unseen_sim)
#         tmp_unseen_sim[:, train_cls_num:] += gzsl_lambda
#         pred_lbl = np.argmax(tmp_unseen_sim, axis=1)
#         acc_u_t_list.append((pred_lbl == (np.asarray(labels_unseen))).mean())

#     auc_score = integrate.trapz(y=acc_s_t_list, x=acc_u_t_list)

#     print("AUC Score is {:.4}".format(auc_score))
