import copy
import os
import random
from functools import partial

import torch
from models.lstm import LSTM
from models.zsl import PointwiseFFN, transformer_map
from torch import nn, optim
from torch.distributed.optim import ZeroRedundancyOptimizer
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


class FineTunerMLM(nn.Module):
    def __init__(self, opt, vocab_size, embedder, embed_dim=768):
        super(FineTunerMLM, self).__init__()
        self.embedder = embedder
        self.embed_dim = embed_dim
        self.linear = nn.Linear(embed_dim, vocab_size)

    def forward(self, x, pos):
        x = self.embedder(x).last_hidden_state
        x = torch.masked_select(x, pos.unsqueeze(-1).expand(-1, -1, self.embed_dim))
        return self.linear(x.view(-1, self.embed_dim))


class FineTunerCLS(nn.Module):
    def __init__(self, opt, classes, embedder, embed_dim=768):
        super(FineTunerCLS, self).__init__()
        self.embedder = embedder
        self.pooling = opt.pooling
        self.last_two = opt.last_two and not opt.last_two_mean
        self.last_two_mean = opt.last_two_mean

        text_dim = embed_dim
        if self.last_two:
            text_dim *= 2

        self.pooler = None
        nclusters = 1
        if self.pooling == "lstm":
            self.pooler = LSTM(
                opt,
                embedding_dim=text_dim,
            )
            text_dim = self.pooler.num_directions * self.pooler.hidden_dim
        elif self.pooling == "local_cluster":
            self.pooler = PointwiseFFN(opt, text_dim)
            nclusters = opt.nclusters

        self.embed_dim = text_dim
        self.linear = nn.Linear(self.embed_dim * nclusters, classes)

    def forward(self, x):
        x = self.embedder(x, output_hidden_states=True)
        if self.last_two:
            x = torch.cat([x.hidden_states[-1], x.hidden_states[-2]], dim=-1)
        else:
            x = x.last_hidden_state
        if self.pooler:
            x = self.pooler(x)
        else:
            x = x.mean(dim=1)
        return self.linear(x)


def pretrain_text_embedder(opt, txt_tokens):
    if not opt.pretrain:
        return opt, None, None

    if opt.pretrained_path:
        embedder = torch.load(opt.pretrained_path).to(opt.device)
    else:
        embedder = AutoModel.from_pretrained(transformer_map[opt.txt_feat]).to(
            opt.device
        )

    n_classes = txt_tokens.size(0)

    pooler = None
    if opt.pretrain_task in ("mlm", "both"):
        train_loader, vocab_size = get_mlm_dataloader(opt, txt_tokens)
        opt, embedder, pooler = train_mlm(
            opt, embedder, train_loader, vocab_size=vocab_size
        )

    if opt.pretrain_task in ("cls", "both"):
        training_set = TensorDataset(txt_tokens, torch.arange(n_classes).long())
        train_loader = DataLoader(
            training_set, shuffle=True, batch_size=opt.pretrain_bs
        )
        opt, embedder, pooler = train_cls(
            opt, embedder, train_loader, classes=n_classes
        )

    return opt, embedder, pooler


def train_mlm(*args, **kwargs):
    return _train(*args, task="mlm", **kwargs)


def train_cls(*args, **kwargs):
    return _train(*args, task="cls", **kwargs)


def _train(opt, embedder, train_loader, vocab_size=None, classes=None, task="mlm"):
    criterion = torch.nn.CrossEntropyLoss()
    lr = opt.pretrain_mlm_lr if task == "mlm" else opt.pretrain_cls_lr
    nunfrozen = (
        opt.pretrain_mlm_nunfrozen if task == "mlm" else opt.pretrain_cls_nunfrozen
    )
    epochs = opt.pretrain_mlm_epochs if task == "mlm" else opt.pretrain_cls_epochs
    params = []

    layer_size = len(
        embedder.transformer.layer
        if opt.txt_feat == "distilbert"
        else embedder.encoder.layer
    )
    if opt.txt_feat == "distilbert":
        nfrozen = 6 - nunfrozen
    else:
        nfrozen = 12 - nunfrozen
    for i in range(
        layer_size - 1,
        nfrozen - 1,
        -1,
    ):
        params.append(
            {
                "params": embedder.transformer.layer[i].parameters()
                if opt.txt_feat == "distilbert"
                else embedder.encoder.layer[i].parameters(),
                "lr": lr,
                "start_lr": lr,
            }
        )
        lr *= 0.9

    if task == "mlm":
        model = FineTunerMLM(opt, vocab_size=vocab_size, embedder=embedder).to(
            opt.device
        )
    else:
        model = FineTunerCLS(opt, classes, embedder=embedder).to(opt.device)
        if model.pooler is not None:
            params.append(
                {
                    "params": model.pooler.parameters(),
                    "lr": opt.pretrain_cls_lr,
                    "start_lr": opt.pretrain_cls_lr,
                }
            )

    if opt.distributed:
        model = DDP(
            model,
            device_ids=[opt.gpu],
            output_device=opt.gpu,
            find_unused_parameters=True,
        )
        optimizer = ZeroRedundancyOptimizer(
            list(model.module.linear.parameters()),
            optimizer_class=torch.optim.Adam,  # partial(torch.optim.Adam, betas=(0.5, 0.9)),
            lr=lr / 10,
        )
        for p in params:
            optimizer.add_param_group(p)
    else:
        optimizer = optim.Adam(
            params,
            lr=lr / 50,
            betas=(0.5, 0.9),
            weight_decay=opt.weight_decay,
        )
        for p in params:
            optimizer.add_param_group(p)
    steps = 0
    warmup = 0.5
    max_scale = 5
    min_scale = 1e-9
    # cooldown_scalar = 2
    total_steps = len(train_loader) * epochs
    for i in range(epochs):
        for g in optimizer.param_groups:
            if opt.gpu == 0 or not opt.distributed:
                print("new lr: ", g["lr"], "steps: ", steps, "/", total_steps)
        correct = 0
        total = 0
        pbar = tqdm(train_loader)
        for bi, batch in enumerate(pbar):
            steps += 1
            if (steps + 1) / total_steps < warmup:
                for g in optimizer.param_groups:
                    if g.get("start_lr"):
                        g["lr"] = g["lr"] + (
                            max_scale * g["start_lr"] - g["start_lr"]
                        ) / (warmup * total_steps)
            else:
                for g in optimizer.param_groups:
                    if g.get("start_lr"):
                        g["lr"] = g["lr"] - (
                            max_scale * g["start_lr"] - min_scale * g["start_lr"]
                        ) / (total_steps * (1 - warmup))
                        if g["lr"] < min_scale * g["start_lr"]:
                            g["lr"] = min_scale * g["start_lr"]

            model.train()
            optimizer.zero_grad()
            if task == "cls":
                x, y = batch
                x = x.to(opt.device)
                y = y.to(opt.device)

                out = model(x)
                total += y.size(0)
                batch_total = y.size(0)
            else:
                x, y, mask = batch
                x = x.to(opt.device)
                y = y.to(opt.device)
                mask = mask.to(opt.device)
                y = torch.masked_select(y, mask).reshape((-1,))
                batch_total = int(torch.sum(mask.float()).item())
                total += batch_total
                out = model(x, mask)

            # print(out.size(), y.size())

            loss = criterion(out, y)
            # loss = criterion2(pred, cls_)
            loss.backward()
            optimizer.step()
            # model.eval()

            batch_correct = int(torch.sum(out.argmax(dim=-1) == y).int().item())
            correct += batch_correct
            pbar.set_postfix(
                {
                    task
                    + " accuracy": str(round(100 * batch_correct / batch_total, 3))
                    + "%",
                    "loss": round(loss.item(), 3),
                }
            )
        if opt.gpu == 0 or not opt.distributed:
            print(
                "epoch: ",
                i + 1,
                "," + task + " accuracy:",
                round(100 * correct / total, 3),
                "%",
            )

    pooler = None
    if opt.distributed:
        embedder = model.module.embedder

        if task == "cls":
            pooler = model.module.pooler
    else:
        embedder = model.embedder

        if task == "cls":
            pooler = model.pooler

    for _, param in embedder.named_parameters():
        param.requires_grad = False

    # opt.unfrozen_layers = 0
    # torch.save(embedder.cpu().state_dict(), path)
    # opt.pretrained_path = path

    del model
    del train_loader

    return opt, embedder, pooler


def get_mlm_dataloader(opt, txt_tokens):
    tokenizer = AutoTokenizer.from_pretrained(transformer_map[opt.txt_feat])
    mask_id = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)
    cls_id = tokenizer.convert_tokens_to_ids(tokenizer.cls_token)
    sep_id = tokenizer.convert_tokens_to_ids(tokenizer.sep_token)
    pad_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)

    sentences, tokens = txt_tokens.size()
    positions_mask = torch.zeros_like(txt_tokens)
    true_values = torch.zeros_like(txt_tokens)
    txt_tokens_copy = copy.deepcopy(txt_tokens)
    vocab = {}
    idx = 0
    for i in range(sentences):
        for j in range(tokens):
            if vocab.get(txt_tokens[i, j].item()) is None and txt_tokens[
                i, j
            ].item() not in (
                mask_id,
                cls_id,
                sep_id,
                pad_id,
            ):
                vocab[txt_tokens[i, j].item()] = idx
                idx += 1
            if random.random() < 0.15 and txt_tokens[i, j].item() not in (
                mask_id,
                cls_id,
                sep_id,
                pad_id,
            ):
                r = random.random()
                if r < 0.8:
                    masked_val = mask_id
                elif r < 0.9:
                    rand_s = int(round(random.random() * (sentences - 1)))
                    rand_j = int(round(random.random() * (tokens - 1)))
                    masked_val = txt_tokens_copy[rand_s, rand_j]
                else:
                    masked_val = txt_tokens[i, j]

                positions_mask[i, j] = 1
                true_values[i, j] = vocab[txt_tokens[i, j].item()]
                txt_tokens[i, j] = masked_val
    del txt_tokens_copy
    training_set = TensorDataset(
        txt_tokens,
        true_values.long(),
        positions_mask.bool(),
    )
    return DataLoader(training_set, shuffle=True, batch_size=opt.pretrain_bs), len(
        vocab
    )
