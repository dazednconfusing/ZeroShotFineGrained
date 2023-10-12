import os
import pickle
import sys

import hdbscan
import numpy as np
import scipy.io as sio
import torch
from preprocessing.embedder import Embedder
from sklearn import metrics
from sklearn.cluster import DBSCAN
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


class DatasetCUB(Dataset):
    def __init__(self, opt, train=True):
        self.train = train

        txt_feat_path_original = r"data/CUB2011/CUB_Porter_7551D_TFIDF_new_original.mat"
        # txt_raw_path = r"data/Raw_Wiki_Articles/CUBird_WikiArticles"
        txt_raw_path = r"data/Raw_Wiki_Articles/CUBird_Sentences"
        if opt.split == "easy":
            if opt.txt_feat == "similarity_VRS":  # 'similarity_VRS'
                txt_feat_path = r"data/CUB2011/CUB_EASY_SPLIT_VRS.mat"
            else:  # vanilla\similarity
                txt_feat_path = r"data/CUB2011/CUB_Porter_7551D_TFIDF_new_original.mat"

            train_test_split_dir = "data/CUB2011/train_test_split_easy.mat"
            pfc_label_path_train = "data/CUB2011/labels_train.pkl"
            pfc_label_path_test = "data/CUB2011/labels_test.pkl"
            pfc_feat_path_train = "data/CUB2011/pfc_feat_train.mat"
            pfc_feat_path_test = "data/CUB2011/pfc_feat_test.mat"
            train_cls_num = 150
            test_cls_num = 50
        else:
            if opt.txt_feat == "similarity_VRS":  # 'similarity_VRS'
                txt_feat_path = r"data/CUB2011/CUB_HARD_SPLIT_VRS.mat"
            else:  # vanilla\similarity
                txt_feat_path = r"data/CUB2011/CUB_Porter_7551D_TFIDF_new_original.mat"

            train_test_split_dir = "data/CUB2011/train_test_split_hard.mat"
            pfc_label_path_train = "data/CUB2011/labels_train_hard.pkl"
            pfc_label_path_test = "data/CUB2011/labels_test_hard.pkl"
            pfc_feat_path_train = "data/CUB2011/pfc_feat_train_hard.mat"
            pfc_feat_path_test = "data/CUB2011/pfc_feat_test_hard.mat"
            train_cls_num = 160
            test_cls_num = 40

        if opt.img_feat == "vpde":
            self.pfc_feat_data_train = sio.loadmat(pfc_feat_path_train)[
                "pfc_feat"
            ].astype(np.float32)
            self.pfc_feat_data_test = sio.loadmat(pfc_feat_path_test)[
                "pfc_feat"
            ].astype(np.float32)
        else:
            # TODO: Rahul
            self.pfc_feat_data_train, self.pfc_feat_data_test = get_resnet()

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

        if self.train:
            self.text_feature, self.text_feature_test = get_text_feature(
                txt_feat_path, train_test_split_dir, txt_feat_path_original, opt
            )
            self.text_dim = self.text_feature.shape[1]
            (
                self.text_embed,
                self.text_tokens,
                self.text_len,
                self.text_embed_test,
                self.text_tokens_test,
                self.text_len_test,
            ) = get_word_embeddings(train_test_split_dir, txt_raw_path, opt)

            # (
            #     self.text_embed,
            #     self.text_len,
            #     self.text_embed_test,
            #     self.text_len_test,
            # ) = get_sentence_embeddings(train_test_split_dir, txt_raw_path, opt)
            # (
            #     self.text_embed,
            #     self.text_len,
            #     self.text_embed_test,
            #     self.text_len_test,
            # ) = get_text_embeddings(train_test_split_dir, txt_raw_path, opt)
            if opt.embed:
                self.text_dim = 768

    def __len__(self):
        if self.train:
            return self.pfc_feat_data_train.shape[0]
        else:
            return self.pfc_feat_data_test.shape[0]

    def __getitem__(self, index):
        "Generates one sample of data"
        # Select sample
        if self.train:
            img_feat = torch.from_numpy(self.pfc_feat_data_train[index])
            label = torch.tensor(self.labels_train[index], dtype=torch.float64)
            return img_feat, label
        else:
            img_feat = torch.from_numpy(self.pfc_feat_data_test[index])
            label = torch.tensor(self.labels_test[index], dtype=torch.float64)

            return img_feat, label


def get_resnet():
    # TODO: Rahul
    return None, None


def get_word_embeddings(train_test_split_dir, txt_sentences_path, opt):
    if not opt.embed:
        return (
            torch.tensor([]),
            torch.tensor([]),
            torch.tensor([]),
            torch.tensor([]),
            torch.tensor([]),
            torch.tensor([]),
        )

    repr = "word_embed"
    repr_tokens = "word_tokens"

    if opt.pretrained_path is not None:
        repr += "_pretrained"
        repr_tokens += "_pretrained"

    output_dir = "preprocessing/"
    dirname = os.path.join(output_dir, opt.txt_feat)
    os.makedirs(dirname, exist_ok=True)
    device = opt.device
    try:
        embed = torch.load(os.path.join(dirname, repr + ".pt")).to(device)
        embed_test = torch.load(os.path.join(dirname, repr + "_test.pt")).to(device)
        lengths = torch.load(os.path.join(dirname, repr + "_lengths.pt")).to(device)
        lengths_test = torch.load(os.path.join(dirname, repr + "_lengths_test.pt")).to(
            device
        )
        tokens = torch.load(os.path.join(dirname, repr_tokens + ".pt")).to(device)
        tokens_test = torch.load(os.path.join(dirname, repr_tokens + "_test.pt")).to(
            device
        )
        print("\n\n Embeddings Loaded!")
    except Exception:
        print("\n\n Computing Embeddings!")
        train_test_split = sio.loadmat(train_test_split_dir)

        train_cid = train_test_split["train_cid"].squeeze() - 1
        test_cid = train_test_split["test_cid"].squeeze() - 1
        train_cid = torch.tensor(train_cid).long().reshape((-1,))
        test_cid = torch.tensor(test_cid).long().reshape((-1,))

        prototypes = [
            (
                "It has a large grey head.",
                [3, 4, 5],
            ),
            (
                "It is medium - sized and spotted.",
                [2, 4, 6],
            ),
            (
                "It is blue and has a unique pattern.",
                [2, 7],
            ),
            (
                "It is brown.",
                [2],
            ),
            (
                "It is green.",
                [2],
            ),
            (
                "It is colored olive.",
                [2, 3],
            ),
            ("Its beak is pale and short.", [1, 3, 5]),
            ("Its eyes are yellow.", [1, 3]),
            ("It has a bright belly with spots around.", [3, 4, 6]),
            ("Its breast is big and red.", [1, 3, 5]),
            ("Its legs are long and feet are black.", [1, 3, 5, 7]),
            ("Its plumage is black.", [1, 3]),
            ("Its tail is black with orange stripes.", [1, 5, 6]),
            (
                "Their bodies are 20 to 30 cm long, with a 50 to 70 cm wingspan, and body mass of 200 to 300g.",
                [1, 7, 14, 17],
            ),
            ("Their wings are small and purple.", [1, 3, 5]),
            ("They have a curved bill shape and they weigh 15g.", [3, 4, 5, 8]),
            ("They can range in size from 10 to 50 cm.", [4]),
            (
                "Its back is solid pink and has white plumes on its neck.",
                [1, 3, 4, 7, 8, 11],
            ),
            (
                "They are small in size.",
                [2, 4],
            ),
            (
                "They are large in size.",
                [2, 4],
            ),
        ]
        files = [f for f in os.listdir(txt_sentences_path)]
        class_names = []
        for i in range(200):
            for file in files:
                if str(i + 1) == file.split(".")[0]:
                    class_name = file.lower().split(".")[1].split("_")
                    class_names.append(" ".join(class_name))

        # body_words = ["head", "back", "belly", "breast", "leg", "wing", "tail"]
        # extra_words = ["color", "appearance", "size", "pattern"]
        """ CAN EITHER DEFINE SENTENCES and use SBERT for vrs extraction
            OMIT NEED FOR LSTM WITH 7 sentences for each part of the bird describing size, color, and patterns
        """
        # extra_words = [
        #     "beak",
        #     "eye",
        #     "nose",
        #     "body",
        #     "feet",
        #     "wingspan",
        #     "feathers",
        #     "plumage",
        #     "color",
        #     "black",
        #     "white",
        #     "red",
        #     "yellow",
        #     "pale",
        #     "brown",
        #     "dark",
        #     "size",
        #     "measure",
        #     "large",
        #     "small",
        #     "weight",
        #     "patterns",
        #     "stripe",
        #     "spots",
        # ]

        # contexts = [
        #     "Birds have heads",
        #     "Birds have backs",
        #     "Birds have bellies",
        #     "Birds have breasts",
        #     "Birds have legs",
        #     "Birds have wings",
        #     "Birds have tails",
        #     "Birds have beaks",
        #     "Birds have eyes",
        #     "Birds have noses",
        #     "Birds have bodies",
        #     "Birds have feet",
        #     "Birds have wingspans",
        #     "Birds have feathers",
        #     "Birds have plumages",
        #     "Birds are different colors",
        #     "Birds can be white",
        #     "Birds can be black",
        #     "Birds can be yellow",
        #     "Birds can be pale",
        #     "Birds can be brown",
        #     "Birds can be dark",
        #     "Birds can be different sizes",
        #     "Birds can measure differently",
        #     "Birds can be large",
        #     "Birds can be small",
        #     "Birds can weigh different amounts",
        #     "Birds can have different patterns",
        #     "Birds can have spots",
        #     "Birds can have stripes",
        # ]
        embedder = Embedder(opt.txt_feat)
        contexts = []
        for s, idxs in prototypes:
            embedded, tokens = embedder.get_embedding(s)
            for i in idxs:
                word = s.split(" ")[i].strip(",.")
                for j in range(len(tokens)):
                    for k in range(j + 1, len(tokens)):
                        # print("pre dec: ", embedder.decode(tokens[j:k]))
                        decoded = "".join(embedder.decode(tokens[j:k]).split(" "))
                        # print(decoded, word)
                        if decoded == word:
                            # print(decoded, word)
                            for l in range(j, k):
                                embed = embedded[l]
                                contexts.append((word, embed))
                            break
        for w, e in contexts:
            print(w, e.shape)
        # ignore = []
        # class_contexts = []
        # for name, s, idxs in class_names:
        #     embedded, tokens = embedder.get_embedding(s)
        #     ignore_temp = []
        #     for j in idxs:
        #         word = s.split(" ")[j].strip(",.")
        #         word_tokens = embedder.get_tokens(word)

        #         if len(word_tokens.shape) == 0:
        #             word_tokens = [word_tokens.item()]

        #         for token in word_tokens:
        #             ignore_temp.append(len(class_contexts))
        #             embed = embedded[tokens.tolist().index(token)]
        #             class_contexts.append((name, word, embed))

        # ignore.append(set(ignore_temp))

        files = [f for f in os.listdir(txt_sentences_path)]
        texts = []
        texts_tokens = []
        num_words = []
        questions = []

        notes_file = os.path.join(dirname, "latest_notes.txt")
        with open(notes_file, "w") as f:
            f.write("")

        for i in range(200):
            for file in files:
                if str(i + 1) == file.split(".")[0]:
                    class_name = " ".join(file.split(".")[1].split("_"))
                    print(
                        f"\n\n\n********* {i+1}:{class_name} ********************\n\n"
                    )
                    with open(notes_file, "a") as f:
                        f.write(
                            f"\n\n\n********* {i+1}:{class_name} ********************\n\n"
                        )
                    with open(os.path.join(txt_sentences_path, file), "r") as f:
                        lines = f.readlines()

                    sentences = []
                    tokens = []
                    for l in lines:
                        # print(j, line)
                        embedded, tokenized = embedder.get_embedding(l.strip())
                        sentences.append(embedded)
                        tokens.append(tokenized)
                    vrs, vrs_tokens, n_words = get_vrs(
                        sentences,
                        embedder,
                        contexts,
                        tokens=tokens,
                        class_names=class_names,
                        ignore=i,
                        notes_file=notes_file,
                    )

                    print("n_words: ", n_words)
                    if len(vrs) < 5:
                        questions.append(file)
                    num_words.append(n_words)
                    class_name_embed, class_name_tokens = embedder.get_embedding(
                        class_name + "."
                    )
                    texts.append(torch.cat([class_name_embed] + vrs))
                    texts_tokens.append(torch.cat([class_name_tokens] + vrs_tokens))

        texts = pad_sequence(texts, batch_first=True, padding_value=0.0)
        texts_tokens = pad_sequence(texts_tokens, batch_first=True, padding_value=0.0)
        lengths = torch.tensor(num_words)
        print(f"max num words: {max(num_words)}")
        print(questions)
        # print("shapes: ", texts.shape, train_cid.shape)
        embed = texts[train_cid, :, :]
        embed_test = texts[test_cid, :, :]
        tokens = texts_tokens[train_cid, :]
        tokens_test = texts_tokens[test_cid, :]

        lengths_test = lengths[test_cid]
        lengths = lengths[train_cid]

        torch.save(embed.cpu(), os.path.join(dirname, repr + ".pt"))
        torch.save(embed_test.cpu(), os.path.join(dirname, repr + "_test.pt"))
        torch.save(tokens.cpu(), os.path.join(dirname, repr_tokens + ".pt"))
        torch.save(embed_test.cpu(), os.path.join(dirname, repr + "_test.pt"))
        torch.save(tokens_test.cpu(), os.path.join(dirname, repr_tokens + "_test.pt"))
        torch.save(lengths.cpu(), os.path.join(dirname, repr + "_lengths.pt"))
        torch.save(lengths_test.cpu(), os.path.join(dirname, repr + "_lengths_test.pt"))
        print("\n\n Text embeddings computed!!")

    return (
        embed.to(device),
        tokens.to(device),
        lengths.to(device),
        embed_test.to(device),
        tokens_test.to(device),
        lengths_test.to(device),
    )


def get_vrs(
    sentences,
    embedder,
    contexts,
    tokens,
    k=7,
    topk=1,
    class_names=None,
    ignore=None,
    sim_thres=0.7,
    notes_file=None,
):
    if class_names is not None:

        print(f"Ignore: {class_names[ignore]}")
    k = max(7, k)
    vrs = []
    vrs_tokens = []
    vrs_idxs = set()

    total_words = 0
    # for w in body_words:
    #     total_words += get_closest(sentences, w, vrs, vrs_idxs, embedder, tokens)

    # k -= 7
    # for w in extra_words:
    #     if k > 0:
    #         total_words += get_closest(sentences, w, vrs, vrs_idxs, embedder, tokens)
    #         k -= 1

    # Find closest sentences to any of the words
    cos = torch.nn.CosineSimilarity()
    while k > 0 and len(vrs_idxs) < len(sentences):
        sim = -float("inf")
        token = None
        matched = None
        sent = None
        idx = -1
        for i, s in enumerate(sentences):
            if i not in vrs_idxs:
                sims = []
                tokes = []
                protos = []
                for w, tgt in contexts:
                    cos_sims = cos(tgt.expand(s.shape[0], -1), s)
                    sims_max, args_max = torch.topk(cos_sims, topk, dim=0)
                    j = 0
                    # while j < args_max.shape[0] - 1 and tokens[i][args_max[j]] in tokes:
                    #     j += 1
                    sims.append(sims_max[j])
                    protos.append(w)
                    tokes.append(tokens[i][args_max[j]])
                zipped = list(zip(sims, tokes, protos))
                zipped.sort(reverse=True)
                sims, tokes, protos = zip(*zipped)

                max_sim = sum(sims[:topk]) / topk
                if max_sim > sim:
                    sim = max_sim
                    token = tokes[:topk]
                    sent = s
                    idx = i
                    matched = protos[:topk]

        if sim >= sim_thres:
            vrs_idxs.add(idx)
            vrs.append(sent)
            vrs_tokens.append(tokens(idx))
            total_words += sent.shape[0]
            print("")
            print(
                f"Sim: {sim}, Proto: {matched}, Match: {[embedder.decode(t) for t in token]}"
            )
            if tokens:
                print(f"Sentence is __{embedder.decode(tokens[idx])}__")
                print("")

            if notes_file is not None:
                with open(notes_file, "a") as f:
                    f.write("\n")
                    f.write(
                        f"Sim: {sim}, Proto: {matched}, Match: {[embedder.decode(t) for t in token]}"
                    )
                    f.write("\n")
                    if tokens:
                        f.write(f"Sentence is __{embedder.decode(tokens[idx])}__")
                        f.write("\n")

        else:
            break
        k -= 1

    if len(vrs_idxs) < 3 and class_names is not None:
        s = f"Their closest relative is the bird."
        embedded, tokens_ = embedder.get_embedding(s)
        class_contexts = []
        for j in [1, 2]:
            word = s.split(" ")[j].strip(",.")
            for k in range(len(tokens_)):
                for l in range(k + 1, len(tokens_)):
                    decoded = "".join(embedder.decode(tokens_[k:l]).split(" "))
                    if decoded == word:
                        for m in range(k, l):
                            embed = embedded[m]
                            class_contexts.append((word, embed))
                        break

        for _ in range(2):
            sim = -float("inf")
            token = None
            matched = None
            sent = None
            idx = -1
            for i, s in enumerate(sentences):
                if i not in vrs_idxs:
                    sims = []
                    tokes = []
                    protos = []
                    for w, tgt in class_contexts:
                        cos_sims = cos(tgt.expand(s.shape[0], -1), s)
                        sims_max, args_max = torch.topk(cos_sims, topk, dim=0)
                        j = 0
                        # while j < args_max.shape[0] - 1 and tokens[i][args_max[j]] in tokes:
                        #     j += 1
                        sims.append(sims_max[j])
                        protos.append(w)
                        tokes.append(tokens[i][args_max[j]])
                    zipped = list(zip(sims, tokes, protos))
                    zipped.sort(reverse=True)
                    sims, tokes, protos = zip(*zipped)

                    max_sim = sum(sims[:topk]) / topk
                    if max_sim > sim:
                        sim = max_sim
                        token = tokes[:topk]
                        sent = s
                        idx = i
                        matched = protos[:topk]

            raw_sent = embedder.decode(tokens[idx])
            names = []
            for name in class_names:
                for x in name.split(" "):
                    names.append(x)
            if sim >= sim_thres and any([name in raw_sent for name in names]):
                vrs_idxs.add(idx)
                vrs.append(sent)
                vrs_tokens.append(tokens(idx))
                total_words += sent.shape[0]
                print("")
                print(
                    f"Sim: {sim}, Proto: {matched}, Match: {[embedder.decode(t) for t in token]}"
                )
                if tokens:
                    print(f"Sentence is __{embedder.decode(tokens[idx])}__")
                    print("")

                if notes_file is not None:
                    with open(notes_file, "a") as f:
                        f.write("\n")
                        f.write(
                            f"Sim: {sim}, Proto: {matched}, Match: {[embedder.decode(t) for t in token]}"
                        )
                        f.write("\n")
                        if tokens:
                            f.write(f"Sentence is __{embedder.decode(tokens[idx])}__")
                            f.write("\n")

        # sim = -float("inf")
        # sims = []
        # token = []
        # matched = []
        # sent = None
        # idx = -1
        # for i, s in enumerate(sentences):
        #     if i not in vrs_idxs:
        #         sim_temp = 0.0
        #         token_temp = []
        #         matched_temp = []
        #         sims_temp = []
        #         for w in words:
        #             tgt, _ = embedder.get_embedding(w)
        #             cos_sims = cos(tgt.expand(s.shape[0], -1), s)
        #             sim_max, arg_max = torch.max(cos_sims, dim=0)
        #             sim_temp += sim_max
        #             if tokens is not None:
        #                 token_temp.append(tokens[i][arg_max])
        #             matched_temp.append(w)
        #             sims_temp.append(sim_max)
        #         if sim_temp / len(words) > sim:
        #             sim = sim_temp
        #             if tokens is not None:
        #                 token = token_temp
        #             matched = matched_temp
        #             sims = sims_temp
        #             sent = s
        #             idx = i

        # vrs.append(sent)
        # vrs_idxs.add(idx)
        # total_words += sent.shape[0]
        # matcher = [embedder.decode(t) for t in token]
        # matchings = list(zip(sims, matched, matcher))
        # matchings.sort(reverse=True)
        # print("")
        # for s, p, m in matchings:
        #     print(f"Sim: {s}, Proto: {p}, Match: {m}")
        # if token:
        #     print(f"Sentence is __{embedder.decode(tokens[idx])}__")
        #     print("")
    if notes_file is not None:
        with open(notes_file, "a") as f:
            f.write(f"total words: {total_words}\n")
    return vrs, vrs_tokens, total_words


def get_closest(sentences, word, vrs, vrs_idxs, embedder, tokens=None):
    tgt, _ = embedder.get_embedding(word)
    sim = -float("inf")
    token = None
    sent = torch.tensor([])
    idx = -1
    cos = torch.nn.CosineSimilarity()
    for i, s in enumerate(sentences):
        if i not in vrs_idxs:
            # print(
            #     "\n\ncos s: ",
            #     tgt.expand(s.shape[0], -1).shape,
            #     s.shape,
            # )
            cos_sims = cos(tgt.expand(s.shape[0], -1), s)
            # print(
            #     "\n\ncos s: ",
            #     t.unsqueeze(0).expand(s.shape[0], -1).shape,
            #     s.shape,
            #     cos_sims.shape,
            # )
            sim_max, arg_max = torch.max(cos_sims, dim=0)
            # print("\n\ns: ", sim_max.shape, arg_max.shape)
            if sim_max > sim:
                sim = sim_max
                if tokens is not None:
                    token = tokens[i][arg_max]
                sent = s
                idx = i

    vrs.append(sent)
    vrs_idxs.add(idx)
    if token is not None:
        print(f"Closest word to __{word}__ is __{embedder.decode(token)}__")
        print(f"Sentence is __{embedder.decode(tokens[idx])}__")
        print("")

    return sent.shape[0]


def get_sentence_embeddings(train_test_split_dir, txt_sentences_path, opt):
    if not opt.embed:
        return torch.tensor([]), torch.tensor([]), torch.tensor([]), torch.tensor([])

    repr = "sentence_embed"

    if opt.pretrained_path is not None:
        repr += "_pretrained"

    output_dir = "preprocessing/"
    dirname = os.path.join(output_dir, opt.txt_feat)
    os.makedirs(dirname, exist_ok=True)
    device = opt.device
    try:
        embed = torch.load(os.path.join(dirname, repr + ".pt")).to(device)
        embed_test = torch.load(os.path.join(dirname, repr + "_test.pt")).to(device)
        lengths = torch.load(os.path.join(dirname, repr + "_lengths.pt")).to(device)
        lengths_test = torch.load(os.path.join(dirname, repr + "_lengths_test.pt")).to(
            device
        )
        print("\n\n Embeddings Loaded!")
    except Exception:
        print("\n\n Computing Embeddings!")
        train_test_split = sio.loadmat(train_test_split_dir)

        train_cid = train_test_split["train_cid"].squeeze() - 1
        test_cid = train_test_split["test_cid"].squeeze() - 1
        train_cid = torch.tensor(train_cid).long().reshape((-1,))
        test_cid = torch.tensor(test_cid).long().reshape((-1,))

        files = [f for f in os.listdir(txt_sentences_path)]
        texts = []
        num_sentences = []
        embedder = Embedder()
        for i in range(200):
            for file in files:
                if str(i + 1) == file.split(".")[0]:
                    sentences = []
                    with open(os.path.join(txt_sentences_path, file), "r") as f:
                        lines = f.readlines()
                    for j, l in enumerate(lines):
                        line = l[len(str(j)) + 1 :].strip()
                        # print(j, line)
                        embedded = embedder.get_single_sentence_embedding(
                            line,
                            transformer=opt.txt_feat,
                            pooling=opt.embed_pooling,
                        )
                        sentences.append(embedded.unsqueeze(0))
                        # print(embedded)
                        # print('\n')
                    # print(file, len(sentences))
                    num_sentences.append(len(sentences))
                    texts.append(torch.cat(sentences))

        texts = pad_sequence(texts, batch_first=True, padding_value=0.0)
        lengths = torch.tensor(num_sentences)

        # print("shapes: ", texts.shape, train_cid.shape)
        embed = texts[train_cid, :, :]
        embed_test = texts[test_cid, :, :]

        lengths_test = lengths[test_cid]
        lengths = lengths[train_cid]

        torch.save(embed.cpu(), os.path.join(dirname, repr + ".pt"))
        torch.save(embed_test.cpu(), os.path.join(dirname, repr + "_test.pt"))
        torch.save(lengths.cpu(), os.path.join(dirname, repr + "_lengths.pt"))
        torch.save(lengths_test.cpu(), os.path.join(dirname, repr + "_lengths_test.pt"))
        print("\n\n Text embeddings computed!!")

    return (
        embed.to(device),
        lengths.to(device),
        embed_test.to(device),
        lengths_test.to(device),
    )


def get_text_embeddings(train_test_split_dir, txt_raw_path, opt):
    if not opt.embed:
        return torch.tensor([]), torch.tensor([]), torch.tensor([]), torch.tensor([])
    train_test_split = sio.loadmat(train_test_split_dir)
    train_cid = train_test_split["train_cid"].squeeze() - 1
    test_cid = train_test_split["test_cid"].squeeze() - 1
    files = [f for f in os.listdir(txt_raw_path)]
    texts = []
    for i in range(200):
        for file in files:
            # print(str(i+1), file[:len(str(i+1))], file)
            if str(i + 1) == file.split(".")[0]:
                # print(file)
                with open(os.path.join(txt_raw_path, file), "r") as f:
                    lines = f.readlines()
                for j, l in enumerate(lines):
                    lines[j] = l[len(str(j)) + 1 :].strip()
                texts.append(" ".join(lines))
                break

    texts = np.array(texts)
    embedder = Embedder()
    tokenize_only = False
    repr = "embed"
    if opt.unfrozen_layers > 0:
        tokenize_only = True
        repr = "token"

    if opt.pretrained_path is not None:
        repr += "_pretrained"

    output_dir = "preprocessing/"
    dirname = os.path.join(output_dir, opt.txt_feat)
    os.makedirs(dirname, exist_ok=True)
    device = opt.device
    try:
        embed = torch.load(os.path.join(dirname, repr + ".pt")).to(device)
        lengths = torch.load(os.path.join(dirname, repr + "_lengths.pt")).to(device)
        embed_test = torch.load(os.path.join(dirname, repr + "_test.pt")).to(device)
        lengths_test = torch.load(os.path.join(dirname, repr + "_lengths_test.pt")).to(
            device
        )
        print("\n\n Embeddings Loaded!")
    except Exception:
        print("\n Computing Embeddings...")
        os.makedirs(dirname, exist_ok=True)
        if tokenize_only:
            embed, lengths = embedder.get_tokens(
                texts[train_cid],
                transformer=opt.txt_feat,
                max_sentence_length=opt.max_sentence_length,
            )
            embed_test, lengths_test = embedder.get_tokens(
                texts[train_cid],
                transformer=opt.txt_feat,
                max_sentence_length=opt.max_sentence_length,
            )
        else:
            embed, lengths = embedder.get_sentence_embeddings(
                texts[train_cid],
                transformer=opt.txt_feat,
                pooling=opt.embed_pooling,
                pretrained_path=opt.pretrained_path,
            )
            embed_test, lengths_test = embedder.get_sentence_embeddings(
                texts[test_cid],
                transformer=opt.txt_feat,
                pooling=opt.embed_pooling,
                pretrained_path=opt.pretrained_path,
            )

        torch.save(embed.cpu(), os.path.join(dirname, repr + ".pt"))
        torch.save(lengths.cpu(), os.path.join(dirname, repr + "_lengths.pt"))
        torch.save(embed_test.cpu(), os.path.join(dirname, repr + "_test.pt"))
        torch.save(lengths_test.cpu(), os.path.join(dirname, repr + "_lengths_test.pt"))
        print("\n\n Text embeddings computed!!")

    return (
        embed.to(device),
        lengths.to(device),
        embed_test.to(device),
        lengths_test.to(device),
    )


def get_text_feature(dir, train_test_split_dir, txt_feat_path_original, opt):
    train_test_split = sio.loadmat(train_test_split_dir)
    train_cid = train_test_split["train_cid"].squeeze() - 1
    test_cid = train_test_split["test_cid"].squeeze() - 1

    text_feature = sio.loadmat(dir)["PredicateMatrix"]
    text_feature_original = sio.loadmat(txt_feat_path_original)["PredicateMatrix"]

    text_feature_new, intersections = add_similarity(
        text_feature, train_cid, test_cid, text_feature_original
    )
    if (
        intersections.shape[0] / test_cid.shape[0] > 0.15
        and opt.txt_feat == "similarity_VRS"
    ):
        print("added similarity features")
        text_feature = text_feature_new

    train_text_feature = text_feature[train_cid].astype(np.float32)  # 0-based index
    test_text_feature = text_feature[test_cid].astype(np.float32)  # 0-based index

    return train_text_feature, test_text_feature


def family_addition_features(text_features, path_text):
    files = os.listdir(path_text)
    files_xls = [f for f in files if f[-3:] == "txt"]
    files_xls.sort(key=lambda x: int(x.split(".")[0]))

    dic_family = {}
    list_family = []
    for counter, file_name in enumerate(files_xls):
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
