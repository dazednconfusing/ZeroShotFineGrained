# from dataloaders.CUBLoader import CUBDataSet, CUBFeatDataSet, NABFeatDataSet

# from torchvision.transforms import RandomCrop, Compose, Resize, PILToTensor

# from torch.utils.data import Dataset, DataLoader


# transforms = Compose(([Resize(128), RandomCrop(64, 64), PILToTensor()]))

# # dataset = CUBDataSet("/Users/nikhilt/Desktop/Sp21/proj/CIS620/data/CUB2011", image_transforms=transforms)
# # for i in range(len(dataset)):
# #     k = dataset[i]
# #     print(k[2].label, k[1].label_info, k[1].captions)

# dataset = DataLoader(NABFeatDataSet("/Users/nikhilt/Desktop/Sp21/proj/CIS620/data/NABird", split="easy"), batch_size=100)

# # print(dataset.bert_test_feat.shape)
# # print(dataset.bert_train_feat.shape)
# for i in dataset:
#     img, lab, org = i
#     print(img.shape)
#     break

import torch
from transformers import AutoModel, AutoTokenizer

from preprocessing.embedder import Embedder

embedder = Embedder("bert")
tgt, tokens = embedder.get_embedding("small")

query, qt = embedder.get_embedding("a small member of the family but kind of large.")
print("\n\n")
# t1 = embedder.get_tokens("small").reshape((-1,))[1:-1]
# t2 = embedder.get_tokens("a small member of the family but kind of large.").reshape(
#     (-1,)
# )[1:-1]
# print(t1.shape, t2.shape)
# print(t1.item())
# print("t2:")
# print(t2)
# for t in t2:
#     print(t.item() == t1.item())
print(tokens.shape, qt.shape)
print(tokens.item())
print("t2:")
print(qt)
for t in qt:
    print(t.item() == tokens.item())
# for q in query:
#     print(tgt.shape, q.unsqueeze(0).shape, "\n")
#     print(torch.all(tgt.eq(q.unsqueeze(0))))

bert = AutoModel.from_pretrained("bert-base-uncased")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

with torch.no_grad():
    e1 = tokenizer("small", return_tensors="pt")
    e1_ids = e1["input_ids"]
    out = bert(e1_ids)
    o1 = out.last_hidden_state[0, 1, :]

    print(out.last_hidden_state.shape)
    e2 = tokenizer("a small member of the family", return_tensors="pt")
    e2_ids = e2["input_ids"]
    out = bert(e2_ids)
    print(e1_ids, e2_ids)
    o2 = out.last_hidden_state[0, :, :]
    print("")
    for o in o2:
        print((o1 - o).norm())
