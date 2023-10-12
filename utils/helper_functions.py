import torch.nn.functional as F

def get_norm_function(name):
    if name == "relu":
        return F.relu
    if name == "sigmoid":
        return F.sigmoid
    if name == "tanh":
        return F.tanh
