
from torch import nn
import torch.nn.functional as F


class TextToImage(nn.Module):
    def __init__(self, img_dim, txt_dim, hidden_dim_1, norm):
        super().__init__()
        self.fc1 = nn.Linear(txt_dim, hidden_dim_1)
        self.fc2 = nn.Linear(hidden_dim_1, img_dim)
        self.norm = norm

    def forward(self, x):
        x = self.norm(self.fc1(x))
        x = self.norm(self.fc2(x))
        return x
