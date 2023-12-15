import torch
from torch import nn
from torch.nn import functional as F
from transformers.activations import gelu_new

class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.mlp_1 = nn.Linear(config.hidden_dim, config.intermediate_dim)
        self.mlp_2 = nn.Linear(config.intermediate_dim, config.hidden_dim)
        self.dropout = nn.Dropout(config.drop_rate)

    def forward(self, x):
        x = self.mlp_1(x)
        x = gelu_new(x)
        x = self.mlp_2(x)

        y = self.dropout(x)

        return y