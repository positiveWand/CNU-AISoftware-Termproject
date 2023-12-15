import torch
from torch import nn
from torch.nn import functional as F
from .multihead import MultiHeadAttention
from .feedforward import FeedForward

class EncoderBlock(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.self_attention = MultiHeadAttention(config)
        self.linear = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.dropout = nn.Dropout(config.drop_rate)
        self.feedforward = FeedForward(config)

        self.ln_1 = nn.LayerNorm(config.hidden_dim, eps=config.ln_eps)
        self.ln_2 = nn.LayerNorm(config.hidden_dim, eps=config.ln_eps)

    def forward(self, x, attention_mask=None):
        residual = x
        x = self.ln_1(x)
        x, _ = self.self_attention(x, x, x, attention_mask)
        x = self.linear(x)
        x = residual + self.dropout(x)

        residual = x
        x = self.ln_2(x)
        x = residual + self.feedforward(x)
        return x

class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config

        self.blocks = nn.ModuleList([EncoderBlock(config) for _ in range(config.block_num)])

    def forward(self, x, attention_mask=None):
        output = x

        for i, block in enumerate(self.blocks):
            output = block(output, attention_mask)

        return output