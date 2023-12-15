import torch
from torch import nn
from torch.nn import functional as F
from .multihead import MultiHeadAttention
from .feedforward import FeedForward

class DecoderBlock(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.self_attention = MultiHeadAttention(config)
        self.linear_1 = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.dropout_1 = nn.Dropout(config.drop_rate)
        self.cross_attention = MultiHeadAttention(config)
        self.linear_2 = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.dropout_2 = nn.Dropout(config.drop_rate)
        self.feedforward = FeedForward(config)

        self.ln_1 = nn.LayerNorm(config.hidden_dim, eps=config.ln_eps)
        self.ln_2 = nn.LayerNorm(config.hidden_dim, eps=config.ln_eps)
        self.ln_3 = nn.LayerNorm(config.hidden_dim, eps=config.ln_eps)

    def forward(self, x, encoder_output, decoder_mask=None, encoder_mask=None, past_key_value=None):
        self_past_key_value = past_key_value[0] if past_key_value is not None else None
        cross_past_key_value = past_key_value[1] if past_key_value is not None else None

        residual = x
        x = self.ln_1(x)
        x, self_past_key_value = self.self_attention(x, x, x, decoder_mask, self_past_key_value)
        x = self.linear_1(x)
        x = residual + self.dropout_1(x)

        residual = x
        x = self.ln_2(x)
        x, cross_past_key_value = self.cross_attention(x, encoder_output, encoder_output, encoder_mask, cross_past_key_value, is_cross_attention=True)
        x = self.linear_2(x)
        x = residual + self.dropout_2(x)

        residual = x
        x = self.ln_3(x)
        x = residual + self.feedforward(x)
        return x, (self_past_key_value, cross_past_key_value)

class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.use_cache = config.use_cache

        self.dropout = nn.Dropout(config.drop_rate)
        self.blocks = nn.ModuleList([DecoderBlock(config) for _ in range(config.block_num)])

    def forward(self, x, encoder_output, decoder_mask=None, encoder_mask=None, past_key_values=None):
        output = x

        next_decoder_cache = () if self.use_cache else None
        for i, block in enumerate(self.blocks):
            past_key_value = past_key_values[i] if past_key_values is not None else None
            output, past_key_value = block(output, encoder_output, decoder_mask, encoder_mask, past_key_value)

            if next_decoder_cache is not None:
                next_decoder_cache += (past_key_value,)

        return output, next_decoder_cache