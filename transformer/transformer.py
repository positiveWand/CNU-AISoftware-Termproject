import torch
from easydict import EasyDict
from torch import nn
from torch.nn import functional as F
from .decoder import *
from .encoder import *
from .embedding import *
import copy

class TextEncoder(nn.Module):
    def __init__(self, token_num: int, max_seq_length: int, token_type_num: int,
                 hidden_dim: int, intermediate_dim: int,
                 head_num: int, block_num: int, 
                 drop_rate=0.1, ln_eps=1e-4):
        super().__init__()

        self.hidden_dim = hidden_dim

        self.config = EasyDict({
            'token_num': token_num,         # token 개수
            'max_seq_length': max_seq_length, # Encoder Sequence 최대 길이
            'token_type_num': token_type_num,    # Encoder token type 종류 개수
            'hidden_dim': hidden_dim,              # hidden state 벡터 차원
            'intermediate_dim': intermediate_dim,    # FeedForward 내부 증대 차원
            'head_num': head_num,                 # Attention Head 개수
            'block_num': block_num,        # Decoder 블록 개수
            'drop_rate': drop_rate,               # Dropout 비율
            'ln_eps': ln_eps,                 # Layernorm EPS 값
            'use_cache': False,              # Attention Key, Value 캐싱 여부
        })

        # Encoder
        self.embedding = SequenceEmbedding(hidden_dim=hidden_dim, total_token_num=token_num, max_seq_length=max_seq_length, token_type_num=token_type_num, drop_rate=drop_rate, ln_eps=ln_eps)
        self.encoder = Encoder(self.config)

    def forward(self, input_ids: torch.Tensor, position_ids: torch.Tensor=None, token_type_ids: torch.Tensor=None, attention_mask: torch.Tensor=None):
        x = self.embedding(input_ids, position_ids, token_type_ids)
        x = self.encoder(x, attention_mask)

        return x

class ImageEncoder(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int,
                 hidden_dim: int, intermediate_dim: int, 
                 head_num: int, block_num: int,
                 drop_rate=0.1, ln_eps=1e-4):
        super().__init__()

        self.hidden_dim = hidden_dim

        self.config = EasyDict({
            'max_seq_length': (input_resolution // patch_size) ** 2 + 1, # Encoder Sequence 최대 길이
            'hidden_dim': hidden_dim,              # hidden state 벡터 차원
            'intermediate_dim': intermediate_dim,    # FeedForward 내부 증대 차원
            'head_num': head_num,                 # Attention Head 개수
            'block_num': block_num,        # Decoder 블록 개수
            'drop_rate': drop_rate,               # Dropout 비율
            'ln_eps': ln_eps,                 # Layernorm EPS 값
            'use_cache': False,              # Attention Key, Value 캐싱 여부
        })

        self.embedding_conv = nn.Conv2d(in_channels=3, out_channels=hidden_dim, kernel_size=patch_size, stride=patch_size, bias=False)
        
        # scale = hidden_dim ** -0.5
        # self.cls_embedding = nn.Parameter(scale * torch.randn(hidden_dim))
        # self.position_embedding = nn.Parameter(scale * torch.randn((self.config.max_seq_length, hidden_dim)))
        # self.proj = nn.Parameter(scale * torch.randn(hidden_dim, output_dim))
        scale = hidden_dim ** -0.5
        self.cls_embedding = nn.Parameter(scale * torch.randn(hidden_dim))
        self.position_embedding = nn.Embedding(self.config.max_seq_length, hidden_dim)
        self.register_buffer("position_ids", torch.arange(self.config.max_seq_length).expand((1, -1)), persistent=False)

        self.ln_eps = ln_eps
        self.ln = nn.LayerNorm(self.config.hidden_dim, eps=self.ln_eps)
        self.drop_rate = drop_rate
        self.dropout = nn.Dropout(self.drop_rate)

        # Encoder
        self.encoder = Encoder(self.config)
    
    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor=None):
        x = self.embedding_conv(x) # (Batch Size, Hidden Dim, Grid, Grid)
        x = x.reshape(x.shape[0], x.shape[1], -1) # (Batch Size, Hidden Dim, Grid ** 2)
        x = x.permute(0,2,1) # (Batch Size, Grid ** 2, Hidden Dim)
        x = torch.cat([
            self.cls_embedding + torch.zeros(x.shape[0], 1, x.shape[-1], device=x.device),
            x
        ], dim=1) # (Batch Size, Grid ** 2 + 1, Hidden Dim)
        x = x + self.position_embedding(self.position_ids)
        
        x = self.ln(x)
        x = self.encoder(x, attention_mask)

        return x
    
class TextDecoder(nn.Module):
    def __init__(self, token_num: int, max_seq_length: int, token_type_num: int,
                 hidden_dim: int, intermediate_dim: int,
                 head_num: int, block_num: int, 
                 drop_rate=0.1, ln_eps=1e-4, use_cache=False):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.token_num = token_num

        self.config = EasyDict({
            'token_num': token_num,         # token 개수
            'max_seq_length': max_seq_length, # Encoder Sequence 최대 길이
            'token_type_num': token_type_num,    # Encoder token type 종류 개수
            'hidden_dim': hidden_dim,              # hidden state 벡터 차원
            'intermediate_dim': intermediate_dim,    # FeedForward 내부 증대 차원
            'head_num': head_num,                 # Attention Head 개수
            'block_num': block_num,        # Decoder 블록 개수
            'drop_rate': drop_rate,               # Dropout 비율
            'ln_eps': ln_eps,                 # Layernorm EPS 값
            'use_cache': use_cache,              # Attention Key, Value 캐싱 여부
        })

        # Encoder
        self.embedding = SequenceEmbedding(hidden_dim=hidden_dim, total_token_num=token_num, max_seq_length=max_seq_length, token_type_num=token_type_num, drop_rate=drop_rate, ln_eps=ln_eps)
        self.decoder = Decoder(self.config)

    def forward(self,
                input_ids: torch.Tensor, 
                encoder_output: torch.Tensor, 
                position_ids: torch.Tensor=None, 
                token_type_ids: torch.Tensor=None, 
                decoder_mask: torch.Tensor=None, 
                encoder_mask: torch.Tensor=None, 
                past_key_values: tuple=None):
        
        past_key_values_len = past_key_values[0][0][0].shape[-2] if past_key_values is not None else 0
        
        x = self.embedding(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            past_key_values_len=past_key_values_len)

        x, new_past_key_values = self.decoder(x, encoder_output, decoder_mask, encoder_mask, past_key_values)

        return x, new_past_key_values