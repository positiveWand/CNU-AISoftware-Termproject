import torch
from torch import nn
from torch.nn import functional as F
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config

        assert config.hidden_dim % config.head_num == 0

        self.use_cache = config.use_cache

        self.hidden_dim = config.hidden_dim
        self.head_num = config.head_num
        self.head_dim = self.hidden_dim // self.head_num

        self.q_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.k_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.v_proj = nn.Linear(config.hidden_dim, config.hidden_dim)

        self.dropout = nn.Dropout(config.drop_rate)

    def forward(self, q, k, v, attention_mask=None, past_key_value=None, is_cross_attention=False):
        assert q.size()[-1] == k.size()[-1] and k.size()[-1] == v.size()[-1] and v.size()[-1] == q.size()[-1]
        batch_size, sequence_len, input_dim = q.size()

        Q = self.q_proj(q)
        # (Batch size, Query sequence length, Hidden dim)
        Q = Q.view(batch_size, -1, self.head_num, self.head_dim).transpose(1,2)
        # (Batch size, Head num, Query sequence length, Head dim)

        if is_cross_attention and past_key_value is not None:
            K = past_key_value[0]
            V = past_key_value[1]
        else:
            K = self.k_proj(k)
            # (Batch size, Key sequence length, Hidden dim)
            V = self.v_proj(v)
            # (Batch size, Value sequence length, Hidden dim)
            K = K.view(batch_size, -1, self.head_num, self.head_dim).transpose(1,2)
            # (Batch size, Head num, Key sequence length, Head dim)
            V = V.view(batch_size, -1, self.head_num, self.head_dim).transpose(1,2)
            # (Batch size, Head num, Value sequence length, Head dim)

            if past_key_value is not None:
                # past_K = past_key_value[0]
                # past_V = past_key_value[1]
                K = torch.cat([past_key_value[0], K], dim=2)
                # (Batch size, Head num, Past Key sequence length + Key sequence length, Head dim)
                V = torch.cat([past_key_value[1], V], dim=2)
                # (Batch size, Head num, Past Value sequence length + Value sequence length, Head dim)

        if self.use_cache:
            past_key_value = (K, V)
        else:
            past_key_value = None

        attn = torch.matmul(Q, K.transpose(-1,-2)) / math.sqrt(self.head_dim)
        # (Batch size, Head num, Query Sequence length, Key Sequence length)
        if attention_mask is not None:
            attn = attn.masked_fill(attention_mask==0, float('-inf'))
        
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        y = attn @ V
        # (Batch size, Head num, Query Sequence length, Head dim)
        y = y.transpose(1,2).contiguous().view((batch_size, -1, input_dim))
        # (Batch size, Query Sequence length, Hidden dim)
        return y, past_key_value