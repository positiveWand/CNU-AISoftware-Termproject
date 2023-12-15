import torch
from torch import nn
from torch.nn import functional as F

class SequenceEmbedding(nn.Module):
    def __init__(self, hidden_dim, total_token_num, max_seq_length, token_type_num, drop_rate=0.2, ln_eps=0.00001, config=None):
        super().__init__()

        assert total_token_num is not None and max_seq_length is not None and token_type_num is not None
        assert total_token_num > 0 or max_seq_length > 0 or token_type_num > 0

        self.embedding_dim = hidden_dim
        self.total_token_num = total_token_num
        self.max_seq_length = max_seq_length
        self.token_type_num = token_type_num

        self.token_embedding = None
        self.position_embedding = None
        self.token_type_embedding = None

        if self.total_token_num is not None and self.total_token_num > 0:
            self.token_embedding = nn.Embedding(self.total_token_num, self.embedding_dim, padding_idx=0)
        if self.max_seq_length is not None and self.max_seq_length > 0:
            self.position_embedding = nn.Embedding(self.max_seq_length, self.embedding_dim)
        if self.token_type_num is not None and self.token_type_num > 0:
            self.token_type_embedding = nn.Embedding(self.token_type_num, self.embedding_dim)

        self.ln_eps = ln_eps
        self.ln = nn.LayerNorm(self.embedding_dim, eps=self.ln_eps)

        self.drop_rate = drop_rate
        self.dropout = nn.Dropout(self.drop_rate)

        # 브로드캐스팅을 위해 맨 앞에 차원 1개 추가
        self.register_buffer(
            "position_ids", torch.arange(self.max_seq_length).expand((1, -1)), persistent=False
        )
        self.register_buffer(
            "token_type_ids", torch.zeros(self.position_ids.size(), dtype=torch.long), persistent=False
        )

    def forward(self, input_ids: torch.Tensor, position_ids=None, token_type_ids=None, past_key_values_len=0):
        batch_size, seq_length = input_ids.shape # (Batch Size, Sequence Length)
        device = input_ids.device

        if position_ids is None:
            # past_key_values의 개수를 고려한 position 계산
            # past_key_values들에 대해서는 임베딩 계산 불필요
            position_ids = self.position_ids[:, past_key_values_len : seq_length + past_key_values_len]

        if token_type_ids is None:
            token_type_ids = self.token_type_ids[:, :seq_length].expand(input_ids.size()[0], seq_length)

        total_embeddings = torch.zeros((batch_size, seq_length, self.embedding_dim), device=device)

        if self.token_embedding is not None:
            total_embeddings += self.token_embedding(input_ids)
        if self.position_embedding is not None:
            total_embeddings += self.position_embedding(position_ids)
        if self.token_type_embedding is not None:
            total_embeddings += self.token_type_embedding(token_type_ids)
        
        total_embeddings = self.ln(total_embeddings)
        total_embeddings = self.dropout(total_embeddings)
        return total_embeddings