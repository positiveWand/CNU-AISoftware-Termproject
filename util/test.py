from process import get_extended_attention_mask
import torch

a = torch.Tensor(
    [
        [1,1,0,0],
        [1,0,0,0],
        [0,0,0,0],
        [1,1,1,0]
    ]
)
print(get_extended_attention_mask(a, use_causal=False))
print(get_extended_attention_mask(a, use_causal=True))
