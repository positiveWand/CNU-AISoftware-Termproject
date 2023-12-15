import torch
from torch.nn import functional as F
from .process import *
import copy

def top_k_logits(logits, k):
    v, ix = torch.topk(logits, k)
    out = logits.clone()
    out[out < v[:, [-1]]] = -float('Inf') # top k의 원소 중 최소값보다 작으면 제거
    return out

@torch.no_grad()
def sample(model, ctx_img, ctx_text, steps, temperature=1.0, sample=False, top_k=None, use_cache=False):
    """
    take a conditioning sequence of indices in x (of shape (b,t)) and predict the next token in
    the sequence, feeding the predictions back into the model each time. Clearly the sampling
    has quadratic complexity unlike an RNN that is only linear, and has a finite context window
    of block_size, unlike an RNN that has an infinite context window.
    """
    model.eval()

    next_token = {
        'input_ids': torch.tensor(ctx_text['input_ids']),
        'attention_mask': None
    }
    past_key_values = None
    img_hidden_cache = None
    for k in range(steps):
        if not use_cache:
            logits, _, img_hidden_cache = model(ctx_img, ctx_text, None, img_hidden_cache)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                logits = top_k_logits(logits, top_k)
            probs = F.softmax(logits, dim=-1)
            if sample:
                ix = torch.multinomial(probs, num_samples=1)
            else:
                _, ix = torch.topk(probs, k=1, dim=-1)
            ctx_text['input_ids'] = torch.cat((ctx_text['input_ids'], ix), dim=1)
        else:
            logits, past_key_values, img_hidden_cache = model(ctx_img, next_token, past_key_values, img_hidden_cache)

            # for a in past_key_values:
            #     print(a[0][0].shape, a[0][1].shape, a[1][0].shape, a[1][1].shape)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                logits = top_k_logits(logits, top_k)
            probs = F.softmax(logits, dim=-1)
            if sample:
                ix = torch.multinomial(probs, num_samples=1)
            else:
                _, ix = torch.topk(probs, k=1, dim=-1)

            next_token = {
                'input_ids': torch.tensor(ix),
                'attention_mask': None
            }
            ctx_text['input_ids'] = torch.cat((ctx_text['input_ids'], ix), dim=1)
            # print(ctx_text['input_ids'].shape, ctx_text['input_ids'])
        
        if ix[0][0] == 3:
            break
    
    return ctx_text['input_ids']

def generator(ctx_img, ctx_text, tokenizer, model, post_process, top_k=3, use_cache=False, decode=True):
    y = sample(model, ctx_img, ctx_text, 45, temperature=2.0, sample=True, top_k=top_k, use_cache=use_cache)[0]
    # completion = ''.join([dm.train_dataset.itos[int(i)] for i in y]) # decoding
    if decode:
        completion = tokenizer.decode(y)
        # 생성된 문장 후처리
        return post_process(completion)
    else:
        return y