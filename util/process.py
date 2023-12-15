import torch
from .object import *

def get_extended_attention_mask(attention_mask: torch.Tensor, use_causal: bool):
    # attention_mask.shape = (Batch Size, Sequence Length)
    # extended_attention_mask.shape = (Batch Size, 1, Sequence Length, Sequence Length)
    # 1 -> 마스킹 안함
    # 0 -> 마스킹 함
    if attention_mask is None:
        return None
    batch_size, seq_len = attention_mask.shape

    extended_attention_mask = attention_mask.unsqueeze(dim=1).repeat((1, seq_len, 1))

    if use_causal:
        causal_mask = torch.ones(extended_attention_mask.shape, device=attention_mask.device).tril()
        extended_attention_mask = torch.logical_and(extended_attention_mask, causal_mask).int()

    extended_attention_mask = extended_attention_mask.unsqueeze(dim=1)
    return extended_attention_mask

def postprocess_text(text: str):
    assert text.startswith('[CLS]') and '[SEP]' in text, text

    text = text.replace('[CLS]', '')
    text = text.replace('[PAD]', '')
    end = text.find('[SEP]')
    text = text[:end]
    text = text.replace('. ;', '.;')
    text = text.replace('은는', '은/는')

    text = text.strip()
    # print('processed text:', text)

    return text

def is_object(token):
    shapes = [0,1,2,3,4]
    colors = ['red', 'blue', 'green', 'yellow', 'black']
    return len(token) == 2 and token[0] in shapes and token[1] in colors
def is_relation(token):
    x_rels = ['L', 'R', 'Sx']
    y_rels = ['F', 'B', 'Sy']
    z_rels = ['T', 'U', 'Sz']
    return len(token) == 3 and token[0] in x_rels and token[1] in y_rels and token[2] in z_rels
def postprocess_graph(graph_seq: list):
    # assert graph_seq[0] == '[CLS]' and '[SEP]' in graph_seq, graph_seq

    graph = []
    node = ()
    for i in graph_seq:
        if i == '[CLS]' or i == '[PAD]' or i == '[MASK]' or i == '[UNK]':
            continue
        elif i == '[SEP]':
            if len(node) != 0:
                graph.append(node)
            break
        elif is_object(i):
            node += (i,)
        elif is_relation(i):
            node += (i,)
            graph.append(node)
            node = ()

    return graph

def generate_text_from_scene(id: int, scene: tuple):
    text_candidate =  Scene(id, scene, None).generate_text()
    all_texts = []
    if len(text_candidate) == 1:
        for first_sentence in text_candidate[0]:
            all_texts.append(first_sentence)
    elif len(text_candidate) == 2:
        for first_sentence in text_candidate[0]:
            for second_sentence in text_candidate[1]:
                all_texts.append(first_sentence + '; ' + second_sentence)
    elif len(text_candidate) == 3:
        for first_sentence in text_candidate[0]:
            for second_sentence in text_candidate[1]:
                for third_sentence in text_candidate[2]:
                    all_texts.append(first_sentence + '; ' + second_sentence + '; ' + third_sentence)
    
    return all_texts