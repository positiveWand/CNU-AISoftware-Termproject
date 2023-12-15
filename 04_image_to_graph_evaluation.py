import pandas as pd
from tqdm import tqdm
import numpy as np
from torchmetrics.functional.text import bleu_score
from util.graph_tokenizer import *

def load_data(file_path):
    df = pd.read_table(file_path,sep='\t',header=None,names=['id','value'])
    return df

def image_to_graph_eval(in_fn, ref_fn, out_fn):
    in_df = load_data(in_fn)
    ref_df = load_data(ref_fn)
    # check number of data
    assert len(in_df) == len(ref_df), "input data({}) and reference data({}) must have the same number of elements!".format(len(in_df), len(ref_df))

    tokenizer = GraphTokenizer()
    
    total_score, error_idx = [], []
    for idx in tqdm(range(len(in_df))):
        pred_value = in_df.iloc[idx]['value']
        ref_value  = ref_df.iloc[idx]['value']

        # 예측값이 존재하지 않는 경우 계산 안되도록
        if pd.isna(pred_value):
            continue

        try:
            pred_value = tokenizer.encode(eval(pred_value))['input_ids']
            pred_value = ' '.join(map(str, pred_value))

            ref_value = tokenizer.encode(eval(ref_value))['input_ids']
            ref_value = ' '.join(map(str, ref_value))

            score = []
            score += [bleu_score(pred_value, [ref_value], n_gram=1).item()]
            score += [bleu_score(pred_value, [ref_value], n_gram=2).item()]
            score += [bleu_score(pred_value, [ref_value], n_gram=3).item()]
            score += [bleu_score(pred_value, [ref_value], n_gram=4).item()]

            total_score.append(score)
        except:
            error_idx.append(idx)
            continue

    # 평균 accuracy 저장
    # 'w' : 쓰기모드, 'a' : 이미 파일 존재할 경우 이어서 작성
    with open(out_fn, 'a', encoding='utf-8', newline='\n') as f:
        print('---- [BLEU] ----')
        print(f'BLEU-1\t{np.mean(total_score, axis=0)[0]}\n')
        print(f'BLEU-2\t{np.mean(total_score, axis=0)[1]}\n')
        print(f'BLEU-3\t{np.mean(total_score, axis=0)[2]}\n')
        print(f'BLEU-4\t{np.mean(total_score, axis=0)[3]}\n')
        print(f'BLEU-1\t{np.mean(total_score, axis=0)[0]}', file=f)
        print(f'BLEU-2\t{np.mean(total_score, axis=0)[1]}', file=f)
        print(f'BLEU-3\t{np.mean(total_score, axis=0)[2]}', file=f)
        print(f'BLEU-4\t{np.mean(total_score, axis=0)[3]}', file=f)
    print("Save : {}\n".format(out_fn))
    
    print('---- [Error] ----') # 오답 idx 출력 -- 하나라도 틀린 경우 출력
    print(error_idx)

# ------------
# args
# ------------
from argparse import ArgumentParser
parser = ArgumentParser()
# -- valid data로 성능 확인할 것 -- #
parser.add_argument('--in_fn', default='./release/graph/201802114.image_to_graph.valid.txt', type=str)
parser.add_argument('--ref_fn', default='./release/graph/reference.valid.txt', type=str)
parser.add_argument('--out_fn', default='./release/graph/bleu.txt', type=str)
args = parser.parse_args()

if __name__ == '__main__':
    image_to_graph_eval(args.in_fn,args. ref_fn, args.out_fn)