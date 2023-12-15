import pandas as pd
from tqdm import tqdm
import numpy as np

def load_data(file_path):
    df = pd.read_table(file_path,sep='\t',header=None,names=['id','value'])
    return df

def text_to_color_eval(in_fn, ref_fn, out_fn):
    in_df = load_data(in_fn)
    ref_df = load_data(ref_fn)
    # check number of data
    assert len(in_df) == len(ref_df), "input data({}) and reference data({}) must have the same number of elements!".format(len(in_df), len(ref_df))
    
    total_acc, wrong_idx = [], []
    for idx in tqdm(range(len(in_df))):
        pred_value = in_df.iloc[idx]['value']
        ref_value  = ref_df.iloc[idx]['value']

        # 예측값이 존재하지 않는 경우 계산 안되도록
        if pd.isna(pred_value):
            continue

        # ' , ' 기준으로 split 후 sorting
        pred_list = pred_value.split(' , ')
        pred_list.sort() 

        ref_list = ref_value.split(' , ')
        ref_list.sort()

        acc = []
        for ref in ref_list:
            if ref in pred_list: acc.append(1)
            else: acc.append(0)
        mean_acc = np.mean(acc)

        total_acc.append(mean_acc)

        if mean_acc != 1.0:
            wrong_idx.append(idx)

    # 평균 accuracy 저장
    # 'w' : 쓰기모드, 'a' : 이미 파일 존재할 경우 이어서 작성
    with open(out_fn, 'a', encoding='utf-8', newline='\n') as f:
        print('---- [Accuracy] ----')
        print(f'Accuracy\t{np.mean(total_acc)}\n')
        print(f'Accuracy\t{np.mean(total_acc)}', file=f)
    print("Save : {}\n".format(out_fn))

    print('---- [Wrong Answer] ----') # 오답 idx 출력 -- 하나라도 틀린 경우 출력
    print(wrong_idx)
    

# ------------
# args
# ------------
from argparse import ArgumentParser
parser = ArgumentParser()
# -- valid data로 성능 확인할 것 -- #
parser.add_argument('--in_fn', default='./release/color/201802114.text_to_color.valid.txt', type=str)
parser.add_argument('--ref_fn', default='./release/color/reference.valid.txt', type=str)
parser.add_argument('--out_fn', default='./release/color/accuracy.txt', type=str)
args = parser.parse_args()

if __name__ == '__main__':
    text_to_color_eval(args.in_fn,args. ref_fn, args.out_fn)