import pytorch_lightning as pl
from tqdm import trange
import pandas as pd
import os, sys
import time
import datetime
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__name__))))

from util.augment import *
from transformer.transformer import *
from text_color import Text2Color # 본인 모델 라이브러리 import
from transformers import BertTokenizer
def cli_main():
    pl.seed_everything(1234)
    os.makedirs('./release/color/', exist_ok=True)
    te = TextEncoder(
        token_num=45,
        max_seq_length=45,
        token_type_num=0,
        hidden_dim=128,
        intermediate_dim=128*4,
        head_num=8,
        block_num=6,
        drop_rate=0.1,
        ln_eps=1e-5
    )

    model = Text2Color.load_from_checkpoint(
        './model/color/Text2Color-epoch=26.ckpt',
        text_encoder = te
    )
    tokenizer = BertTokenizer.from_pretrained('./model/color/', do_lower_case=False)
    torch.set_grad_enabled(False)

    student_id = 201802114 # 본인의 학번
    task = "text_to_color" # [image_to_shape, text_to_color] 목적에 맞는 task 이름으로 설정

    # ---- Validation DATA에 대한 prediction --- #
    to_fn = f"./release/color/{student_id}.{task}.valid.txt"
    description = f"Generating Validation {task} {student_id}"
    excel = pd.read_excel("./data/valid/scene.all.xlsx", engine='openpyxl', index_col="id")
    ## Generating all outputs for the testing sentences
    start_id = 7000
    end_id = 8500
    start_time = time.time()
    with open(to_fn, 'w', encoding='utf-8') as f:
        for id in trange(start_id, end_id, desc=description):
            # id 통한 데이터 불러오기, excel 활용
            text = excel.loc[id]['text']
            # 데이터 전처리
            text = text_transform('은/는', '은는')(text)
            text = tokenizer(
                text=text,
                add_special_tokens=True,
                max_length=45,
                padding='max_length',
                return_token_type_ids=False,
                return_tensors='pt'
            )
            text['input_ids'] = text['input_ids'].to(model.device)
            text['attention_mask'] = text['attention_mask'].to(model.device)
            # 모델에 입력하고 출력
            pred = model(text)
            pred = F.sigmoid(pred)
            pred = pred >= 0.5
            result = []
            for i in range(5):
                if pred[0, i]:
                    if i == 0:
                        result.append('red')
                    elif i == 1:
                        result.append('blue')
                    elif i == 2:
                        result.append('green')
                    elif i == 3:
                        result.append('yellow')
                    elif i == 4:
                        result.append('black')
            result = list(set(result))
            result.sort()
            # 출력 후처리 -> completion
            completion = ' , '.join(map(str, result))
            print(f"{id}\t{completion}", file=f)
            f.flush()
        f.close()
    end_time = time.time()


    # # ---- Test DATA에 대한 prediction ---  #
    # to_fn = f"./release/color/{student_id}.{task}.test.txt"
    # description = f"Generating Test {task} {student_id}"
    # excel = pd.read_excel("./data/test/scene.stu.xlsx", engine='openpyxl', index_col="id")
    # ## Generating all outputs for the testing sentences
    # start_id = 8500
    # end_id = 10000
    # start_time = time.time()
    # with open(to_fn, 'w', encoding='utf-8') as f:
    #     for id in trange(start_id, end_id, desc=description):
    #         # id 통한 데이터 불러오기, excel 활용
    #         text = excel.loc[id]['text']
    #         # 데이터 전처리
    #         text = text_transform('은/는', '은는')(text)
    #         text = tokenizer(
    #             text=text,
    #             add_special_tokens=True,
    #             max_length=45,
    #             padding='max_length',
    #             return_token_type_ids=False,
    #             return_tensors='pt'
    #         )
    #         text['input_ids'] = text['input_ids'].to(model.device)
    #         text['attention_mask'] = text['attention_mask'].to(model.device)
    #         # 모델에 입력하고 출력
    #         pred = model(text)
    #         pred = F.sigmoid(pred)
    #         pred = pred >= 0.5
    #         result = []
    #         for i in range(5):
    #             if pred[0, i]:
    #                 if i == 0:
    #                     result.append('red')
    #                 elif i == 1:
    #                     result.append('blue')
    #                 elif i == 2:
    #                     result.append('green')
    #                 elif i == 3:
    #                     result.append('yellow')
    #                 elif i == 4:
    #                     result.append('black')
    #         result = list(set(result))
    #         result.sort()
    #         # 출력 후처리 -> completion
    #         completion = ' , '.join(map(str, result))
    #         print(f"{id}\t{completion}", file=f)
    #         f.flush()
    #     f.close()
    # end_time = time.time()

    print("[Save] Generated texts -- ", to_fn)
    sec = (end_time - start_time)
    result = datetime.timedelta(seconds=sec)
    print(f"{task} {student_id} take: {result}")


   
if __name__ == '__main__':
    cli_main()