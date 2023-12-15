import pytorch_lightning as pl
from tqdm import trange
import pandas as pd
import os, sys
import time
import datetime
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__name__))))

from PIL import Image
from util.augment import *
from util.generate import *
from util.graph_tokenizer import *
from transformer.transformer import *
from img_graph import Img2Graph # 본인 모델 라이브러리 import
def cli_main():
    pl.seed_everything(1234)
    os.makedirs('./release/graph/', exist_ok=True)
    ie = ImageEncoder(
        input_resolution=256,
        patch_size=24,
        hidden_dim=128,
        intermediate_dim=128*4,
        head_num=8,
        block_num=8,
        drop_rate=0.1,
        ln_eps=1e-5
    )
    td = TextDecoder(
        token_num=57,
        max_seq_length=12,
        token_type_num=0,
        hidden_dim=128,
        intermediate_dim=128*4,
        head_num=8,
        block_num=8,
        drop_rate=0.1,
        ln_eps=1e-5,
        use_cache=True
    )

    model = Img2Graph.load_from_checkpoint(
        './model/graph/Img2Graph-epoch=29.ckpt',
        img_encoder=ie,
        text_decoder=td
    )
    tokenizer = GraphTokenizer()
    torch.set_grad_enabled(False)

    student_id = 201802114 # 본인의 학번
    task = "image_to_graph" # [image_to_shape, text_to_color] 목적에 맞는 task 이름으로 설정

    # ---- Validation DATA에 대한 prediction --- #
    to_fn = f"./release/graph/{student_id}.{task}.valid.txt"
    description = f"Generating Validation {task} {student_id}"
    excel = pd.read_excel("./data/valid/scene.all.xlsx", engine='openpyxl', index_col="id")
    ## Generating all outputs for the testing sentences
    start_id = 7000
    end_id = 8500
    start_time = time.time()
    with open(to_fn, 'w', encoding='utf-8') as f:
        for id in trange(start_id, end_id, desc=description):
            # id 통한 데이터 불러오기, excel 활용
            img = Image.open(excel.loc[id]['image_fn'].replace('\\', '/'))
            graph = {
                'input_ids': torch.tensor([[tokenizer.encode_token('[CLS]')]]).long().to(model.device),
                'attention_mask': None
            }
            # 데이터 전처리
            img = img_transform(256)(img)
            img = img.to(model.device)
            img = img.unsqueeze(dim=0)

            # 모델에 입력하고 출력
            pred = generator(ctx_img=img, 
                      ctx_text=graph, 
                      tokenizer=tokenizer, 
                      model=model, 
                      post_process=postprocess_graph, 
                      top_k=1, 
                      use_cache=True
            )
            # 출력 후처리 -> completion
            completion = pred
            print(f"{id}\t{completion}", file=f)
            f.flush()
        f.close()
    end_time = time.time()


    # # ---- Test DATA에 대한 prediction ---  #
    # to_fn = f"./release/graph/{student_id}.{task}.test.txt"
    # description = f"Generating Test {task} {student_id}"
    # excel = pd.read_excel("./data/test/scene.stu.xlsx", engine='openpyxl', index_col="id")
    # ## Generating all outputs for the testing sentences
    # start_id = 8500
    # end_id = 10000
    # start_time = time.time()
    # with open(to_fn, 'w', encoding='utf-8') as f:
    #     for id in trange(start_id, end_id, desc=description):
    #         # id 통한 데이터 불러오기, excel 활용
    #         img = Image.open(excel.loc[id]['image_fn'])
    #         graph = {
    #             'input_ids': torch.tensor([[tokenizer.encode_token('[CLS]')]]).long().to(model.device),
    #             'attention_mask': None
    #         }
    #         # 데이터 전처리
    #         img = img_transform(256)(img)
    #         img = img.to(model.device)
    #         img = img.unsqueeze(dim=0)

    #         # 모델에 입력하고 출력
    #         pred = generator(ctx_img=img, 
    #                   ctx_text=graph, 
    #                   tokenizer=tokenizer, 
    #                   model=model, 
    #                   post_process=postprocess_graph, 
    #                   top_k=1, 
    #                   use_cache=True
    #         )
    #         # 출력 후처리 -> completion
    #         completion = pred
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