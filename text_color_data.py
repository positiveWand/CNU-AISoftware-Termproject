from torch.utils.data import Dataset, DataLoader, random_split
from pytorch_lightning import LightningDataModule
import pandas as pd
import pathlib
import torch
from util.augment import *
from transformers import BertTokenizer

class TextColorDataset(Dataset):
    def __init__(self,
                 id: pd.Series,
                 scene: pd.Series,
                 text: str,
                 vocab_path: pathlib.Path,
                 max_text_length
                 ):
        self.id = id
        self.scene = scene
        self.text = text
        self.transform = text_transform('은/는', '은는')
        self.tokenizer = BertTokenizer.from_pretrained(vocab_path, do_lower_case=False)
        self.max_text_length = max_text_length
    
    def __len__(self):
        return len(self.scene)
    
    def __getitem__(self, index):
        text = self.transform(self.text[index])
        text = self.tokenizer(
            text=text,
            add_special_tokens=True,
            max_length=self.max_text_length,
            padding='max_length',
            return_token_type_ids=False,
            return_tensors='pt'
        )
        text = {
            'input_ids': text['input_ids'][0],
            'attention_mask': text['attention_mask'][0]
        }

        scene = eval(self.scene[index])
        target = [0] * 5
        for obj in scene:
            color = self.get_color(obj)
            target[color] = 1
        
        target = torch.Tensor(target)

        return text, target

    def get_color(self, obj):
        color = obj[0]
        return self.encode_color(color)
    
    @classmethod
    def encode_color(cls, color)-> int:
        code = None
        if color == 'red':
            code = 0
        elif color == 'blue':
            code = 1
        elif color == 'green':
            code = 2
        elif color == 'yellow':
            code = 3
        elif color == 'black':
            code = 4
        return code
    @classmethod
    def decode_color(cls, code)-> str:
        color = None
        if code == 0:
            color = 'red'
        elif code == 1:
            color = 'blue'
        elif code == 2:
            color = 'green'
        elif code == 3:
            color = 'yellow'
        elif code == 4:
            color = 'black'
        return color
    
class TextColorDataModule(LightningDataModule):
    def __init__(self,
                 batch_size: pathlib.Path,
                 train_path: pathlib.Path,
                 valid_path: pathlib.Path,
                 vocab_path: pathlib.Path,
                 max_text_length):
        super().__init__()

        self.batch_size = batch_size
        self.train_path = train_path
        self.valid_path = valid_path
        self.vocab_path = vocab_path
        self.max_text_length = max_text_length

        self.train_dataset = None
        self.validation_dataset = None

    def setup(self, stage='fit', train_frac=0.8):
        train_scene_all = pd.read_excel(self.train_path / 'scene.all.xlsx')
        valid_scene_all = pd.read_excel(self.valid_path / 'scene.all.xlsx')
        train_id = train_scene_all['id']
        train_scene = train_scene_all['scene']
        train_text = train_scene_all['text']
        valid_id = valid_scene_all['id']
        valid_scene = valid_scene_all['scene']
        valid_text = valid_scene_all['text']

        self.train_dataset = TextColorDataset(
            id=train_id, 
            scene=train_scene, 
            text=train_text,
            vocab_path=self.vocab_path,
            max_text_length=self.max_text_length
        )
        self.validation_dataset = TextColorDataset(
            id=valid_id,
            scene=valid_scene,
            text=valid_text,
            vocab_path=self.vocab_path,
            max_text_length=self.max_text_length
        )
        
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.validation_dataset, batch_size=self.batch_size)