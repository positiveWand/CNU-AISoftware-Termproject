from torch.utils.data import Dataset, DataLoader, random_split
from pytorch_lightning import LightningDataModule
import pandas as pd
import pathlib
from PIL import Image
import torch
from util.augment import *
import random
from transformers import BertTokenizer

class PretrainDataset(Dataset):
    def __init__(self,
                 id: pd.Series,
                 text: pd.Series,
                 img_dir: pathlib.Path,
                 vocab_path: pathlib.Path,
                 max_text_length
                 ):
        self.id = id
        self.text = text
        self.img_dir = img_dir
        self.img_transform = img_transform(256)
        self.img_augment_transform = img_augment_transform(256, 15)
        self.text_transform = text_transform('은/는', '은는')
        self.tokenizer = BertTokenizer.from_pretrained(vocab_path, do_lower_case=False)
        self.max_text_length = max_text_length
    
    def __len__(self):
        return len(self.text)
    
    def __getitem__(self, index):
        id = self.id[index]
        filename = str(id) + '.jpg'
        img = Image.open(self.img_dir / filename)
        if random.random() <= 0.5:
            img = self.img_transform(img)
        else:
            img = self.img_augment_transform(img)

        text = self.text_transform(self.text[index])
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

        return img, text
    
class PretrainDataModule(LightningDataModule):
    def __init__(self,
                 batch_size: pathlib.Path,
                 train_path: pathlib.Path,
                 valid_path: pathlib.Path,
                 vocab_path: pathlib.Path,
                 max_text_length
                 ):
        super().__init__()

        self.batch_size = batch_size
        self.train_path = train_path
        self.valid_path = valid_path
        self.vocab_path = vocab_path
        self.max_text_length = max_text_length

        self.train_dataset = None
        self.validation_dataset = None

    def setup(self, stage='fit', train_frac=0.8):
        train_img_dir = self.train_path / 'images'
        valid_img_dir = self.valid_path / 'images'

        train_scene_all = pd.read_excel(self.train_path / 'scene.all.xlsx')
        valid_scene_all = pd.read_excel(self.valid_path / 'scene.all.xlsx')
        train_id = train_scene_all['id']
        train_text = train_scene_all['text']
        valid_id = valid_scene_all['id']
        valid_text = valid_scene_all['text']

        self.train_dataset = PretrainDataset(id=train_id, text=train_text, img_dir=train_img_dir, vocab_path=self.vocab_path, max_text_length=self.max_text_length)
        self.validation_dataset = PretrainDataset(id=valid_id, text=valid_text, img_dir=valid_img_dir, vocab_path=self.vocab_path, max_text_length=self.max_text_length)
        
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.validation_dataset, batch_size=self.batch_size)