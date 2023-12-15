from torch.utils.data import Dataset, DataLoader, random_split
from pytorch_lightning import LightningDataModule
import pandas as pd
import pathlib
from PIL import Image
import torch
from util.augment import *
import random
from util.graph_tokenizer import GraphTokenizer

class ImgGraphDataset(Dataset):
    def __init__(self,
                 id: pd.Series,
                 graph: pd.Series,
                 img_dir: pathlib.Path,
                 max_seq_length
                 ):
        self.id = id
        self.graph = graph
        self.img_dir = img_dir
        self.img_transform = img_transform(256)
        self.img_augment_transform = img_augment_transform(256, 10)
        self.tokenizer = GraphTokenizer()
        self.max_seq_length = max_seq_length
    
    def __len__(self):
        return len(self.graph)
    
    def __getitem__(self, index):
        id = self.id[index]
        filename = str(id) + '.jpg'
        img = Image.open(self.img_dir / filename)
        if random.random() <= 0.5:
            img = self.img_transform(img)
        else:
            img = self.img_augment_transform(img)

        graph = eval(self.graph[index])
        graph = self.tokenizer(
            graphs=[graph],
            add_special_tokens=True,
            max_length=12,
            padding=True
        )
        target = graph['input_ids'][0][1:]
        graph = {
            'input_ids': graph['input_ids'][0][:-1],
            'attention_mask': graph['attention_mask'][0][:-1]
        }

        return img, graph, target
    
class ImgGraphDataModule(LightningDataModule):
    def __init__(self,
                 batch_size: pathlib.Path,
                 train_path: pathlib.Path,
                 valid_path: pathlib.Path,
                 max_seq_length
                 ):
        super().__init__()

        self.batch_size = batch_size
        self.train_path = train_path
        self.valid_path = valid_path
        self.max_seq_length = max_seq_length

        self.train_dataset = None
        self.validation_dataset = None

    def setup(self, stage='fit', train_frac=0.8):
        train_img_dir = self.train_path / 'images'
        valid_img_dir = self.valid_path / 'images'

        train_scene_all = pd.read_excel(self.train_path / 'scene.all.xlsx')
        valid_scene_all = pd.read_excel(self.valid_path / 'scene.all.xlsx')
        train_id = train_scene_all['id']
        train_graph = train_scene_all['graph']
        valid_id = valid_scene_all['id']
        valid_graph = valid_scene_all['graph']

        self.train_dataset = ImgGraphDataset(id=train_id, graph=train_graph, img_dir=train_img_dir, max_seq_length=self.max_seq_length)
        self.validation_dataset = ImgGraphDataset(id=valid_id, graph=valid_graph, img_dir=valid_img_dir, max_seq_length=self.max_seq_length)
        
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.validation_dataset, batch_size=self.batch_size)