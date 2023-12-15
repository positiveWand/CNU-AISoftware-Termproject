from torch.utils.data import Dataset, DataLoader, random_split
from pytorch_lightning import LightningDataModule
import pandas as pd
import pathlib
from PIL import Image
import torch
from util.augment import *
import random

class ImgShapeDataset(Dataset):
    def __init__(self,
                 id: pd.Series,
                 scene: pd.Series,
                 img_dir: pathlib.Path
                 ):
        self.id = id
        self.scene = scene
        self.img_dir = img_dir
        self.transform = img_transform(256)
        self.augment_transform = img_augment_transform(256, 15)
    
    def __len__(self):
        return len(self.scene)
    
    def __getitem__(self, index):
        id = self.id[index]
        filename = str(id) + '.jpg'
        img = Image.open(self.img_dir / filename)
        if random.random() <= 0.5:
            img = self.transform(img)
        else:
            img = self.augment_transform(img)

        scene = eval(self.scene[index])
        target = [0] * 5
        for obj in scene:
            shape = self.get_shape(obj)
            target[shape] = 1
        
        target = torch.Tensor(target)

        return img, target

    def get_shape(self, obj):
        return obj[1][0]
    
class ImgShapeDataModule(LightningDataModule):
    def __init__(self,
                 batch_size: pathlib.Path,
                 train_path: pathlib.Path,
                 valid_path: pathlib.Path):
        super().__init__()

        self.batch_size = batch_size
        self.train_path = train_path
        self.valid_path = valid_path

        self.train_dataset = None
        self.validation_dataset = None

    def setup(self, stage='fit', train_frac=0.8):
        train_img_dir = self.train_path / 'images'
        valid_img_dir = self.valid_path / 'images'

        train_scene_all = pd.read_excel(self.train_path / 'scene.all.xlsx')
        valid_scene_all = pd.read_excel(self.valid_path / 'scene.all.xlsx')
        train_id = train_scene_all['id']
        train_scene = train_scene_all['scene']
        valid_id = valid_scene_all['id']
        valid_scene = valid_scene_all['scene']

        self.train_dataset = ImgShapeDataset(id=train_id, scene=train_scene, img_dir=train_img_dir)
        self.validation_dataset = ImgShapeDataset(id=valid_id, scene=valid_scene, img_dir=valid_img_dir)
        
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.validation_dataset, batch_size=self.batch_size)