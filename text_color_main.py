from typing import Any
from easydict import EasyDict
from pytorch_lightning.utilities.types import STEP_OUTPUT
from transformers import BertTokenizer
from transformer.transformer import *
import math
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from pretrain import *
from text_color import *
from text_color_data import *

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

data_dir = pathlib.Path('.').absolute() / 'data'
train_aug_dir = data_dir / 'train_augmented'
train_aug_img_dir = train_aug_dir / 'images'
valid_dir = data_dir / 'valid'
valid_img_dir = valid_dir / 'images'
model_folder = pathlib.Path('.').absolute() / 'model'

dm = TextColorDataModule(
    batch_size=10,
    train_path=train_aug_dir,
    valid_path=valid_dir,
    vocab_path=pathlib.Path('.').absolute(),
    max_text_length=45
)
dm.setup()

pretrain_model = PretrainCLIP.load_from_checkpoint(
    model_folder / 'pretrain/PretrainCLIP-epoch=23.ckpt',
    img_encoder = ie,
    text_encoder = te
)

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
# train_epochs = 50
train_epochs = 30
tokens_per_epoch = len(dm.train_dataset) * dm.train_dataset.max_text_length
model = Text2Color(
    text_encoder=te,
    intermediate_dim=128,
    color_num=5,
    ln_eps=1e-5,
    drop_rate=0.1,
    # learning_rate=0.00005,
    learning_rate=0.00002,
    warmup_tokens=tokens_per_epoch,
    final_tokens=train_epochs*tokens_per_epoch,
    weight_decay=0.01,
    adamw_betas=(0.9, 0.97)
)
model.load_from_pretrain(
    pretrain_model.text_encoder.state_dict(),
    pretrain_model.text_proj.state_dict(),
    pretrain_model.ln_text.state_dict()
)

checkpoint_callback = ModelCheckpoint(
    monitor='train_loss',
    dirpath=model_folder,
    filename='Text2Color-{epoch:02d}',
    save_on_train_epoch_end=True,
    save_top_k=5
)
logger = TensorBoardLogger(model_folder, name='tensorboard')
trainer = Trainer(
    max_epochs=train_epochs,
    accelerator="gpu",devices=1,
    logger = logger,
    callbacks=[
            checkpoint_callback,
            LearningRateMonitor(logging_interval='step'),
            EarlyStopping(monitor='val_loss', patience=3)
    ],
    gradient_clip_val=0.5,
    gradient_clip_algorithm="norm",
    accumulate_grad_batches=3
)
trainer.fit(model, datamodule=dm)