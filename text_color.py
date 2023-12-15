import torch
from torch import nn
from torch.nn import functional as F
from torchmetrics.functional.classification import multilabel_accuracy
from pytorch_lightning import LightningModule
import numpy as np
import math
from util.process import *

class Text2Color(LightningModule):
    def __init__(self,
                 text_encoder: torch.nn.Module,
                 intermediate_dim, color_num, ln_eps, drop_rate,
                 learning_rate,
                 warmup_tokens,
                 final_tokens,
                 weight_decay,
                 adamw_betas,
                 threshold=0.5
                ):
        super().__init__()
        self.save_hyperparameters(ignore=['text_encoder'])

        self.tokens = 0
        self.color_num = color_num

        self.text_encoder = text_encoder
        self.ln = nn.LayerNorm(text_encoder.hidden_dim, eps=ln_eps)
        self.text_proj = nn.Linear(text_encoder.hidden_dim, intermediate_dim, bias=False)
        self.dropout = nn.Dropout(drop_rate)
        self.classifier = nn.Sequential(
            nn.Linear(intermediate_dim, color_num)
        )
    
    def forward(self, text):
        attention_mask = get_extended_attention_mask(attention_mask=text['attention_mask'], use_causal=False)
        text_hidden = self.text_encoder.forward(input_ids=text['input_ids'], attention_mask=attention_mask)

        text_representation = self.ln(text_hidden)
        text_representation = self.text_proj(text_representation[:,0,:])

        text_representation = text_representation / text_representation.norm(p=2, dim=1, keepdim=True)

        y = self.dropout(text_representation)
        y = self.classifier(y)

        return y
        
    def training_step(self, batch, batch_idx):
        text, target = batch
        logits = self.forward(text)
        loss = F.multilabel_soft_margin_loss(logits, target)

        logits = F.sigmoid(logits)
        accuracy = multilabel_accuracy(
            logits, target,
            num_labels=self.hparams.color_num,
            threshold=self.hparams.threshold,
            average='macro'
        )
        self.tokens += (text['input_ids'] >= 0).sum()

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_accuracy', accuracy, on_epoch=True, logger=True)

        return loss
    
    def validation_step(self, batch, batch_idx):
        text, target = batch
        logits = self.forward(text)
        loss = F.multilabel_soft_margin_loss(logits, target)

        logits = F.sigmoid(logits)
        accuracy = multilabel_accuracy(
            logits, target,
            num_labels=self.hparams.color_num,
            threshold=self.hparams.threshold,
            average='macro'
        )

        self.log('val_loss', loss, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_accuracy', accuracy, on_epoch=True, logger=True)

        return loss

    def predict_step(self, batch):
        img, target = batch

        logits = self.forward(img)
        logits = F.sigmoid(logits)        

        return logits
    
    def configure_optimizers(self):
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (nn.Linear, nn.Conv2d)
        blacklist_weight_modules = (nn.LayerNorm, nn.Embedding, nn.Parameter)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name

                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        # no_decay.add('pos_emb')

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {
                "params": [param_dict[pn] for pn in sorted(list(decay))],
                "weight_decay": self.hparams.weight_decay
            },
            {
                "params": [param_dict[pn] for pn in sorted(list(no_decay))], 
                "weight_decay": 0.0
            },
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=self.hparams.learning_rate, betas=self.hparams.adamw_betas, eps=1e-08)
        return optimizer

    def is_warm_up_phase(self):
        if self.tokens < self.hparams.warmup_tokens:
            return True
        else: return False

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
        # learning rate warm-up
        if self.is_warm_up_phase():
            # [warm-up phase]
            lr_mult = float(self.tokens) / float(max(1, self.hparams.warmup_tokens))
        else:
            # [decay phase]
            progress = float(self.tokens - self.hparams.warmup_tokens) / float(max(1, self.hparams.final_tokens - self.hparams.warmup_tokens))
            lr_mult = max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))

        new_lr = self.hparams.learning_rate * lr_mult

        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr

        self.cur_lr = new_lr

        # update params
        optimizer.step(closure=optimizer_closure)
    
    def load_from_pretrain(self, encoder_state_dict, proj_state_dict, ln_state_dict):
        if encoder_state_dict is not None:
            self.text_encoder.load_state_dict(encoder_state_dict)
        if proj_state_dict is not None:
            self.text_proj.load_state_dict(proj_state_dict)
        if ln_state_dict is not None:
            self.ln.load_state_dict(ln_state_dict)