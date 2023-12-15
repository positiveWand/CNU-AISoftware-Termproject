import torch
from torch import nn
from torch.nn import functional as F
from pytorch_lightning import LightningModule
import numpy as np
import math
from util.process import *

class PretrainCLIP(LightningModule):
    def __init__(self,
                 img_encoder: torch.nn.Module,
                 text_encoder: torch.nn.Module,
                 output_dim: int, ln_eps,
                 learning_rate,
                 warmup_tokens,
                 final_tokens,
                 weight_decay,
                 adamw_betas
                 ):
        super().__init__()
        self.save_hyperparameters(ignore=['img_encoder', 'text_encoder'])

        self.tokens = 0

        self.img_encoder = img_encoder
        self.text_encoder = text_encoder

        self.ln_img = nn.LayerNorm(img_encoder.hidden_dim, eps=ln_eps)
        self.ln_text = nn.LayerNorm(text_encoder.hidden_dim, eps=ln_eps)
        self.img_proj = nn.Linear(img_encoder.hidden_dim, output_dim, bias=False)
        self.text_proj = nn.Linear(text_encoder.hidden_dim, output_dim, bias=False)

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    #     self.apply(self._init_weights)

    # def _init_weights(self, module):
    #     if isinstance(module, (nn.Linear, nn.Parameter, nn.Embedding, nn.Conv2d)):
    #         module.weight.data.normal_(mean=0.0, std=0.02)
    #         if isinstance(module, (nn.Linear, nn.Conv2d)) and module.bias is not None:
    #             module.bias.data.zero_()
    #     elif isinstance(module, nn.LayerNorm):
    #         module.bias.data.zero_()
    #         module.weight.data.fill_(1.0)
    
    def forward(self, img, text):
        img_hidden = self.img_encoder.forward(img)
        attention_mask = get_extended_attention_mask(attention_mask=text['attention_mask'], use_causal=False)
        text_hidden = self.text_encoder.forward(input_ids=text['input_ids'], attention_mask=attention_mask)

        img_representation = self.ln_img(img_hidden)
        img_representation = self.img_proj(img_representation[:,0,:])

        text_representation = self.ln_text(text_hidden)
        text_representation = self.text_proj(text_representation[:,0,:])

        img_representation = img_representation / img_representation.norm(p=2, dim=1, keepdim=True)
        text_representation = text_representation / text_representation.norm(p=2, dim=1, keepdim=True)

        return img_representation, text_representation
        
    def training_step(self, batch, batch_idx):
        img, text = batch
        img_representation, text_representation = self.forward(img, text)

        self.tokens += (text['input_ids'] >= 0).sum()

        logits = torch.matmul(img_representation, text_representation.transpose(-1,-2)) * self.logit_scale.exp()

        labels = torch.arange(len(logits), device=logits.device)

        loss_i = nn.functional.cross_entropy(logits, labels)
        loss_t = nn.functional.cross_entropy(logits, labels)
        loss = loss_i + loss_t

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss
    
    def validation_step(self, batch, batch_idx):
        img, text = batch
        img_representation, text_representation = self.forward(img, text)
        logits = torch.matmul(img_representation, text_representation.transpose(-1,-2)) * self.logit_scale.exp()
        labels = torch.arange(img.shape[0], device=logits.device)
        loss_i = nn.functional.cross_entropy(logits, labels)
        loss_t = nn.functional.cross_entropy(logits, labels)
        loss = loss_i + loss_t

        self.log('val_loss', loss, on_epoch=True, logger=True)

        return loss
    
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
        no_decay.add('img_encoder.cls_embedding')
        no_decay.add('logit_scale')

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
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=optimizer, T_0=550, T_mult=2)
        # return [optimizer], [scheduler]

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