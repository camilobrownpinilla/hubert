import torch
import os

import torch.nn as nn
import lightning as L
import torchmetrics.functional as FM

from lightning.pytorch.callbacks import ModelCheckpoint
from transformers import AutoModelForMaskedLM, AutoConfig

from configs.config import BaseConfig
from configs.registry import MODEL_DICT
from .language_modeling import MLMHead
from .hypformer import HypFormer
from .optim import Optimizer, build_scheduler

SLURM_JOB_ID  = os.environ.get("SLURM_JOB_ID")

class HUBERT(nn.Module):

    def __init__(self, args, model_cfg):
        """Initialize a HUBERT (Hyperbolic BERT) model

        Args:
            args: Object containing HypFormer arguments
            model_cfg: Object containing HUBERT arguments
        """
        super().__init__()
        self.hidden_dim = model_cfg.hidden_dim
        self.vocab_size = model_cfg.vocab_size

        self.embedding = HUBERTEmbeddings(model_cfg)
        self.layers = HypFormer(
            in_channels=self.hidden_dim,
            hidden_channels=self.hidden_dim,
            out_channels=self.hidden_dim,
            trans_num_layers=model_cfg.n_layers,
            trans_num_heads=model_cfg.n_heads,
            trans_dropout=model_cfg.dropout,
            args=args
        )

    def forward(self, x, mask=None):
        if mask is None:
            # If no mask provided, assumes pad token is 0 and masks pad tokens
            mask = (x > 0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)
        x = self.embedding(x) #[bsz, seq_len, hidden_dim]
        return self.layers(x, mask=mask)
    
class HUBERTForLM(nn.Module):
    """
    Hyperbolic BERT with Masked Language Modeling head
    """
    def __init__(self, hubert: HUBERT):
        super().__init__()
        self.hubert = hubert
        self.mlm = MLMHead(hubert)

    def forward(self, x, mask=None):
        x = self.hubert(x, mask=mask)
        return self.mlm(x)
    
class REVISEDHUBERTForMaskedLM(nn.Module):
    def __init__(self, hubert: HUBERT):
        super().__init__()
        self.activation = nn.functional.gelu

        self.hubert = hubert
        self.vocab_transform = nn.Linear(hubert.hidden_dim, hubert.hidden_dim)
        self.vocab_layer_norm = nn.LayerNorm(hubert.hidden_dim, eps=1e-12)
        self.vocab_projector = nn.Linear(hubert.hidden_dim, hubert.vocab_size)

        # Should I do some kind of weight init?

    def forward(self, input_ids, mask=None):
        hidden_states = self.hubert(input_ids, mask=mask)
        prediction_logits = self.vocab_transform(hidden_states)
        prediction_logits = self.activation(prediction_logits)
        prediction_logits = self.vocab_layer_norm(prediction_logits)
        prediction_logits = self.vocab_projector(prediction_logits)

        return prediction_logits

    
class HUBERTLightning(L.LightningModule):
    """
    Wrapper around Hubert models for distributed training via Pytorch Lightning
    """
    def __init__(self, model_cfg, hyp_args):
        super().__init__()
        self.args = hyp_args

        self.hubert = HUBERT(self.args, model_cfg)
        self.model = HUBERTForLM(self.hubert)
        self.opts = Optimizer(self.model, self.args)
        self.automatic_optimization = False # We use 2 optimizers for euc/hyp params 
        self.criterion = nn.CrossEntropyLoss()

        self.save_hyperparameters(ignore=['model'])
        self.callbacks = [
            ModelCheckpoint(dirpath='/n/netscratch/sham_lab/Everyone/cbrownpinilla/models/hubert',
                            monitor='val_acc',
                            mode='max',
                            filename='hubert-{step}-{val_loss:.2f}',
                            every_n_train_steps=10_000,
                            save_last=True)
        ]
        self.total_throughput = 0 # Monitor how many tokens our model sees

        # Initialize all weights in self.model (including the HUBERT backbone)
        self.model.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.xavier_uniform_(module.weight)
            if getattr(module, "bias", None) is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def iteration_step(self, batch):
        x, mask, labels = batch['input_ids'], batch['attention_mask'], batch['labels']
        if self.training:
            self.total_throughput += batch['input_ids'].view(-1).shape[0] # process (bsz * seq_len) tokens
        logits = self.model(x, mask=mask).transpose(1, 2)
        preds = torch.argmax(logits, dim=1)
        acc = FM.accuracy(preds, 
                          labels, 
                          task='multiclass',
                          num_classes=self.model.hubert.vocab_size,
                          ignore_index=-100)
        loss = self.criterion(logits, labels)
        return loss, acc
    
    def training_step(self, batch, batch_idx):
        loss, acc = self.iteration_step(batch)

        opts = self.opts
        opts.zero_grad()
        self.manual_backward(loss)
        # log gradient norm
        grad_norm = torch.sqrt(
            sum(p.grad.pow(2).sum() for p in self.model.parameters() if p.grad is not None)
        )
        self.log('grad_norm', grad_norm)
        # log learning rate(s)
        for opt in self.opts.optimizer:
            for i, pg in enumerate(opt.param_groups):
                self.log(f'lr_group_{i}', pg['lr'])
        opts.step()

        self.log('train_loss', loss)
        self.log('train_acc', acc)
        self.log('throughput', self.total_throughput)

    def validation_step(self, batch, batch_idx):
        loss, acc = self.iteration_step(batch)

        self.log("val_loss", loss, sync_dist=True)
        self.log("val_acc", acc, sync_dist=True)

    def test_step(self, batch, batch_idx):
        loss, acc = self.iteration_step(batch)

        self.log("test_loss", loss)
        self.log("test_acc", acc)
        
    
    def configure_optimizers(self):
        return self.opts.optimizer
    
class REVISEDHUBERTLightning(L.LightningModule):
    """
    very temporary, very ugly
    """
    def __init__(self, cfg: BaseConfig):
        super().__init__()
        self.cfg = cfg
        if cfg._cfg.load_path is None:
            self.hubert = HUBERT(cfg.hypformer_config, cfg.model_config)
            self.model = REVISEDHUBERTForMaskedLM(self.hubert)
        else:
            print(f'Loading model from {cfg._cfg.load_path}')
            self.model = torch.load(MODEL_DICT.get(cfg._cfg.load_path), weights_only=False)
        self.opts = Optimizer(self.model, cfg)
        self.scheduler = build_scheduler(cfg.scheduler_config)
        self.automatic_optimization = False # We use 2 optimizers for euc/hyp params 
        self.criterion = nn.CrossEntropyLoss()

        print(f'Saving model to {cfg.save_folder}/hubert_{SLURM_JOB_ID}/model')
        os.makedirs(f'{self.cfg.save_folder}/hubert_{SLURM_JOB_ID}/model', exist_ok=True)

        self.total_throughput = 0 # Monitor how many tokens our model sees

        # Initialize all weights in self.model (including the HUBERT backbone)
        if cfg.load_path is None:
            self.model.apply(self._init_weights)
        self.step = 0

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.xavier_uniform_(module.weight)
            if getattr(module, "bias", None) is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def iteration_step(self, batch):
        x, mask, labels = batch['input_ids'], batch['attention_mask'], batch['labels']
        if self.training:
            self.total_throughput += batch['input_ids'].view(-1).shape[0] # process (bsz * seq_len) tokens
        logits = self.model(x, mask=mask).transpose(1, 2)
        preds = torch.argmax(logits, dim=1)
        acc = FM.accuracy(preds, 
                          labels, 
                          task='multiclass',
                          num_classes=self.model.hubert.vocab_size,
                          ignore_index=-100)
        loss = self.criterion(logits, labels)
        return loss, acc
    
    def training_step(self, batch, batch_idx):
        self.step += 1
        if self.step % self.cfg.save_interval == 0:
            print(f'DEBUG: Saving model, {self.step=}')
            torch.save(self.model, f'{self.cfg.save_folder}/hubert_{SLURM_JOB_ID}/model/model-step{self.step}.pt')
        loss, acc = self.iteration_step(batch)

        opts = self.opts
        opts.zero_grad()
        self.manual_backward(loss)
        # log gradient norm
        grad_norm = torch.sqrt(
            sum(p.grad.pow(2).sum() for p in self.model.parameters() if p.grad is not None)
        )
        self.log('grad_norm', grad_norm)
        # Clip gradients & adjust LR
        for opt in self.opts.optimizer:
            self.clip_gradients(opt, gradient_clip_val=1.0, gradient_clip_algorithm="norm")
        self.optim_step()

        self.log('train_loss', loss)
        self.log('train_acc', acc)
        self.log('throughput', self.total_throughput)

    def validation_step(self, batch, batch_idx):
        loss, acc = self.iteration_step(batch)

        self.log("val_loss", loss, sync_dist=True)
        self.log("val_acc", acc, sync_dist=True)

    def test_step(self, batch, batch_idx):
        loss, acc = self.iteration_step(batch)

        self.log("test_loss", loss)
        self.log("test_acc", acc)
        
    
    def configure_optimizers(self):
        return self.opts.optimizer
    
    def optim_step(self):
        # Adjust lr
        for opt in self.opts.optimizer:
            for group in opt.param_groups:
                group["lr"] = self.scheduler.get_lr(
                self.cfg.optimizer.lr, self.global_step, self.cfg.max_duration
                )
        self.opts.step()
    
class HFLightning(L.LightningModule):
    """
    very temporary, very ugly
    """
    def __init__(self, model, cfg: BaseConfig):
        super().__init__()
        self.cfg = cfg
        self.model = model
        print(f'DEBUG: Vocab size for resizing: {cfg.model_config.vocab_size}')
        self.model.resize_token_embeddings(cfg.model_config.vocab_size)
        print('DEBUG: Embeddings resized.')

        self.opts = Optimizer(self.model, cfg)
        self.scheduler = build_scheduler(cfg.scheduler_config)
        self.automatic_optimization = False # We use 2 optimizers for euc/hyp params 
        self.criterion = nn.CrossEntropyLoss()
        print('DEBUG: Optimizer and scheduler constructed.')

        print(f'Saving model to {cfg.save_folder}/hubert_{SLURM_JOB_ID}/model')
        os.makedirs(f'{self.cfg.save_folder}/hubert_{SLURM_JOB_ID}/model', exist_ok=True)

        self.total_throughput = 0 # Monitor how many tokens our model sees

        # Initialize all weights in self.model (including the HUBERT backbone)
        if cfg.load_path is None:
            self.model.apply(self._init_weights)
        print('DEBUG: Model initialized.')
        self.step = 0

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.xavier_uniform_(module.weight)
            if getattr(module, "bias", None) is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def iteration_step(self, batch):
        x, mask, labels = batch['input_ids'], batch['attention_mask'], batch['labels']
        if self.training:
            self.total_throughput += batch['input_ids'].view(-1).shape[0] # process (bsz * seq_len) tokens
        logits = self.model(
            input_ids=x,
            attention_mask=mask,
            labels=labels
        ).logits.transpose(1, 2)
        preds = torch.argmax(logits, dim=1)
        acc = FM.accuracy(preds, 
                          labels, 
                          task='multiclass',
                          num_classes=self.cfg.model_config.vocab_size,
                          ignore_index=-100)
        loss = self.criterion(logits, labels)
        return loss, acc
    
    def training_step(self, batch, batch_idx):
        self.step += 1
        if self.step % self.cfg.save_interval == 0:
            print(f'DEBUG: Saving model, {self.step=}')
            torch.save(self.model, f'{self.cfg.save_folder}/hubert_{SLURM_JOB_ID}/model/model-step{self.step}.pt')
        loss, acc = self.iteration_step(batch)

        opts = self.opts
        opts.zero_grad()
        self.manual_backward(loss)
        # log gradient norm
        grad_norm = torch.sqrt(
            sum(p.grad.pow(2).sum() for p in self.model.parameters() if p.grad is not None)
        )
        self.log('grad_norm', grad_norm)
        # Clip gradients & adjust LR
        for opt in self.opts.optimizer:
            self.clip_gradients(opt, gradient_clip_val=1.0, gradient_clip_algorithm="norm")
        self.optim_step()

        self.log('train_loss', loss)
        self.log('train_acc', acc)
        self.log('throughput', self.total_throughput)

    def validation_step(self, batch, batch_idx):
        loss, acc = self.iteration_step(batch)

        self.log("val_loss", loss, sync_dist=True)
        self.log("val_acc", acc, sync_dist=True)

    def test_step(self, batch, batch_idx):
        loss, acc = self.iteration_step(batch)

        self.log("test_loss", loss)
        self.log("test_acc", acc)
        
    
    def configure_optimizers(self):
        return self.opts.optimizer
    
    def optim_step(self):
        # Adjust lr
        for opt in self.opts.optimizer:
            for group in opt.param_groups:
                group["lr"] = self.scheduler.get_lr(
                self.cfg.optimizer.lr, self.global_step, self.cfg.max_duration
                )
        self.opts.step()

class HUBERTEmbeddings(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.word_embeddings = nn.Embedding(cfg.vocab_size, 
                                            cfg.hidden_dim, 
                                            padding_idx=cfg.pad_token_id)
        self.position_embeddings = nn.Embedding(cfg.max_position_embeddings, cfg.hidden_dim)

        self.LayerNorm = nn.LayerNorm(cfg.hidden_dim, eps=1e-12)
        self.dropout = nn.Dropout(cfg.dropout)
        self.register_buffer(
            "position_ids", torch.arange(cfg.max_position_embeddings).expand((1,-1)), persistent=False
        )

    def forward(self, input_ids):
        input_embeds = self.word_embeddings(input_ids) #[bsz, seq_len, dim]

        seq_len = input_embeds.size(1)

        position_ids = self.position_ids[:, :seq_len]
        position_embeddings = self.position_embeddings(position_ids)

        embeddings = input_embeds + position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings