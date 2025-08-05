from hypformer import HypFormer
from manifolds.layer import Optimizer

import torch
import wandb
import torch.nn as nn
import torchmetrics.functional as FM
import lightning as L

from torch.utils.data import Dataset, DataLoader
from transformers import DataCollatorForLanguageModeling, AutoTokenizer
from datasets import Dataset, load_dataset
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger


# Make HypArgs and model config
class Args:
    def __init__(self):
        self.k_in = 1.0
        self.k_out = 1.0
        self.decoder_type = 'hyp'
        self.add_positional_encoding = True
        self.attention_type = 'full'
        self.power_k = 2
        self.trans_heads_concat = False
        self.optimizer_type = 'adam'
        self.hyp_optimizer_type = 'radam'
        self.weight_decay = 0.0
        self.hyp_weight_decay = 0.005
        self.lr = 0.01
        self.hyp_lr = 0.01

class Cfg:
    def __init__(self):
        self.save_path = '/n/netscratch/sham_lab/Everyone/cbrownpinilla/hyperfilter/models/hyperbolic_bert'
        self.tokenizer = 'allenai/eleuther-ai-gpt-neox-20b-pii-special'
        self.batch_size = 32
        self.model_max_len = 512
        self.hidden_dim = 768
        self.n_layers = 4
        self.n_heads = 4
        self.dropout = 0.1
        self.max_duration = 20000

        self.make_tokenizer()
        self.vocab_size = len(self.tokenzier)

    def make_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(self.tokenizer, return_special_tokens_mask=True)
        tokenizer.model_max_length = self.model_max_len
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.add_special_tokens({'mask_token': '[MASK]'})
        self.tokenizer = tokenizer
        return 

# Make data loaders
# For right now might be hardcoded for finefineweb, but might fix later?
def get_loaders(cfg):
    train, test, val = _get_loaders('train', cfg), _get_loaders('test', cfg), _get_loaders('validation', cfg)
    collator = DataCollatorForLanguageModeling(tokenizer=cfg.tokenizer)
    train_loader = DataLoader(train, batch_size=cfg.batch_size, collate_fn=collator)
    test_loader = DataLoader(test, batch_size=cfg.batch_size, collate_fn=collator)
    val_loader = DataLoader(val, batch_size=cfg.batch_size, collate_fn=collator)
    return train_loader, test_loader, val_loader

def _get_loaders(split: str, cfg):
    assert split in ['train', 'test', 'validation'], "Choose one of 'train', 'test', 'validation'"
    key = f'-{split}' if split in ['test', 'validation'] else ''
    ds = load_dataset('m-a-p/FineFineWeb'+key, streaming=True, split='train').remove_columns(['language_score', 'date', 'url', 'file_path', 'dump', 'global_id', 'lang', 'domain', 'token_count'])
    ds = ds.with_format(type='torch')
    ds = ds.map(lambda ex: cfg.tokenizer(ex['text'], 
                                                   truncation=True, 
                                                   padding=True,
                                                   add_special_tokens=True), 
                            batched=True, remove_columns=['text'])
    return ds

# Define models
class MLMHead(nn.Module):
    def __init__(self, model):

        # MLM projects logits to R^{vocab} to choose token from voacab
        super().__init__()
        self.linear = nn.Linear(model.hidden_dim, model.vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        return self.softmax(self.linear(x))

class HUBERT(nn.Module):
    def __init__(self, vocab_size, args, hidden_dim=768, n_layers=12, attn_heads=12, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=0)
        self.layers = HypFormer(
            in_channels=hidden_dim,
            hidden_channels=hidden_dim,
            out_channels=hidden_dim,
            trans_num_layers=n_layers,
            trans_num_heads=attn_heads,
            trans_dropout=dropout,
            args=args
        )

    def forward(self, x, mask=None):
        if mask is not None:
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

# Pytorch LIhgtning for training
class LightningWrapper(L.LightningModule):
    """
    Wrapper around moodels for distributed training via PytorchLightning
    """
    def __init__(self, model, hyp_args):
        super().__init__()
        # We use 2 optimizers for euc/hyp params 
        self.automatic_optimization = False
        self.model = model
        self.args = hyp_args
        self.model.hubert.layers.trans_conv.device = self.device # let Lightning handle devices
        self.criterion = nn.CrossEntropyLoss()
        self.opts = Optimizer(model, hyp_args)
        self.save_hyperparameters(ignore=['model'])
        self.callbacks = [
            ModelCheckpoint(monitor='val_loss',
                            mode='max',
                            filename='{step}-{val_loss:.2f}')
        ]

    def iteration_step(self, batch):
        x, mask, labels = batch['input_ids'], batch['attention_mask'], batch['labels']
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
        opts.step()

        self.log('train_loss', loss)
        self.log('train_acc', acc)

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

if __name__ == '__main__':
    wandb.login()
    hyp_args, model_config = Args(), Cfg()

    train_loader, test_loader, val_loader = get_loaders('m-a-p/FineFineWeb-sample', model_config)
    hubert = HUBERT(len(model_config.tokenizer), 
                    hyp_args, 
                    n_layers=model_config.n_layers, 
                    attn_heads=model_config.n_heads)
    hubertlm = HUBERTForLM(hubert)

    wrapped_model = LightningWrapper(hubertlm, hyp_args)
    wandb_logger = WandbLogger(project='HUBERT', log_model='all')
    trainer = L.Trainer(
        accelerator='cuda',
        devices=2,
        num_nodes=1,
        strategy='ddp',
        max_steps=model_config.max_duration,
        val_check_interval=1000,
        logger=wandb_logger,
        default_root_dir=model_config.save_path,
        callbacks=wrapped_model.callbacks)
    wandb_logger.watch(wrapped_model)
    trainer.fit(wrapped_model, train_loader, val_loader)
