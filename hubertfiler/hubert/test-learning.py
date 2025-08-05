import torch
import lightning as L
import torchmetrics.functional as FM
import torch.nn as nn

from manifolds.layer import Optimizer
from utils import get_toy_loaders, get_loaders
from lightning.pytorch.callbacks import ModelCheckpoint
from transformers import AutoTokenizer, AutoConfig, AutoModelForMaskedLM
from lightning.pytorch.loggers import WandbLogger
from models.hubert import HUBERTLightning, REVISEDHUBERTLightning


class BERTLightning(L.LightningModule):
    """
    Wrapper around Hubert models for distributed training via Pytorch Lightning
    """
    def __init__(self, model, model_cfg, hyp_args):
        super().__init__()
        self.args = hyp_args
        self.cfg = model_cfg

        self.model = model
        self.opts = Optimizer(self.model, self.args)
        self.automatic_optimization = False # We use 2 optimizers for euc/hyp params 
        self.criterion = nn.CrossEntropyLoss()

        self.save_hyperparameters(ignore=['model'])
        self.callbacks = [
            ModelCheckpoint(dirpath='/n/netscratch/sham_lab/Everyone/cbrownpinilla/models/hubert',
                            monitor='val_acc',
                            mode='max',
                            filename='hubert-{step}-{val_loss:.2f}',
                            every_n_train_steps=model_cfg.validation_interval,
                            save_last=True)
        ]
        self.total_throughput = 0 # Monitor how many tokens our model sees


    def iteration_step(self, batch):
        x, mask, labels = batch['input_ids'], batch['attention_mask'], batch['labels']
        btch = {'input_ids': x, 'attention_mask': mask, 'labels': labels}
        if self.training:
            self.total_throughput += batch['input_ids'].view(-1).shape[0] # process (bsz * seq_len) tokens
        out = self.model(**btch)#.logits.transpose(1, 2)
        logits, hf_loss = out.logits.transpose(1, 2), out.loss
        preds = torch.argmax(logits, dim=1)
        acc = FM.accuracy(preds, 
                          labels, 
                          task='multiclass',
                          num_classes=self.cfg.vocab_size,
                          ignore_index=-100)
        loss = self.criterion(logits, labels)
        return loss, acc, hf_loss
    
    def training_step(self, batch, batch_idx):
        loss, acc, hf_loss = self.iteration_step(batch)

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
        self.log('hf_loss', hf_loss) # loss as calculated by HF model as sanity check

    def validation_step(self, batch, batch_idx):
        loss, acc, hf_loss = self.iteration_step(batch)

        self.log("val_loss", loss, sync_dist=True)
        self.log("val_acc", acc, sync_dist=True)
        self.log("hf_loss", hf_loss)

    def test_step(self, batch, batch_idx):
        loss, acc, hf_loss = self.iteration_step(batch)

        self.log("test_loss", loss)
        self.log("test_acc", acc)
        self.log("hf_loss", hf_loss)
        
    
    def configure_optimizers(self):
        return self.opts.optimizer
    
class Args:
    def __init__(self):
        self.k_in = .5
        self.k_out = .5
        self.decoder_type = 'euc'
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu' # If using HUBERTLightning, will be overwritten
        self.add_positional_encoding = True
        self.attention_type = 'full'
        self.power_k = 2
        self.trans_heads_concat = False
        self.optimizer_type = 'adam'
        self.hyp_optimizer_type = 'radam'
        self.weight_decay = 0.0
        self.hyp_weight_decay = 0.005
        self.lr = 1e-4
        self.hyp_lr = 1e-4

class HubertCfg:
    def __init__(self):
        self.save_path = '/n/netscratch/sham_lab/Everyone/cbrownpinilla/hyperfilter/models/hyperbolic_bert'
        self.tokenizer = 'allenai/eleuther-ai-gpt-neox-20b-pii-special'
        self.batch_size = 32
        self.model_max_len = 512
        self.hidden_dim = 768
        self.n_layers = 6
        self.n_heads = 4
        self.dropout = 0.1
        self.max_duration = 10000
        self.validation_interval = 100
        self.log_interval = 10
        self.max_position_embeddings = 512

        self.make_tokenizer()
        self.vocab_size = len(self.tokenizer)
        self.padding_idx = 0

    def make_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(self.tokenizer, return_special_tokens_mask=True)
        tokenizer.model_max_length = self.model_max_len
        tokenizer.pad_token = tokenizer.eos_token
        if tokenizer.mask_token is None:
            tokenizer.add_special_tokens({'mask_token': '[MASK]'})
        self.tokenizer = tokenizer
        return 
    
class RobertaCfg:
    def __init__(self):
        self.save_path = '/n/netscratch/sham_lab/Everyone/cbrownpinilla/hyperfilter/models/hyperbolic_bert'
        self.tokenizer = 'distilbert/distilroberta-base'
        self.batch_size = 32
        self.model_max_len = 512
        self.hidden_dim = 768
        self.n_layers = 1
        self.n_heads = 1
        self.dropout = 0.1
        self.max_duration = 100
        self.validation_interval = 10

        self.make_tokenizer()
        self.vocab_size = len(self.tokenizer)
        self.padding_idx = self.tokenizer.pad_token_id
        self.log_interval = 10

    def make_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(self.tokenizer, return_special_tokens_mask=True)
        tokenizer.model_max_length = self.model_max_len
        tokenizer.pad_token = tokenizer.eos_token
        if tokenizer.mask_token is None:
            tokenizer.add_special_tokens({'mask_token': '[MASK]'})
        self.tokenizer = tokenizer
        return 
    
if __name__ == '__main__':
    
    # Uncomment this to test pretrained roBERTa -------------------------------+
    # cfg, args = RobertaCfg(), Args()
    # # Load pre-trained weights instead of initializing from config to test the training loop
    # bert = AutoModelForMaskedLM.from_pretrained('distilbert/distilroberta-base')
    # model = BERTLightning(bert, cfg, args)

    # Uncomment this to test roBERTa from scratch -----------------------------+
    # cfg, args = RobertaCfg(), Args()
    # bert_config = AutoConfig.from_pretrained('distilbert/distilroberta-base')
    # bert = AutoModelForMaskedLM.from_config(bert_config)
    # model = BERTLightning(bert, cfg, args)

    # Uncomment this to test HUBERT -------------------------------------------+
    # cfg, args = HubertCfg(), Args()
    # model = HUBERTLightning(cfg, args)

    # Uncomment this to test HUBERT with revised language head ----------------+
    cfg, args = HubertCfg(), Args()
    model = REVISEDHUBERTLightning(cfg, args)

    wandb_logger = WandbLogger(project='HUBERT', log_model='all', name='Overfit-HUBERT-l6-h4-newembeds-gelu-fullattn-k=.5-bsz=32', entity='harvardml')
    train_loader, test_loader, val_loader = get_toy_loaders(cfg)

    trainer = L.Trainer(
    devices='auto',
    num_nodes=1,
    strategy='ddp_find_unused_parameters_true',
    max_steps=cfg.max_duration,
    val_check_interval=cfg.validation_interval,
    logger=wandb_logger,
    default_root_dir=cfg.save_path,
    callbacks=model.callbacks,
    log_every_n_steps=cfg.log_interval
    )
    wandb_logger.watch(model)
    trainer.fit(model, train_loader, val_loader)
    trainer.test(model, test_loader)
