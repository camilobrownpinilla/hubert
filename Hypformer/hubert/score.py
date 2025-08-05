import wandb
import lightning as L
import torch
import os
import torch.nn as nn
import torchmetrics.functional as FM
import math
import json

from argparse import ArgumentParser
from omegaconf import OmegaConf as om

from models.hubert import HUBERT, REVISEDHUBERTForMaskedLM
from configs.config import BaseConfig
from configs.registry import MODEL_DICT
from models.optim import Optimizer, build_scheduler
from data.dataloaders import build_train_dataloader, build_eval_dataloader


# Use lightning to make distributed scoring easier?
SLURM_JOB_ID = os.environ.get("SLURM_JOB_ID")
parser = ArgumentParser(description='Train on masked lanugage modeling.')
parser.add_argument('config', type=str, help='Path to base yaml')
parser.add_argument('--overwrites', type=str, default=None, help='yaml with config overwrites')


class Scorer(L.LightningModule):
    """
    very temporary, very ugly
    """
    def __init__(self, model, cfg: BaseConfig):
        super().__init__()
        self.cfg = cfg
        self.model = model
        # self.opts = Optimizer(self.model, cfg)
        # self.scheduler = build_scheduler(cfg.scheduler_config)
        self.automatic_optimization = False # We use 2 optimizers for euc/hyp params 
        self.criterion = nn.CrossEntropyLoss(reduction='none')

        self.score_path = os.path.join(f'{cfg.save_folder}/hubert_{SLURM_JOB_ID}/score', 'chunk_scores.jsonl')

        print(f'Saving scores to {self.score_path}')
        os.makedirs(f'{cfg.save_folder}/hubert_{SLURM_JOB_ID}/score', exist_ok=True)

        self.total_throughput = 0 # Monitor how many tokens our model sees
        self.step = 0


    def iteration_step(self, batch):
        batch_data = {}
        microbatches = [batch]#self.split_batch(batch)
        batch_scores = []
        for micro_batch in microbatches:
            if 'hubert' in self.cfg.load_path:
                logits = self.model(micro_batch['input_ids'], mask=micro_batch['attention_mask']).transpose(1,2)
            elif 'roberta' in self.cfg.load_path:
                logits = self.model(
                    input_ids=micro_batch['input_ids'],
                    attention_mask=micro_batch['attention_mask'],
                    labels=micro_batch['labels']
                ).logits.transpose(1, 2)
            else:
                raise ValueError("Currently only support hubert and roberta")
            loss = self.criterion(logits, micro_batch['labels']).view(micro_batch['input_ids'].shape[0], -1)
            loss = loss.mean(dim=-1, keepdim=True)
            batch_scores.append(loss)

        batch_scores = torch.concatenate(batch_scores, dim=0)
        batch_data["score"] = batch_scores.detach().cpu()
        return batch_data    
    
    def training_step(self, batch, batch_idx):
        with open(self.score_path, 'a') as json_out:
            batch_data = self.iteration_step(batch)
            for j in range(batch["input_ids"].shape[0]):
                        json_line = \
                            {"score": batch_data["score"][j], 
                            "metadata": batch["metadata"][j]}
                        for k, v in json_line.items():
                            if isinstance(v, torch.Tensor):
                                json_line[k] = v.tolist()
                        json_line = json.dumps(json_line)
                        json_out.write(json_line + "\n")
        return 
    
    def configure_optimizers(self):
        return 

    
    
    def split_batch(self, batch):
        microbatch_size = self.cfg._cfg.microbatch_size
        batch_size = batch["input_ids"].shape[0]
        if batch_size <= microbatch_size:
            return [batch]
        else:
            micro_batches = {}
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    micro_batches[key] = value.split(microbatch_size, dim=0)
                elif isinstance(value, list):
                    micro_batches[key] = [
                        value[microbatch_size * i : microbatch_size * i + microbatch_size]
                        for i in range(math.ceil(batch_size / microbatch_size))
                    ]
                else:
                    raise ValueError(f"unexpected item in batch: '{key}={value}'")
            return [
                {key: value[i] for key, value in micro_batches.items()}  # type: ignore
                for i in range(len(micro_batches["input_ids"]))
            ]
        
if __name__ == '__main__':
    args = parser.parse_args()
    print(f'Loading base config {args.config} and overwrites {args.overwrites}')
    cfg = BaseConfig(args.config, args.overwrites)
    print(f'Loading model from {MODEL_DICT.get(cfg.load_path)}')
    model = torch.load(MODEL_DICT.get(cfg.load_path), weights_only=False)
    model = Scorer(model, cfg)
    loader = build_train_dataloader(cfg)
    trainer = L.Trainer(
        devices='auto',
        strategy='ddp',
        max_steps=cfg.max_duration,
        default_root_dir=f'{cfg.save_folder}/{cfg.wandb_config.run_name}_{SLURM_JOB_ID}/',
        log_every_n_steps=cfg.log_interval,
        limit_val_batches=cfg.eval_config.num_eval_batches,
    )

    print('Beginning scoring...')
    # wandb_logger.watch(model)
    trainer.fit(model, loader)
