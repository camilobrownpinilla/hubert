import wandb
import lightning as L
import torch
import os

from argparse import ArgumentParser
from omegaconf import OmegaConf as om
from lightning.pytorch.loggers import WandbLogger
from data.utils import get_local_rank
from transformers import AutoModelForMaskedLM, AutoConfig
from lightning.pytorch.callbacks import ModelCheckpoint

from models.hubert import REVISEDHUBERTLightning, HFLightning
from configs.config import BaseConfig
from configs.registry import MODEL_DICT
from data.dataloaders import build_train_dataloader, build_eval_dataloader

parser = ArgumentParser(description='Train on masked lanugage modeling.')
parser.add_argument('config', type=str, help='Path to base yaml')
parser.add_argument('--overwrites', type=str, default=None, help='yaml with config overwrites')


if __name__ == '__main__':
    slurm_task_id = os.environ.get("SLURM_ARRAY_JOB_ID")
    print(f'DEBUG: Slurm job id: {slurm_task_id}')
    args = parser.parse_args()
    print(f'Loading base config {args.config} and overwrites {args.overwrites}')
    cfg = BaseConfig(args.config, args.overwrites)
    wandb.login()
    wandb_logger = WandbLogger(project=cfg.wandb_config.project,
                               entity=cfg.wandb_config.entity,
                               name=cfg.wandb_config.run_name, 
                               log_model='all')
    # TODO: Only print for rank 0, checkout why overwrites not happening, fix gradient clipping.
    print(f'Initializing training with config:\n{om.to_yaml(cfg._cfg)}')
    if cfg._cfg.which_model == 'hubert':
        # Loading logic handled in model init
        model = REVISEDHUBERTLightning(cfg)
    else:
        if cfg.load_path is None:
            hf_config = AutoConfig.from_pretrained(cfg.model_config.identifier)
            model = AutoModelForMaskedLM.from_config(hf_config)
            model = HFLightning(model, cfg)
        else:
            model = torch.load(MODEL_DICT.get(cfg._cfg.load_path), weights_only=False)
            model = HFLightning(model, cfg)
            
    train_loader = build_train_dataloader(cfg)
    print('DEBUG: Loaded train loader')
    eval_loader = build_eval_dataloader(cfg)
    print('DEBUG: Loaded eval loader')

    trainer = L.Trainer(
        devices='auto',
        strategy='ddp' if torch.cuda.is_available() else 'auto',
        max_steps=cfg.max_duration,
        val_check_interval=cfg.validation_interval,
        default_root_dir=f'{cfg.save_folder}/{cfg.wandb_config.run_name}_{slurm_task_id}/',
        log_every_n_steps=cfg.log_interval,
        logger=wandb_logger,
        limit_val_batches=cfg.eval_config.num_eval_batches,
    )

    print('Beginning training...')
    # wandb_logger.watch(model)
    trainer.fit(model, train_loader, eval_loader)
    trainer.save_checkpoint(filepath=f"{cfg.save_folder}/{cfg.wandb_config.run_name}_{slurm_task_id}/final.ckpt")
