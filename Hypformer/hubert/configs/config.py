from omegaconf import OmegaConf as om
from os import PathLike
from typing import Optional, Union, Tuple
from dataclasses import dataclass
from transformers import AutoTokenizer
from enum import Enum

PathOrStr = Union[str, PathLike]

#+---------- Custom types -----------------------------------------------------+
class StrEnum(str, Enum):
    """
    This is equivalent to Python's :class:`enum.StrEnum` since version 3.11.
    We include this here for compatibility with older version of Python.
    """

    def __str__(self) -> str:
        return self.value

    def __repr__(self) -> str:
        return f"'{str(self)}'"

class PaddingDirection(StrEnum):
    right = "right"
    left = "left"

class SchedulerType(StrEnum):
    cosine_with_warmup = "cosine_with_warmup"

class SchedulerUnits(StrEnum):
    steps = "steps"
    tokens = "tokens"
#+-----------------------------------------------------------------------------+

@dataclass
class WandbConfig:
    project: str
    entity: str
    run_name: str

@dataclass
class HypformerConfig:
    k_in: float
    k_out: float
    decoder_type: str
    add_positional_encoding: bool
    attention_type: str
    power_k: int
    trans_heads_concat: bool

@dataclass
class OptimizerConfig:
    optimizer_type: str
    hyp_optimizer_type: str
    weight_decay: float = 0.01
    hyp_weight_decay: float = 0.01
    lr: float = 1e-4
    hyp_lr: float = 1e-4
    betas: Tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-15
    decay_norm_and_bias: bool = False
    decay_embeddings: bool = False

@dataclass
class SchedulerConfig:
    name: SchedulerType = SchedulerType.cosine_with_warmup
    units: SchedulerUnits = SchedulerUnits.steps
    t_warmup: Union[int, float] = 100
    t_max: Optional[Union[int, float]] = None
    alpha_f: float = 0.1

    grad_clip_warmup_steps: Optional[Union[int, float]] = None
    """
    The warmup period for which the max grad norm (or norm ratio) will be set to its
    warmup value of `max_grad_norm * grad_clip_warmup_factor`.
    """

    grad_clip_warmup_factor: Optional[float] = None
    """
    The ratio of the max allowed gradient norm (or norm ratio) for clipping during the warmup period
    vs after the warmup period.
    """

    warmup_min_lr: Optional[float] = None
    """
    The starting LR during the warmup period. If not set this defaults to 10% of
    the target LR.
    """

@dataclass
class HUBERTConfig:
    hidden_dim: int
    n_heads: int
    n_layers: int
    model_max_len: int
    dropout: float
    max_position_embeddings: int
    eos_token_id: int
    pad_token_id: int
    vocab_size: int 

@dataclass 
class HFModelConfig:
    identifier: str
    model_max_len: int
    eos_token_id: int
    pad_token_id: int
    vocab_size: int

@dataclass
class DataConfig:
    paths: str
    num_workers: int
    prefetch_factor: int
    drop_last: bool
    pin_memory: bool
    persistent_workers: bool
    generate_attention_mask: bool
    timeout: int
    pad_direction: PaddingDirection = PaddingDirection.right
    num_eval_batches: Optional[int] = None # Train & Eval set share same config, but this should only be set for eval

class BaseConfig:
    def __init__(self, yaml_path: PathOrStr, overwrites: Optional[PathOrStr] = None):
        # Want to initialize from yaml, allowing overwrites
        cfg = om.load(yaml_path)
        if overwrites:
            override_cfg = om.load(overwrites)
            cfg = om.merge(cfg, override_cfg)
        self._cfg = cfg
        
        # Then want to assign subconfigs to relevant dataclasses
        self.wandb_config = om.merge(om.structured(WandbConfig), cfg.wandb)
        self.hypformer_config = om.merge(om.structured(HypformerConfig), cfg.hypformer)
        self.optimizer_config = om.merge(om.structured(OptimizerConfig), cfg.optimizer)
        self.scheduler_config = om.merge(om.structured(SchedulerConfig), cfg.scheduler)
        self.data_config = om.merge(om.structured(DataConfig), cfg.data)
        self.eval_config = om.merge(om.structured(DataConfig), cfg.eval)
        self.model_config = self._setup_model_config()

    def _setup_model_config(self):
        if self._cfg.which_model == 'hubert':
            model_config = om.merge(om.structured(HUBERTConfig), self._cfg.hubert)
        elif self._cfg.which_model == 'hf_model':
            model_config = om.merge(om.structured(HFModelConfig), self._cfg.hf_model)
        else:
            raise ValueError("BaseConfig.which_model must be one of ['hubert', 'hf_model']")
        
        # add tokenizer. Cannot add to model_config b/c of OmegaConf value checking, but can add to baseconfig
        tokenizer = AutoTokenizer.from_pretrained(self._cfg.tokenizer.identifier, padding=True, truncation=True)
        tokenizer.model_max_length = model_config.model_max_len
        tokenizer.pad_token_id = model_config.pad_token_id
        if tokenizer.mask_token is None:
            tokenizer.add_special_tokens({'mask_token': '[MASK]'})
        self.tokenizer = tokenizer
        model_config.vocab_size = len(tokenizer)
        return model_config        
    
    def __getattr__(self, name):
        if name in self._cfg:
            return self._cfg[name]
        raise AttributeError(f"Config has no attribute '{name}'")
