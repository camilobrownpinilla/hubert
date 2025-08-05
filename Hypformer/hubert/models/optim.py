import torch
import torch.nn as nn
import torch.nn.init as init

from math import cos, pi
from geoopt import ManifoldParameter
from geoopt.optim.rsgd import RiemannianSGD
from geoopt.optim.radam import RiemannianAdam
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass, replace
from typing import Any, Dict, List, Optional, Tuple

from configs.config import OptimizerConfig, SchedulerConfig, BaseConfig, SchedulerType
from manifolds.layer import HypLinear, HypLayerNorm
from models.hypformer import TransConvLayer


class Optimizer(object):
    """
    Optimizer for Euclidean and Hyperbolic parameters

    Parameters:
        model (nn.Module): The model containing the parameters to optimize.
        optim_config (OptimizerConfig): The config containing optimizer settings.
    """
    # TODO: Add support for adamW and lr scheduler
    def __init__(self, model: nn.Module, cfg: BaseConfig):
        optim_config = cfg.optimizer_config
        euc_optimizer_type = optim_config.optimizer_type
        hyp_optimizer_type = optim_config.hyp_optimizer_type
        euc_lr = optim_config.lr
        hyp_lr = optim_config.hyp_lr
        euc_weight_decay = optim_config.weight_decay
        hyp_weight_decay = optim_config.hyp_weight_decay

        euc_params = [p for _, p in model.named_parameters() if
                      p.requires_grad and not isinstance(p, ManifoldParameter)]
        embedding_params = [p for n, p in model.named_parameters()
                            if p.requires_grad and 'embed' in n.lower()]
        # TODO: Build a function to get hyperbolic parameter groups at some point
        hyp_params = [p for _, p in model.named_parameters() if p.requires_grad and isinstance(p, ManifoldParameter)]

        print(f""">> Total trainable parameters: {sum(p.numel() for p in euc_params) + 
                                                sum(p.numel() for p in embedding_params) +
                                                sum(p.numel() for p in hyp_params)}""")
        print(f""">> Total non-embedding Euclidean parameters: {sum(p.numel() for p in euc_params) - 
                                                              sum(p.numel() for p in embedding_params)}""")
        print(f">> Number of Hyperbolic parameters: {sum(p.numel() for p in hyp_params)}")
        self.optimizer = []  # Optimizers for Euclidean and Hyperbolic parts of the model

        if euc_params:
            optimizer_euc = build_optmizer(model, cfg)
            self.optimizer.append(optimizer_euc)

        if hyp_params:
            if hyp_optimizer_type == 'radam':
                optimizer_hyp = RiemannianAdam(hyp_params, lr=hyp_lr, stabilize=10, weight_decay=hyp_weight_decay)
            elif hyp_optimizer_type == 'rsgd':
                optimizer_hyp = RiemannianSGD(hyp_params, lr=hyp_lr, stabilize=10, weight_decay=hyp_weight_decay)
            else:
                raise NotImplementedError(f"Unknown Hyperbolic optimizer type: {hyp_optimizer_type}")
            self.optimizer.append(optimizer_hyp)

    def step(self):
        """Performs a single optimization step."""
        for optimizer in self.optimizer:
            optimizer.step()

    def zero_grad(self):
        """Sets the gradients of all optimized tensors to zero."""
        for optimizer in self.optimizer:
            optimizer.zero_grad()

@dataclass
class Scheduler(metaclass=ABCMeta):
    # NOTE: these fields are not given default values because otherwise dataclasses complains
    # about how the scheduler subclasses are defined.
    grad_clip_warmup_steps: Optional[int]
    grad_clip_warmup_factor: Optional[float]
    warmup_min_lr: Optional[float]

    @abstractmethod
    def get_lr(self, initial_lr: float, step: int, max_steps: int) -> float:
        raise NotImplementedError

    def _get_max_grad_norm_coeff(
        self, initial_value: Optional[float], step: int, max_steps: int
    ) -> Optional[float]:
        del max_steps  # might need this in the future, but for now I just wanted to match the API of `get_lr()`.
        if initial_value is None:
            return None
        elif (
            self.grad_clip_warmup_steps is None
            or self.grad_clip_warmup_factor is None
            or step > self.grad_clip_warmup_steps
        ):
            return initial_value
        else:
            return self.grad_clip_warmup_factor * initial_value

    def get_max_grad_norm(
        self, initial_max_grad_norm: Optional[float], step: int, max_steps: int
    ) -> Optional[float]:
        return self._get_max_grad_norm_coeff(initial_max_grad_norm, step, max_steps)

    def get_max_grad_norm_ratio(
        self, initial_max_grad_norm_ratio: Optional[float], step: int, max_steps: int
    ) -> Optional[float]:
        return self._get_max_grad_norm_coeff(initial_max_grad_norm_ratio, step, max_steps)

    def _linear_warmup(self, initial_lr: float, step: int, warmup_steps: int = 2000) -> float:
        warmup_min_lr = self.warmup_min_lr if self.warmup_min_lr is not None else initial_lr * 0.10
        assert 0 <= warmup_min_lr < initial_lr
        return warmup_min_lr + (initial_lr - warmup_min_lr) * min(step, warmup_steps) / warmup_steps


@dataclass
class CosWithWarmup(Scheduler):
    warmup_steps: int
    alpha_f: float = 0.1
    t_max: Optional[int] = None

    def get_lr(self, initial_lr: float, step: int, max_steps: int) -> float:
        max_steps = max_steps if self.t_max is None else self.t_max
        eta_min = initial_lr * self.alpha_f
        if step < self.warmup_steps:
            return self._linear_warmup(initial_lr, step, self.warmup_steps)
        elif step >= max_steps:
            return eta_min
        else:
            step = step - self.warmup_steps
            max_steps = max_steps - self.warmup_steps
            return eta_min + (initial_lr - eta_min) * (1 + cos(pi * step / max_steps)) / 2
        

def build_optmizer(model: nn.Module, cfg: BaseConfig) -> torch.optim.Optimizer:
    param_groups = get_param_groups(cfg, model)
    print(f'Constructing optimizer with {len(param_groups)} parameter groups...')
    if cfg.optimizer.optimizer_type == 'adamW':
        return torch.optim.AdamW(
            param_groups,
            lr=cfg.optimizer.lr,
            betas=cfg.optimizer.betas,
            weight_decay=cfg.optimizer.weight_decay,
            eps=cfg.optimizer.eps,
        )
    elif cfg.optimizer.optimizer_type == 'adam':
        return torch.optim.Adam(
            param_groups,
            lr=cfg.optimizer.lr,
            weight_decay=cfg.optimizer.weight_decay
        )
    elif cfg.optimizer.optimizer_type == 'sgd':
        return torch.optim.SGD(
            param_groups,
            lr=cfg.optimizer.lr,
            weight_decay=cfg.optimizer.weight_decay
        )
    else:
        raise NotImplementedError
    
def build_scheduler(cfg: SchedulerConfig) -> Scheduler:
    if cfg.name == SchedulerType.cosine_with_warmup:
        return CosWithWarmup(
            grad_clip_warmup_steps=(
                None if cfg.grad_clip_warmup_steps is None else int(cfg.grad_clip_warmup_steps)
            ),
            grad_clip_warmup_factor=cfg.grad_clip_warmup_factor,
            warmup_steps=int(cfg.t_warmup),
            alpha_f=cfg.alpha_f,
            t_max=None if cfg.t_max is None else int(cfg.t_max),
            warmup_min_lr=cfg.warmup_min_lr,
        )
    else:
        # Don't currently support any other schedulers
        raise NotImplementedError



PARAM_GROUP_FIELDS = ("sharded", "max_grad_norm", "max_grad_norm_ratio", "param_names")


def get_param_groups(cfg: BaseConfig, model: nn.Module) -> List[Dict[str, Any]]:
    """
    Separate parameters into weight decay and non weight decay groups.
    """
    # +----------------------- HACKY FIX --------------------------------------+
    # NOTE: HF models don't agree with this group splitting logic, throwing an 
    #       error that's hard to trace. Since we default to allowing decay for
    #       all parameters, this hacky workaround will suffice. 
    if cfg._cfg.which_model == 'hf_model':
        return [p for _, p in model.named_parameters() if p.requires_grad]
    # +------------------------------------------------------------------------+
    param_groups: List[Dict[str, Any]]
    param_group_defaults = {
        "sharded": False,
        "max_grad_norm": cfg.max_grad_norm,
        "max_grad_norm_ratio": cfg._cfg.get("max_grad_norm_ratio", None),
    }

    # Separate out parameters that we don't want to apply weight decay to, like norms and biases.
    decay = set()
    no_decay = set()
    all_params = {}
    for mn, m in model.named_modules():
        for pn, p in m.named_parameters():
            # NOTE: because named_modules and named_parameters are recursive
            # we will see the same tensors p many many times, but doing it this way
            # allows us to know which parent module any tensor p belongs to...
            if not p.requires_grad:
                continue

            fpn = f"{mn}.{pn}" if mn else pn
            all_params[fpn] = p

            if pn.endswith("bias"):
                if cfg.optimizer_config.decay_norm_and_bias:
                    decay.add(fpn)
                else:
                    no_decay.add(fpn)
            elif pn.endswith("weight") and isinstance(m, (nn.Linear, HypLinear)):
                decay.add(fpn)
            elif pn.endswith("weight") and isinstance(m, (HypLayerNorm, nn.LayerNorm)):
                if cfg.optimizer_config.decay_norm_and_bias:
                    decay.add(fpn)
                else:
                    no_decay.add(fpn)
            elif pn.endswith("weight") and isinstance(m, (nn.Embedding)):
                if cfg.optimizer_config.decay_embeddings:
                    decay.add(fpn)
                else:
                    no_decay.add(fpn)
            elif pn.endswith("scale") and isinstance(m, TransConvLayer):
                #NOTE: Maybe look into allowing scale decay?
                no_decay.add(fpn)

    # Validate that we've considered every parameter
    inter_params = decay & no_decay
    union_params = decay | no_decay
    assert len(inter_params) == 0, f"parameters {inter_params} made it into both decay/no_decay sets!"
    assert (
        len(all_params.keys() - union_params) == 0
    ), f"parameters {all_params.keys() - union_params} were not separated into either decay/no_decay set!"

    # Create the pytorch optimizer groups.
    decay_sorted = sorted(list(decay))
    no_decay_sorted = sorted(list(no_decay))
    param_groups = []
    if len(decay_sorted) > 0:
        param_groups.append(
            {
                "params": [all_params[pn] for pn in decay_sorted],
                "param_names": decay_sorted,
                **param_group_defaults,
            }
        )
    if len(no_decay_sorted) > 0:
        param_groups.append(
            {
                "params": [all_params[pn] for pn in no_decay_sorted],
                "param_names": no_decay_sorted,
                "weight_decay": 0.0,
                **param_group_defaults,
            }
        )

    # Validate fields.
    for group in param_groups:
        for key in PARAM_GROUP_FIELDS:
            assert key in group

    return param_groups