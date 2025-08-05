from pathlib import Path
from typing import Any, Dict, List, Optional

from torch.utils.data import DataLoader, DistributedSampler
from .collator import DataCollator
from .memmap_dataset import MemMapDataset
from .iterable_dataset import IterableDataset
from .utils import barrier, get_global_rank, get_world_size
from configs.registry import DATA_DICT

__all__ = [
    "build_train_dataloader",
    "build_eval_dataloader"
]

def build_memmap_dataset(model_config, 
                         data_config, 
                         include_instane_metadata: bool = True) -> MemMapDataset:
    paths: List[str]
    metadata: List[Dict[str, Any]] = []

    if not data_config.paths:
        raise ValueError('Must supply paths in data config')
    paths = data_config.paths
    paths = list(Path(DATA_DICT.get(paths, paths)).glob("*.npy"))
    for path in paths:
        metadata.append({"path": str(path)})

    print(f"Loaded data from: {data_config.paths}")
    return MemMapDataset(
        *paths,
        chunk_size=model_config.model_max_len,
        metadata=metadata,
        include_instance_metadata=include_instane_metadata,
        pad_token_id=model_config.pad_token_id,
        generate_attention_mask=data_config.generate_attention_mask,
    )

def build_train_dataloader(base_config, world_size: Optional[int] = None) -> DataLoader:
    # TODO: make sure MLM collator preserves metadata
    collator = DataCollator.from_train_config(base_config)
    dataset = build_memmap_dataset(base_config.model_config, base_config.data_config, include_instane_metadata=True)
    work_dir = Path(base_config.save_folder) / "train_data"
    if get_global_rank() == 0:
        if work_dir.is_dir() and not base_config.save_overwrite:
            raise FileExistsError(f"Dataset working directory exists. Set save_overwrite: true to overwrite")
        else:
            work_dir.mkdir(exist_ok=True, parents=True)
    sampler = DistributedSampler(
        dataset,
        drop_last=base_config.data_config.drop_last,
        shuffle=True,
        num_replicas=get_world_size(),
        rank=get_global_rank(),
        seed=base_config._cfg.seed,
    )
    barrier()
    return DataLoader(
        dataset,
        batch_size=base_config.batch_size,
        drop_last=base_config.data_config.drop_last,
        sampler=sampler,
        collate_fn=collator,
        num_workers=base_config.data_config.num_workers,
        pin_memory=base_config.data_config.pin_memory,
        prefetch_factor=None if base_config.data_config.num_workers==0 else base_config.data_config.prefetch_factor,
        persistent_workers=False if base_config.data_config.num_workers==0 else base_config.data_config.persistent_workers,
        timeout=base_config.data_config.timeout
    )

def build_eval_dataloader(base_config, shuffle=True) -> DataLoader:
    dataset = build_memmap_dataset(base_config.model_config, base_config.eval_config, include_instane_metadata=True)
    collator = DataCollator.from_train_config(base_config)
    seed = base_config.seed
    if base_config.eval_config.drop_last:
        samples_per_device = len(dataset) // get_world_size()
        batch_size = min(base_config.batch_size, samples_per_device)
        assert batch_size > 0, f'Dataset for {base_config.eval_config.paths} is too small'
    sampler = DistributedSampler(
        dataset,
        drop_last=base_config.eval_config.drop_last,
        shuffle=shuffle,
        num_replicas=get_world_size(),
        rank=get_global_rank(),
        seed=seed,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collator,
        num_workers=base_config.eval_config.num_workers,
        sampler=sampler,
        pin_memory=base_config.eval_config.pin_memory,
        prefetch_factor=None if base_config.eval_config.num_workers==0 else base_config.eval_config.prefetch_factor,
        persistent_workers=False if base_config.eval_config.num_workers==0 else base_config.eval_config.persistent_workers,
        timeout=base_config.eval_config.timeout
    )

