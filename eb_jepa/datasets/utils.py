from pathlib import Path

import torch
import yaml

from eb_jepa.datasets.two_rooms.utils import update_config_from_yaml
from eb_jepa.datasets.two_rooms.wall_dataset import WallDataset, WallDatasetConfig

DATASETS_DIR = Path(__file__).parent

_DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def load_env_data_config(env_name: str, overrides: dict = None) -> dict:
    """Load base data config for an environment and apply overrides."""
    config_path = DATASETS_DIR / env_name / "data_config.yaml"
    with open(config_path) as f:
        base_config = yaml.safe_load(f)
    if overrides:
        base_config.update(overrides)
    return base_config


def init_data(env_name, cfg_data=None, device=None, **kwargs):
    """Initialize data loaders for the specified environment.

    Supports three pipeline modes via cfg_data["pipeline"]["mode"]:

      - "online" (default): standard DataLoader with on-the-fly CPU generation.
        No extra config needed.

      - "stream": GPU-resident double-buffered pipeline; swaps a small chunk into
        VRAM every N training steps so the GPU never waits for a full epoch of data.
        cfg_data["pipeline"]["backend"] selects the generation backend:
          "cpu" — pool of CPU worker processes (AsyncChunkGenerator)
          "gpu" — on-GPU vectorised generation (GPUWallGenerator)
        Caller MUST invoke manager.warm_up() before iterating the loader,
        and manager.shutdown() at the end of training.

      - "offline": read pre-generated memmaps from disk.
        Pre-generate once with eb_jepa/datasets/two_rooms/gpu_generator.py, then
        point cfg_data["pipeline"]["data_dir"] at the output directory.
        cfg_data["pipeline"]["stream"]=True reads the dataset through the
        double-buffered VRAM stream pipeline (large sequential chunk reads)
        instead of a per-sample random-access DataLoader — far faster on Lustre.
        cfg_data["pipeline"]["shuffle"] (default False) picks sequential vs
        block-shuffle traversal. Requires device; caller must call
        manager.warm_up() and manager.shutdown() (non-None manager).

    Args:
        env_name: Name of the environment (currently only "two_rooms" is supported).
        cfg_data: Configuration overrides for the dataset.
        device: Required when pipeline.mode is "stream".

    Returns:
        Tuple of (train_loader, val_loader, config, pipeline_manager).
        pipeline_manager is None for "online" and "offline" modes.
    """
    if env_name != "two_rooms":
        raise ValueError(f"Unknown env: {env_name}. Only 'two_rooms' is supported.")

    merged_cfg = load_env_data_config(env_name, cfg_data)
    config = update_config_from_yaml(WallDatasetConfig, merged_cfg)

    pipeline_cfg = (cfg_data or {}).get("pipeline") or {}
    mode = str(pipeline_cfg.get("mode", "online")).lower()

    num_workers = merged_cfg.get("num_workers", 0)
    pin_mem = merged_cfg.get("pin_mem", False)
    persistent_workers = merged_cfg.get("persistent_workers", False) and num_workers > 0
    prefetch_factor = merged_cfg.get("prefetch_factor")

    loader_kwargs = dict(
        num_workers=num_workers,
        pin_memory=pin_mem,
        drop_last=True,
        persistent_workers=persistent_workers,
    )
    if num_workers > 0 and prefetch_factor is not None:
        loader_kwargs["prefetch_factor"] = prefetch_factor

    # Validation loader: always a small online generator (never the bottleneck).
    val_dset = WallDataset(config=config)
    val_loader = torch.utils.data.DataLoader(
        val_dset, batch_size=4, shuffle=False, **loader_kwargs
    )

    # ---- stream mode ----
    if mode == "stream":
        if device is None:
            raise ValueError(
                "init_data: device must be provided when pipeline.mode='stream'"
            )
        chunk_size = int(pipeline_cfg.get("chunk_size", merged_cfg["size"]))
        dtype_name = str(pipeline_cfg.get("dtype", "bfloat16")).lower()
        dtype = _DTYPE_MAP.get(dtype_name)
        if dtype is None:
            raise ValueError(
                f"Unknown pipeline.dtype={dtype_name!r}; expected one of {list(_DTYPE_MAP)}"
            )
        backend = str(pipeline_cfg.get("backend", "cpu")).lower()

        if backend == "gpu":
            from eb_jepa.datasets.gpu_precomputed import init_gpu_precomputed_data

            gen_batch_size = pipeline_cfg.get("gen_batch_size")
            gen_batch_size = int(gen_batch_size) if gen_batch_size else None
            loader, manager = init_gpu_precomputed_data(
                env_config_dict=merged_cfg,
                chunk_size=chunk_size,
                epoch_size=config.size,
                batch_size=config.batch_size,
                device=device,
                dtype=dtype,
                gen_batch_size=gen_batch_size,
                drop_last=True,
            )
        elif backend == "cpu":
            from eb_jepa.datasets.precomputed import init_precomputed_data

            num_gen_workers = int(pipeline_cfg.get("num_gen_workers", 16))
            loader, manager = init_precomputed_data(
                env_config_dict=merged_cfg,
                chunk_size=chunk_size,
                epoch_size=config.size,
                batch_size=config.batch_size,
                device=device,
                dtype=dtype,
                num_workers=num_gen_workers,
                drop_last=True,
            )
        else:
            raise ValueError(
                f"Unknown pipeline.backend={backend!r}; expected 'cpu' or 'gpu'"
            )
        return loader, val_loader, config, manager

    # ---- offline mode ----
    if mode == "offline":
        data_dir = pipeline_cfg.get("data_dir")
        if not data_dir:
            raise ValueError(
                "init_data: pipeline.data_dir must be set when pipeline.mode='offline'"
            )

        # pipeline.stream=True: read the pre-generated dataset through the
        # double-buffered VRAM stream pipeline (sequential chunk reads, no
        # per-sample random access). Much faster than the naive DataLoader on
        # Lustre. Iterates the dataset in stored order (no shuffle).
        if bool(pipeline_cfg.get("stream", False)):
            if device is None:
                raise ValueError(
                    "init_data: device must be provided when pipeline.stream=True"
                )
            from eb_jepa.datasets.two_rooms.offline_dataset import (
                init_offline_stream_data,
            )

            chunk_size = int(pipeline_cfg.get("chunk_size", 9600))
            dtype_name = str(pipeline_cfg.get("dtype", "bfloat16")).lower()
            dtype = _DTYPE_MAP.get(dtype_name)
            if dtype is None:
                raise ValueError(
                    f"Unknown pipeline.dtype={dtype_name!r}; "
                    f"expected one of {list(_DTYPE_MAP)}"
                )
            shuffle = bool(pipeline_cfg.get("shuffle", False))
            # read_workers: intra-chunk parallel disk reads (>1 fans the read of
            # each chunk over disjoint sub-ranges — multiple outstanding I/O
            # requests against Lustre, the fix for slow single-threaded reads).
            # prefetch_depth: number of chunks kept reading+staging continuously
            # in VRAM (>1 -> DeepPrefetchManager; chunks load without blocking and
            # each GPU chunk is freed as soon as it is consumed).
            read_workers = int(pipeline_cfg.get("read_workers", 1))
            prefetch_depth = int(pipeline_cfg.get("prefetch_depth", 1))
            # epoch_size = config.size (per-epoch budget, e.g. 100k = 260 steps),
            # NOT the dataset size: one epoch matches the online baseline's budget,
            # and successive epochs advance through the dataset (chunk_id keeps
            # incrementing), so optim.epochs × size samples are consumed in order.
            loader, manager = init_offline_stream_data(
                data_dir=data_dir,
                chunk_size=chunk_size,
                epoch_size=config.size,
                batch_size=config.batch_size,
                device=device,
                dtype=dtype,
                drop_last=True,
                shuffle=shuffle,
                read_workers=read_workers,
                prefetch_depth=prefetch_depth,
            )
            return loader, val_loader, config, manager

        from eb_jepa.datasets.two_rooms.offline_dataset import OfflineWallDataset

        dset = OfflineWallDataset(data_dir)
        config.size = len(dset)
        loader = torch.utils.data.DataLoader(
            dset, batch_size=config.batch_size, shuffle=True, **loader_kwargs
        )
        return loader, val_loader, config, None

    # ---- online mode (default) ----
    if mode != "online":
        raise ValueError(
            f"Unknown pipeline.mode={mode!r}; expected 'online', 'stream', or 'offline'"
        )
    dset = WallDataset(config=config)
    loader = torch.utils.data.DataLoader(
        dset, batch_size=config.batch_size, shuffle=True, **loader_kwargs
    )
    return loader, val_loader, config, None
