"""COPY ME to start a new track.  ->  cp template_track.py my_track.py

Fill every `# TODO` below, then register your track by adding this line to
tracks/__init__.py:

    from examples.pipeline.tracks import my_track   # noqa: F401

Run it with:
    python -m examples.pipeline.cli.pretrain --track <yourname> --config <cfg.yaml>
    python -m examples.pipeline.cli.evaluate --track <yourname> --ckpt <latest.pth.tar>

A fully-working example to mimic (no external data, no track-specific solution):
    tracks/synthetic_track.py
Stuck on a hook? Re-read tracks/base.py and the core/ docstrings, or ask a mentor —
no reference solution is shipped here.
"""
import numpy as np
import torch

from examples.pipeline.tracks.base import Track
from examples.pipeline.core.registry import register_track


# @register_track("my_track")          # TODO: uncomment + pick a unique name
class MyTrack(Track):
    # TODO: declare the downstream task
    task_type = "classification"        # classification | regression | forecasting
    n_classes = 2                       # classification only
    in_channels = 1                     # number of input channels C (sizes the encoder)

    # ---- DATA ----------------------------------------------------------------
    def build_pretrain_dataset(self, cfg, split="train"):
        """TODO: return a Dataset of UNLABELED windows for SSL.
        - predictive JEPA (cfg.model.ssl == 'predictive') -> __getitem__ returns x [C, T]
        - VICReg          (cfg.model.ssl == 'vicreg')      -> returns (v1, v2)
          (build the two views with self.augment(x, rng))
        Load your files (csv / npz / edf / h5), window them, normalize per channel."""
        raise NotImplementedError("TODO: build_pretrain_dataset")

    def build_eval_dataset(self, cfg, split):
        """TODO: return a Dataset yielding (x [C, T], y) for the probe.
        y is int (classification) / float (regression) / [C, H] array (forecasting).
        Respect the split discipline of your data (temporal, patient-disjoint, ...)."""
        raise NotImplementedError("TODO: build_eval_dataset")

    # ---- MODEL (optional) ----------------------------------------------------
    # The default build_encoder() returns a 1D Conv1dEncoder sized by in_channels.
    # TODO (only for non-1D modalities): override to wrap a 2D CNN / PointNet / mel
    # encoder — see the pointers in core/encoders.py.
    # def build_encoder(self, cfg): ...

    # ---- EVAL (optional) -----------------------------------------------------
    # The default metric is chosen by task_type (core/metrics.py). Override only
    # if you need something custom (per-frame macro-AUROC, VRMSE rollout, ...).
    # def metric(self, y_true, y_pred): ...

    # TODO (optional): a strong task-specific baseline to beat, e.g. DLinear for
    # forecasting or FNO/U-Net for fields. Return {name: {metric: value, ...}}.
    # def extra_baselines(self, cfg, device): ...

    # TODO (optional): custom two-view augmentation (SO(3) for point clouds,
    # SpecAugment for mel, block masking for fields). Default is generic 1D.
    # def augment(self, x, rng): ...
