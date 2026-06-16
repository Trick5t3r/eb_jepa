"""The Track contract — THIS is the surface students implement.

Everything else in examples/pipeline/core is generic and runs unchanged. To add a
track you subclass ``Track``, fill the methods marked TODO, and register it. The
generic engine then gives you, for free: the predictive-JEPA and VICReg SSL
objectives, the AMP training loop, and the frozen / random / supervised probe
comparison with the right metric.

Minimum to implement (see tracks/template_track.py for a copy-me skeleton):
  * task_type + (n_classes if classification)         — declares the downstream task
  * build_pretrain_dataset(cfg, split)                — unlabeled windows for SSL
  * build_eval_dataset(cfg, split)                    — (x, y) pairs for the probe
  * in_channels (or override build_encoder)           — sizes the default Conv1dEncoder

Optional overrides:
  * build_encoder(cfg)        — swap in a 2D / PointNet / mel encoder (TODO pointers
                                in core/encoders.py)
  * metric(y_true, y_pred)    — custom metric (default chosen by task_type)
  * augment(x, rng)           — custom two-view augmentation (default: generic 1D)
  * extra_baselines(cfg,dev)  — track-specific baselines (DLinear, FNO, ...) -> dict
"""
from examples.pipeline.core import transforms
from examples.pipeline.core.encoders import build_encoder
from examples.pipeline.core.metrics import default_metric


class Track:
    name = "base"
    task_type = "classification"      # classification | regression | forecasting
    n_classes = 2                     # only used for classification
    in_channels = 1                   # sizes the default Conv1dEncoder

    # ---- data (TODO per track) ------------------------------------------------
    def build_pretrain_dataset(self, cfg, split="train"):
        """Return a Dataset of UNLABELED items for SSL.
        predictive JEPA -> yields one tensor x [C, T]
        VICReg          -> yields a tuple (v1, v2) (use self.augment)."""
        raise NotImplementedError("TODO: load unlabeled windows for SSL pretraining")

    def build_eval_dataset(self, cfg, split):
        """Return a Dataset yielding (x [C, T], y) for the downstream probe.
        y is an int (classification), a float (regression), or a [C, H] array
        (forecasting)."""
        raise NotImplementedError("TODO: load (x, y) pairs for the downstream probe")

    # ---- model (sensible default; override for non-1D modalities) -------------
    def build_encoder(self, cfg):
        # conv | transformer, selected by cfg.model.encoder (see core/encoders.py)
        return build_encoder(self.in_channels, cfg)

    # ---- eval (sensible default; override only if custom) ---------------------
    def metric(self, y_true, y_pred):
        return default_metric(self.task_type, self.n_classes)(y_true, y_pred)

    def augment(self, x, rng):
        """Two-view augmentation for VICReg. Default: generic 1D ops. Override for
        SO(3) rotations (point clouds), SpecAugment (mel), masking (fields)."""
        return transforms.default_view(x, rng)

    def extra_baselines(self, cfg, device):
        """Optional track-specific baselines, e.g. {'dlinear': {'mse':..,'mae':..}}.
        TODO (optional): implement DLinear/NLinear for LTSF, FNO/U-Net for fields."""
        return {}
