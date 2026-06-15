"""A fully-working REFERENCE track on synthetic data (no external files).

Each sample is a multivariate window whose class is encoded in its base frequency;
SSL has real temporal structure to capture, and the probe classifies the frequency
band. Use it to run the whole pipeline end-to-end (smoke test, CI) and as a worked
example of what a real track must provide. Copy tracks/template_track.py to start
your own track.
"""
import numpy as np
import torch

from examples.pipeline.core import transforms
from examples.pipeline.tracks.base import Track
from examples.pipeline.core.registry import register_track

C, T, N_CLASSES = 4, 128, 4
_N = {"train": 4000, "val": 800, "test": 1000}
_SEED = {"train": 1, "val": 2, "test": 3}


def _generate(split):
    """Deterministic synthetic set: [N, C, T] z-scored windows + class labels."""
    rng = np.random.default_rng(_SEED[split])
    n = _N[split]
    y = rng.integers(0, N_CLASSES, size=n)
    freqs = 3.0 + 1.3 * np.arange(N_CLASSES)            # closer base frequencies per class
    t = np.linspace(0, 1, T, dtype=np.float32)
    X = np.empty((n, C, T), dtype=np.float32)
    for i in range(n):
        f = freqs[y[i]]
        for c in range(C):
            phase = rng.uniform(0, 2 * np.pi)
            # strong noise floor so the task is NOT trivially separable by a random encoder
            X[i, c] = np.sin(2 * np.pi * f * (c + 1) * t + phase) + 1.1 * rng.standard_normal(T)
    X = (X - X.mean(axis=2, keepdims=True)) / (X.std(axis=2, keepdims=True) + 1e-6)
    return X.astype(np.float32), y.astype(np.int64)


class _DS(torch.utils.data.Dataset):
    def __init__(self, X, y, mode):
        self.X, self.y, self.mode = X, y, mode   # mode: ssl_single | ssl_two | supervised
        self._rng = np.random.default_rng()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        x = self.X[i]
        if self.mode == "supervised":
            return torch.from_numpy(x.copy()), int(self.y[i])
        self._rng = np.random.default_rng(torch.randint(0, 2**31 - 1, (1,)).item())
        if self.mode == "ssl_single":
            return torch.from_numpy(transforms.default_view(x, self._rng))
        v1 = transforms.default_view(x, self._rng)
        v2 = transforms.default_view(x, self._rng)
        return torch.from_numpy(v1), torch.from_numpy(v2)


@register_track("synthetic")
class SyntheticTrack(Track):
    task_type = "classification"
    n_classes = N_CLASSES
    in_channels = C

    def __init__(self):
        self._cache = {}

    def _xy(self, split):
        if split not in self._cache:
            self._cache[split] = _generate(split)
        return self._cache[split]

    def build_pretrain_dataset(self, cfg, split="train"):
        X, y = self._xy(split)
        mode = "ssl_two" if cfg.model.ssl == "vicreg" else "ssl_single"
        return _DS(X, y, mode)

    def build_eval_dataset(self, cfg, split):
        X, y = self._xy(split)
        return _DS(X, y, "supervised")
