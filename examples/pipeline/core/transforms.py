"""Generic augmentations for the two-view (VICReg) SSL objective.

These are MODALITY-AGNOSTIC operations on a [C, T] array (channels x time/length),
which covers every time-series-like track (finance, LTSF, EEG, raw audio). A track
that needs different views (SO(3) rotations for point clouds, SpecAugment for mel
spectrograms, masking for fields) overrides ``Track.augment`` — see tracks/base.py.

All functions take and return a float32 numpy array of shape [C, T].
"""
import numpy as np


def scale_jitter(x, rng, amount=0.1):
    if amount <= 0:
        return x
    s = 1.0 + rng.uniform(-amount, amount, size=(x.shape[0], 1)).astype(np.float32)
    return x * s


def add_noise(x, rng, std=0.05):
    return x + rng.normal(0, std, size=x.shape).astype(np.float32) if std > 0 else x


def channel_dropout(x, rng, p=0.1):
    if p <= 0:
        return x
    keep = (rng.random(x.shape[0]) > p).astype(np.float32)
    return x * keep[:, None]


def time_mask(x, rng, frac=0.2):
    if frac <= 0:
        return x
    x = x.copy()
    mlen = int(rng.uniform(0, frac) * x.shape[1])
    if mlen > 0:
        s = int(rng.integers(0, x.shape[1] - mlen + 1))
        x[:, s:s + mlen] = 0.0
    return x


def default_view(x, rng, noise_std=0.05, scale=0.1, chan_drop=0.1, t_mask=0.2):
    """Compose the four generic augmentations into one stochastic view."""
    x = scale_jitter(x.copy(), rng, scale)
    x = add_noise(x, rng, noise_std)
    x = channel_dropout(x, rng, chan_drop)
    x = time_mask(x, rng, t_mask)
    return x
