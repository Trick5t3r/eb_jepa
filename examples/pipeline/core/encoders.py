"""Encoders for the pipeline.

GENERIC (implemented): ``Conv1dEncoder`` — a strided 1D-conv backbone for any
[B, C, T] signal (finance, LTSF, EEG, raw audio). Exposes the two read-outs the
rest of the pipeline relies on:

  * ``represent(x) -> [B, D]``    global time-pooled embedding (probe + VICReg)
  * ``frames(x)    -> [B, F, D]`` short latent SEQUENCE (for the predictive JEPA)

Any encoder used by the pipeline only has to provide ``represent`` (and ``frames``
if the track uses the predictive JEPA), plus an ``out_dim`` attribute and an
``n_frames`` attribute.

TRACK-SPECIFIC (left to students): a 2D/patch encoder for images & PDE fields, a
PointNet for point clouds, a mel-spectrogram CNN for audio. The eb_jepa core
already ships several you can wrap — see the #TODO pointers below.
"""
import torch
from torch import nn


def _conv1d_stack(in_channels, hidden, out_dim, depth, kernel=5):
    widths = [hidden * (2 ** i) for i in range(depth - 1)] + [out_dim]
    blocks, c = [], in_channels
    for w in widths:
        blocks += [nn.Conv1d(c, w, kernel, stride=2, padding=kernel // 2),
                   nn.BatchNorm1d(w), nn.GELU()]
        c = w
    return nn.Sequential(*blocks)


class Conv1dEncoder(nn.Module):
    """Generic 1D backbone with global + frame-sequence read-outs."""

    def __init__(self, in_channels, hidden=64, out_dim=256, depth=4,
                 n_frames=8, kernel=5, final_ln=True):
        super().__init__()
        self.in_channels, self.out_dim, self.n_frames = in_channels, out_dim, n_frames
        self.net = _conv1d_stack(in_channels, hidden, out_dim, depth, kernel)
        self.final_ln = nn.LayerNorm(out_dim) if final_ln else nn.Identity()

    def _pool(self, h):
        return self.final_ln(h.mean(dim=-1))

    def represent(self, x):                      # [B, C, T] -> [B, D]
        return self._pool(self.net(x))

    def frames(self, x):                         # [B, C, T] -> [B, F, D]
        b, c, t = x.shape
        f = self.n_frames
        fl = t // f
        x = x[:, :, : fl * f].reshape(b, c, f, fl).permute(0, 2, 1, 3).reshape(b * f, c, fl)
        return self._pool(self.net(x)).reshape(b, f, self.out_dim)

    def forward(self, x):
        return self.represent(x)


# ---------------------------------------------------------------------------
# TODO (track-specific encoders). Wrap one of the eb_jepa core architectures and
# expose the same `represent()` / `frames()` / `out_dim` / `n_frames` interface:
#
#   * 2D images / PDE fields  -> eb_jepa.architectures.ResNet5 / ImpalaEncoder
#         represent(x: [B, C, H, W]) -> [B, D]
#   * point clouds            -> eb_jepa.architectures.PointNetEncoder
#         represent(x: [B, 3, N]) -> [B, D]   (no meaningful frames(): set n_frames=1)
#   * mel-spectrogram audio   -> a 2D CNN over [B, 1, n_mels, T]
#
# For an encoder without a natural frame axis, return a length-1 sequence in
# frames() (n_frames=1) and use the two-view VICReg objective instead of predictive.
# ---------------------------------------------------------------------------
