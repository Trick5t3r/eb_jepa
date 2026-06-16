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


class TransformerEncoder1d(nn.Module):
    """Token-Transformer backbone for [B, C, T]. A cheap conv stem tokenizes the
    signal and adaptive-pools it to exactly ``n_frames`` tokens (so it works for any
    T), then a TransformerEncoder mixes them. Same read-out interface as Conv1dEncoder:
      * frames(x)    -> [B, n_frames, D]  (the token sequence)
      * represent(x) -> [B, D]            (mean over tokens)
    """

    def __init__(self, in_channels, out_dim=256, n_frames=8, n_layers=3, n_heads=4,
                 ff=None, stem_kernel=7, stem_stride=4, final_ln=True):
        super().__init__()
        self.in_channels, self.out_dim, self.n_frames = in_channels, out_dim, n_frames
        self.stem = nn.Conv1d(in_channels, out_dim, stem_kernel, stride=stem_stride,
                              padding=stem_kernel // 2)
        self.pool = nn.AdaptiveAvgPool1d(n_frames)
        self.pos = nn.Parameter(torch.zeros(n_frames, out_dim))
        nn.init.normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(out_dim, n_heads, dim_feedforward=ff or 4 * out_dim,
                                           batch_first=True, activation="gelu", norm_first=True)
        self.tf = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.final_ln = nn.LayerNorm(out_dim) if final_ln else nn.Identity()

    def frames(self, x):                         # [B, C, T] -> [B, n_frames, D]
        h = self.pool(self.stem(x)).transpose(1, 2)   # [B, n_frames, D]
        return self.tf(h + self.pos.unsqueeze(0))

    def represent(self, x):                      # [B, C, T] -> [B, D]
        return self.final_ln(self.frames(x).mean(dim=1))

    def forward(self, x):
        return self.represent(x)


def build_encoder(in_channels, cfg):
    """Encoder factory selected by `model.encoder` (conv | transformer)."""
    m = cfg.model
    kind = m.get("encoder", "conv")
    if kind == "conv":
        return Conv1dEncoder(in_channels, hidden=m.get("hidden", 64), out_dim=m.get("out_dim", 256),
                             depth=m.get("depth", 4), n_frames=m.get("n_frames", 8))
    if kind == "transformer":
        return TransformerEncoder1d(in_channels, out_dim=m.get("out_dim", 256),
                                    n_frames=m.get("n_frames", 8), n_layers=m.get("tf_layers", 3),
                                    n_heads=m.get("tf_heads", 4), ff=m.get("tf_ff", None))
    raise ValueError(f"unknown encoder {kind!r} (expected 'conv' or 'transformer')")


# ---------------------------------------------------------------------------
# TODO (other modalities). Wrap an eb_jepa core architecture behind the same
# represent()/frames()/out_dim/n_frames interface and return it from your
# Track.build_encoder override:
#   * 2D images / PDE fields  -> eb_jepa.architectures.ResNet5 / ImpalaEncoder
#   * point clouds            -> a PointNet (set n_frames=1, use the VICReg objective)
#   * mel-spectrogram audio   -> a 2D CNN over [B, 1, n_mels, T]
# For an encoder with no natural frame axis, return a length-1 sequence in frames().
# ---------------------------------------------------------------------------
