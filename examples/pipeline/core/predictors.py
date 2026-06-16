"""Latent predictors for the predictive JEPA — pluggable via `model.predictor`.

Both implement the same interface, ``predict(z, n_context) -> z_hat_future``:
given the online frame latents ``z`` [B, F, D] and the number of context frames,
return the predicted latents for the future frames [B, F - n_context, D] (regressed
onto the EMA-target latents by PredictiveJEPA).

  * rnn         — autoregressive GRU roll (eb_jepa.architectures.RNNPredictor); the
                  unconditioned step is driven by a learned step-token.
  * transformer — V-JEPA-style: replace future positions with a learned mask token,
                  add positional embeddings, run a TransformerEncoder, read the
                  predictions at the future positions (parallel, not autoregressive).
"""
import torch
from torch import nn

from eb_jepa.architectures import RNNPredictor


class RNNLatentPredictor(nn.Module):
    def __init__(self, dim, action_dim=8):
        super().__init__()
        self.dim = dim
        self.rnn = RNNPredictor(hidden_size=dim, action_dim=action_dim,
                                num_layers=1, final_ln=nn.LayerNorm(dim))
        self.step_token = nn.Parameter(torch.zeros(action_dim))

    def predict(self, z, n_context):
        b = z.shape[0]
        s = z[:, n_context - 1].reshape(b, self.dim, 1, 1, 1)
        a = self.step_token.view(1, -1, 1).expand(b, -1, 1)
        out = []
        for _ in range(z.shape[1] - n_context):
            s = self.rnn(s, a)
            out.append(s.reshape(b, self.dim))
        return torch.stack(out, dim=1)


class TransformerLatentPredictor(nn.Module):
    def __init__(self, dim, n_frames, n_layers=2, n_heads=4, ff=None):
        super().__init__()
        self.dim = dim
        self.mask_token = nn.Parameter(torch.zeros(dim))
        self.pos = nn.Parameter(torch.zeros(n_frames, dim))
        nn.init.normal_(self.pos, std=0.02)
        nn.init.normal_(self.mask_token, std=0.02)
        layer = nn.TransformerEncoderLayer(dim, n_heads, dim_feedforward=ff or 4 * dim,
                                           batch_first=True, activation="gelu", norm_first=True)
        self.tf = nn.TransformerEncoder(layer, num_layers=n_layers)

    def predict(self, z, n_context):
        b, f, d = z.shape
        masks = self.mask_token.view(1, 1, d).expand(b, f - n_context, d)
        seq = torch.cat([z[:, :n_context], masks], dim=1) + self.pos[:f].unsqueeze(0)
        return self.tf(seq)[:, n_context:]


def build_predictor(name, dim, n_frames, mcfg):
    """mcfg: the OmegaConf `model` section (.get works on it)."""
    if name == "rnn":
        return RNNLatentPredictor(dim, action_dim=mcfg.get("action_dim", 8))
    if name == "transformer":
        return TransformerLatentPredictor(dim, n_frames, n_layers=mcfg.get("tf_layers", 2),
                                          n_heads=mcfg.get("tf_heads", 4), ff=mcfg.get("tf_ff", None))
    raise ValueError(f"unknown predictor {name!r} (expected 'rnn' or 'transformer')")
