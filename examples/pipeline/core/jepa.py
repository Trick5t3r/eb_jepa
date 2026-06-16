"""The two SSL objectives, both built on eb_jepa core primitives. GENERIC — a
track picks one via config and never needs to touch this file.

PredictiveJEPA  — latent prediction + anti-collapse (the JEPA recipe):
    z      = encoder.frames(x)            # online frame latents   [B, F, D]
    z_tgt  = ema_encoder.frames(x)        # momentum target (no grad)
    roll eb_jepa.architectures.RNNPredictor from the last context frame to predict
    the future frame latents; regress onto the EMA targets; eb_jepa.losses.VCLoss
    (VICReg variance/covariance) on the online latents prevents collapse.
    Note: the RNNPredictor is an *action-conditioned* GRU; with no exogenous action
    we feed a learned constant step-token, making it an unconditioned latent roll.
    A track WITH a real control/action channel should pass it instead (TODO hook).

TwoViewVICReg  — invariance + anti-collapse (non-predictive):
    two augmented views -> encoder.represent -> Projector -> eb_jepa.losses.VICRegLoss.

Both expose ``compute_loss(batch) -> (loss, logs)`` so the trainer is agnostic.
"""
import copy

import torch
from torch import nn

from eb_jepa.architectures import Projector
from eb_jepa.losses import VCLoss, VICRegLoss
from examples.pipeline.core.predictors import build_predictor


class PredictiveJEPA(nn.Module):
    def __init__(self, encoder, predictor="rnn", ema=0.996, std_coeff=25.0,
                 cov_coeff=1.0, reg_coeff=1.0, n_context=4, mcfg=None):
        super().__init__()
        self.encoder = encoder
        self.dim, self.n_frames = encoder.out_dim, encoder.n_frames
        self.n_context, self.ema, self.reg_coeff = n_context, ema, reg_coeff
        self.target = copy.deepcopy(encoder)
        for p in self.target.parameters():
            p.requires_grad_(False)
        self.predictor = build_predictor(predictor, self.dim, self.n_frames, mcfg or {})
        self.reg = VCLoss(std_coeff=std_coeff, cov_coeff=cov_coeff)

    @torch.no_grad()
    def update_target(self):
        for pt, po in zip(self.target.parameters(), self.encoder.parameters()):
            pt.mul_(self.ema).add_(po.detach(), alpha=1.0 - self.ema)
        for bt, bo in zip(self.target.buffers(), self.encoder.buffers()):
            bt.copy_(bo)

    def compute_loss(self, x):
        z = self.encoder.frames(x)
        with torch.no_grad():
            z_tgt = self.target.frames(x)
        nc = self.n_context
        z_hat = self.predictor.predict(z, nc)
        pred = nn.functional.mse_loss(z_hat, z_tgt[:, nc:].detach())
        reg, _, rd = self.reg(z.reshape(-1, self.dim))
        return pred + self.reg_coeff * reg, {"pred": pred.item(), "reg": reg.item(), **rd}

    def on_step_end(self):
        self.update_target()


class TwoViewVICReg(nn.Module):
    def __init__(self, encoder, projector_spec=None, std_coeff=25.0, cov_coeff=1.0):
        super().__init__()
        self.encoder = encoder
        spec = projector_spec or f"{encoder.out_dim}-1024-1024"
        self.projector = Projector(spec)
        self.loss_fn = VICRegLoss(std_coeff=std_coeff, cov_coeff=cov_coeff)

    def _embed(self, x):
        return self.projector(self.encoder.represent(x))

    def compute_loss(self, batch):
        v1, v2 = batch
        out = self.loss_fn(self._embed(v1), self._embed(v2))
        return out["loss"], {"inv": out["invariance_loss"].item(),
                             "var": out["var_loss"].item(), "cov": out["cov_loss"].item()}

    def on_step_end(self):
        pass


def build_ssl(method, encoder, cfg):
    """Factory: 'predictive' | 'vicreg'. cfg is the OmegaConf model section."""
    if method == "predictive":
        return PredictiveJEPA(encoder, predictor=cfg.get("predictor", "rnn"),
                              ema=cfg.get("ema", 0.996), std_coeff=cfg.get("std_coeff", 25.0),
                              cov_coeff=cfg.get("cov_coeff", 1.0), reg_coeff=cfg.get("reg_coeff", 1.0),
                              n_context=cfg.get("n_context", encoder.n_frames // 2), mcfg=cfg)
    if method == "vicreg":
        return TwoViewVICReg(encoder, projector_spec=cfg.get("projector", None),
                             std_coeff=cfg.get("std_coeff", 25.0), cov_coeff=cfg.get("cov_coeff", 1.0))
    raise ValueError(f"unknown ssl method {method!r} (expected 'predictive' or 'vicreg')")
