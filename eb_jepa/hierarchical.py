"""Hierarchical (two-level) JEPA for the maze: a COARSE predictor that models
long-horizon abstractions on top of the FINE per-step predictor, so planning can
run at the coarse level and cut the search horizon — enabling A*-free navigation.

The maze impala encoder pools to a 1x1 latent, so a state is a vector z in R^D.
The coarse predictor P_c(z, dir) predicts the latent reached after taking K fine
steps in cardinal direction `dir` (wall-stopping). It is trained by DISTILLING the
frozen fine world-model's K-step constant-direction rollout into a single coarse
jump, plus a coarse temporal-similarity (VICReg-style) term so the coarse latents
stay informative. A coarse value V_c(z, z_goal) (TD on coarse rollouts) then drives
coarse MPC toward the GLOBAL goal — no A* waypoints.

This realises the "modular encoder/predictor/regularizer split as a natural
starting point" for multi-temporal-resolution planning.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# 4 cardinal macro-directions as unit (row, col) steps; scaled by cell_size at use.
CARDINALS = torch.tensor([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])


class SubgoalPredictor(nn.Module):
    """High-level policy that REPLACES A* waypoint generation (feudal/subgoal style).

    Given the current state latent (which encodes the WHOLE maze — the wall mask is
    in the obs image — plus the agent position) and the goal position, predict the
    position of the NEXT waypoint ~N cells along the route to the goal. Trained
    SUPERVISED on A* trajectories (label = the A* position N steps ahead), so at
    eval it proposes waypoints itself and the low-level reacher follows them — A*
    is used only as a training teacher, never at eval.
    """

    def __init__(self, dim, hidden=512):
        super().__init__()
        self.dim = dim
        self.net = nn.Sequential(
            nn.Linear(dim + 2, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, 2),
        )

    def forward(self, z, goal_xy):
        """z: [B,dim,1,1,1] (or [B,dim]); goal_xy: [B,2] (normalized). -> [B,2]."""
        v = z.reshape(z.shape[0], self.dim)
        return self.net(torch.cat([v, goal_xy], dim=-1))


class CoarsePredictor(nn.Module):
    """z [B,D,1,1,1] + cardinal dir -> coarse latent (K fine steps later). MLP over
    the pooled latent vector (maze latent is spatially 1x1)."""

    def __init__(self, dim, hidden=512, n_dirs=4):
        super().__init__()
        self.dim = dim
        self.dir_emb = nn.Embedding(n_dirs, 64)
        self.net = nn.Sequential(
            nn.Linear(dim + 64, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, z, dir_idx):
        """z: [B,D,1,1,1] (or [B,D]); dir_idx: [B] long. Returns same shape as z."""
        shp = z.shape
        v = z.reshape(shp[0], self.dim)
        e = self.dir_emb(dir_idx)
        dz = self.net(torch.cat([v, e], dim=-1))
        out = v + dz                       # residual: predict the change
        return out.reshape(shp)


@torch.no_grad()
def fine_kstep_target(jepa, obs_init, dir_idx, K, cell_size, ctxt_window_time=1):
    """Frozen fine world-model rolled K steps with a CONSTANT cardinal action =
    the coarse-jump target latent. obs_init: [B,C,1,H,W]; dir_idx: [B] long."""
    B = obs_init.shape[0]
    dirs = CARDINALS.to(obs_init.device)[dir_idx]          # [B,2]
    a = (dirs * cell_size).unsqueeze(-1).repeat(1, 1, K)   # [B,2,K]
    pred, _ = jepa.unroll(obs_init, a, nsteps=K, unroll_mode="autoregressive",
                          ctxt_window_time=ctxt_window_time, compute_loss=False,
                          return_all_steps=False)           # [B,D,1+K,1,1]
    return pred[:, :, -1:]                                   # [B,D,1,1,1]


def coarse_similarity_loss(z_coarse, std_coeff=10.0, cov_coeff=1.0, eps=1e-4):
    """VICReg-style anti-collapse on the coarse latents (the coarse-level
    temporal-similarity / regularizer term). z_coarse: [B,D,...]."""
    z = z_coarse.reshape(z_coarse.shape[0], -1).float()
    std = torch.sqrt(z.var(dim=0) + eps)
    std_loss = torch.relu(1.0 - std).mean()
    zc = z - z.mean(dim=0, keepdim=True)
    cov = (zc.T @ zc) / (z.shape[0] - 1)
    cov_loss = (cov.fill_diagonal_(0) ** 2).sum() / z.shape[1]
    return std_coeff * std_loss + cov_coeff * cov_loss


def coarse_rollout(coarse_pred, z0, dir_seq):
    """Roll the coarse predictor over a sequence of macro-directions.
    z0: [B,D,1,1,1]; dir_seq: [B,L] long. Returns list of L coarse latents."""
    z = z0
    outs = []
    for t in range(dir_seq.shape[1]):
        z = coarse_pred(z, dir_seq[:, t])
        outs.append(z)
    return outs


@torch.no_grad()
def plan_coarse(coarse_pred, value_head, z_cur, z_goal, depth, n_samples, device,
                gamma=0.9, beam=8):
    """A*-FREE coarse MPC by BEAM SEARCH over macro-direction sequences. At each of
    `depth` coarse steps, expand each beam by the 4 cardinals, score the resulting
    coarse latent by the learned value toward the GLOBAL goal (best-so-far over the
    rollout, discounted), keep the top-`beam`. Return the first macro-direction of
    the best beam. Deterministic and far less noisy than random sampling. No A*."""
    z = z_cur.reshape(1, *z_cur.shape[1:])                  # [1,D,1,1,1]
    g0 = z_goal.reshape(1, *z_goal.shape[1:])
    beams_z = z                                             # [b,D,1,1,1], b=1
    beams_first = torch.full((1,), -1, device=device, dtype=torch.long)
    beams_score = torch.zeros(1, device=device)            # accumulated value
    for t in range(depth):
        b = beams_z.shape[0]
        zexp = beams_z.repeat_interleave(4, dim=0)          # [b*4,...]
        dirs = torch.arange(4, device=device).repeat(b)     # [b*4]
        first = beams_first.repeat_interleave(4)
        first = torch.where(first < 0, dirs, first)         # set first action at t=0
        znext = coarse_pred(zexp, dirs)
        g = g0.expand(znext.shape[0], -1, -1, -1, -1)
        v = value_head(znext.float(), g.float()).squeeze(1) # [b*4]
        score = beams_score.repeat_interleave(4) + (gamma ** t) * v
        k = min(beam, score.shape[0])
        top = torch.topk(score, k).indices
        beams_z = znext[top]; beams_first = first[top]; beams_score = score[top]
    return int(beams_first[beams_score.argmax()].item())
