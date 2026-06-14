"""Train the COARSE level of a two-level (hierarchical) maze JEPA.

The FINE world model + encoder + position probe + fine value are loaded FROZEN
from a checkpoint trained WITH A* data (e.g. exp_value). Here we add:
  - a CoarsePredictor distilling K fine steps (constant direction) into one macro
    jump, with a coarse temporal-similarity (VICReg) term;
  - a coarse value V_c(z, z_goal) trained by TD(0) on coarse rollouts (hindsight
    goals, reward from the probe position) — so coarse MPC can later plan toward
    the GLOBAL goal WITHOUT A* waypoints (eval_hierarchical.py).

Run: python -m examples.ac_video_jepa.main_hierarchical <fine_ckpt> <out_dir> [K=4]
"""
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.optim import AdamW

from eb_jepa.architectures import ImpalaEncoder, RNNPredictor
from eb_jepa.jepa import JEPA
from eb_jepa.losses import SquareLossSeq
from eb_jepa.state_decoder import GoalValueHead, MLPXYHead
from eb_jepa.training_utils import load_checkpoint
from eb_jepa.datasets.utils import init_data
from eb_jepa.hierarchical import (CARDINALS, CoarsePredictor, coarse_similarity_loss,
                                  fine_kstep_target)


class _DummyReg(nn.Module):
    def forward(self, state, actions):
        z = torch.tensor(0.0, device=state.device)
        return z, z, {}


def build_fine(cfg, data_config, device):
    enc = ImpalaEncoder(width=1, stack_sizes=(16, cfg.model.henc, cfg.model.dstc),
                        num_blocks=2, dropout_rate=None, layer_norm=False,
                        input_channels=cfg.model.dobs, final_ln=True, mlp_output_dim=512,
                        input_shape=(cfg.model.dobs, data_config.img_size, data_config.img_size))
    test_out = enc(torch.rand(1, cfg.model.dobs, 1, data_config.img_size, data_config.img_size))
    f = test_out.shape[1]
    pred = RNNPredictor(hidden_size=enc.mlp_output_dim, final_ln=enc.final_ln)
    jepa = JEPA(enc, nn.Identity(), pred, _DummyReg(), SquareLossSeq()).to(device)
    return jepa, f


def main():
    fine_ckpt, out_dir = sys.argv[1], sys.argv[2]
    K = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    epochs_override = int(sys.argv[4]) if len(sys.argv) > 4 else None
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = OmegaConf.load(Path(fine_ckpt).parent / "config.yaml")
    # LONG windows so the coarse value sees a genuine long-horizon traversal:
    # subsampled at stride K it covers the whole maze (the fix for A*-free).
    cfg.data.sample_length = int(cfg.data.get("n_steps", 91)) - 1  # ~90 frames
    cfg.data.batch_size = 96  # 90-frame encodes are heavy -> smaller batch

    loader, _, data_config, data_pipeline = init_data(
        env_name=cfg.data.env_name,
        cfg_data=OmegaConf.to_container(cfg.data, resolve=True), device=device)
    cell_size = float(data_config.cell_size)
    normalizer = loader.dataset.normalizer

    jepa, f = build_fine(cfg, data_config, device)
    xy_head = MLPXYHead(input_shape=f, normalizer=normalizer).to(device)
    fine_value = GoalValueHead(f).to(device)
    info = load_checkpoint(Path(fine_ckpt), jepa, optimizer=None, scheduler=None,
                           device=device, strict=False)
    if "xy_head_state_dict" in info:
        xy_head.load_state_dict(info["xy_head_state_dict"])
    if "value_head_state_dict" in info:
        fine_value.load_state_dict(info["value_head_state_dict"])
    for m in (jepa, xy_head, fine_value):
        m.eval()
        for p in m.parameters():
            p.requires_grad_(False)
    print(f"[hier] loaded fine model (f={f}) | K={K} cell_size={cell_size}", flush=True)

    coarse = CoarsePredictor(f).to(device)
    cvalue = GoalValueHead(f).to(device)
    ctarget = GoalValueHead(f).to(device); ctarget.load_state_dict(cvalue.state_dict())
    for p in ctarget.parameters():
        p.requires_grad_(False)
    opt = AdamW(coarse.parameters(), lr=1e-3, weight_decay=1e-5)
    vopt = AdamW(cvalue.parameters(), lr=1e-3, weight_decay=1e-5)

    def probe_pos(z):  # [B,f,1,1,1] -> [B,2] pixel
        xy = xy_head(z.float()).permute(0, 2, 1)[:, 0]  # [B,2] normalized
        return normalizer.unnormalize_location(xy)

    gamma = 0.9
    epochs = epochs_override or int(cfg.optim.get("coarse_epochs", 4))
    for epoch in range(epochs):
        t0 = time.time(); nd = 0; dl_tot = 0.0; vl_tot = 0.0
        for x, a, loc, _, _ in loader:
            x = x.to(device, non_blocking=True)
            obs0 = x[:, :, :1]
            with torch.no_grad():
                z0 = jepa.encode(obs0)                       # [B,f,1,1,1]
            # ---- distill coarse predictor: P_c(z0,d) ~ fine K-step (constant d) ----
            preds = []
            distill = 0.0
            for d in range(4):
                di = torch.full((z0.shape[0],), d, device=device, dtype=torch.long)
                tgt = fine_kstep_target(jepa, obs0, di, K, cell_size)  # [B,f,1,1,1]
                p = coarse(z0, di)
                distill = distill + F.mse_loss(p, tgt)
                preds.append(p)
            sim = coarse_similarity_loss(torch.cat(preds, 0))
            closs = distill + 0.1 * sim
            opt.zero_grad(); closs.backward(); opt.step()

            # ---- coarse value: TD(0) on the REAL A* trajectory subsampled at
            # stride K (hindsight goal = trajectory end). Reward is position-based
            # (probe) so the goal-padded tail frames all count as "reached".
            with torch.no_grad():
                z_all = jepa.encode(x)                 # [B,f,T,1,1]
                z_c = z_all[:, :, ::K].contiguous()    # [B,f,Tc,1,1] coarse states
                Tc = z_c.shape[2]
                g = z_c[:, :, -1:]                     # hindsight goal latent
                pos = [probe_pos(z_c[:, :, t:t + 1]) for t in range(Tc)]  # [B,2] each
                pg = pos[-1]
                reached = [(torch.norm(pos[t] - pg, dim=-1) < cell_size).float()
                           for t in range(Tc)]         # [B] each
                vnext = [ctarget(z_c[:, :, t + 1:t + 2].float(), g.float()).squeeze(1)
                         for t in range(Tc - 1)]
            vloss = 0.0
            for t in range(Tc - 1):
                v_pred = cvalue(z_c[:, :, t:t + 1].float(), g.float()).squeeze(1)
                done = reached[t + 1]                  # [B]
                y = done + gamma * (1.0 - done) * vnext[t]
                vloss = vloss + F.mse_loss(v_pred, y.detach())
            vloss = vloss / (Tc - 1)
            vopt.zero_grad(); vloss.backward(); vopt.step()
            with torch.no_grad():
                for pt, p in zip(ctarget.parameters(), cvalue.parameters()):
                    pt.mul_(0.99).add_(p.detach(), alpha=0.01)
            dl_tot += float(distill); vl_tot += float(vloss); nd += 1
        print(f"[hier] epoch {epoch} {time.time()-t0:.0f}s distill={dl_tot/max(nd,1):.4f} "
              f"vloss={vl_tot/max(nd,1):.4f}", flush=True)
        torch.save({"coarse": coarse.state_dict(), "cvalue": cvalue.state_dict(),
                    "K": K, "f": f}, os.path.join(out_dir, "coarse.pth.tar"))
    if data_pipeline is not None:
        data_pipeline.shutdown()
    print(f"[hier] DONE -> {out_dir}/coarse.pth.tar", flush=True)


if __name__ == "__main__":
    main()
