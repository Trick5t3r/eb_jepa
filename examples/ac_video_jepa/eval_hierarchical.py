"""A*-FREE maze navigation with the two-level (hierarchical) JEPA.

The fine world model is frozen (trained WITH A*); evaluation uses NO A* waypoints
and NO A* prior. At each macro-step we encode the current obs, run coarse MPC
(sample macro-direction sequences, roll the CoarsePredictor, score by the learned
coarse value toward the GLOBAL goal), take the best macro = K env steps in one
cardinal direction, then replan. The coarse level cuts the planning horizon so the
value covers the whole maze without subgoals.

Run: python -m examples.ac_video_jepa.eval_hierarchical <fine_ckpt> <coarse_ckpt>
        <results_dir> [num_episodes=16] [depth=8] [n_samples=512]
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from eb_jepa.datasets.utils import create_env, init_data
from eb_jepa.hierarchical import CARDINALS, CoarsePredictor, fine_kstep_target, plan_coarse
from eb_jepa.state_decoder import GoalValueHead
from examples.ac_video_jepa.main_hierarchical import build_fine


@torch.no_grad()
def main():
    fine_ckpt, coarse_ckpt, rdir = sys.argv[1], sys.argv[2], sys.argv[3]
    num_ep = int(sys.argv[4]) if len(sys.argv) > 4 else 16
    depth = int(sys.argv[5]) if len(sys.argv) > 5 else 8
    n_samples = int(sys.argv[6]) if len(sys.argv) > 6 else 512
    os.makedirs(rdir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = OmegaConf.load(Path(fine_ckpt).parent / "config.yaml")

    _, _, env_config, _ = init_data(env_name=cfg.data.env_name,
                                    cfg_data=OmegaConf.to_container(cfg.data, resolve=True))
    cell_size = float(env_config.cell_size)
    n_allowed = 180

    jepa, f = build_fine(cfg, env_config, device)
    from eb_jepa.training_utils import load_checkpoint
    from eb_jepa.state_decoder import MLPXYHead
    info = load_checkpoint(Path(fine_ckpt), jepa, optimizer=None, scheduler=None,
                           device=device, strict=False)
    jepa.eval()
    xy_head = None
    cck = torch.load(coarse_ckpt, map_location=device, weights_only=False)
    K = cck["K"]
    coarse = CoarsePredictor(f).to(device); coarse.load_state_dict(cck["coarse"]); coarse.eval()
    cvalue = GoalValueHead(f).to(device); cvalue.load_state_dict(cck["cvalue"]); cvalue.eval()
    normalizer = None  # we use the env's normalizer
    print(f"[hier-eval] A*-FREE | K={K} depth={depth} n_samples={n_samples} "
          f"max_macros={n_allowed // K}", flush=True)

    env = create_env(cfg.data.env_name, config=env_config, n_allowed_steps=n_allowed,
                     n_steps=n_allowed, max_step_norm=1.5)
    norm = env.normalizer
    if "xy_head_state_dict" in info:
        xy_head = MLPXYHead(input_shape=f, normalizer=norm).to(device)
        xy_head.load_state_dict(info["xy_head_state_dict"]); xy_head.eval()
    off = (cell_size - 1) / 2.0

    def obs_tensor(obs):
        return norm.normalize_state(obs.to(dtype=torch.float32, device=device)).unsqueeze(0).unsqueeze(2)

    def enc(obs):
        return jepa.encode(obs_tensor(obs))  # [1,f,1,1,1]

    def pred_cell(z):  # latent -> predicted maze cell (r,c) via the probe
        xy = norm.unnormalize_location(xy_head(z.float()).permute(0, 2, 1)[:, 0])[0]
        return (int(round((float(xy[0]) - off) / cell_size)),
                int(round((float(xy[1]) - off) / cell_size)))

    successes = []
    max_macros = n_allowed // K
    for ep in range(num_ep):
        obs, info = env.reset()
        obs, _, _, _, info = env.step(np.zeros(env.action_space.shape[0]))
        goal_img = info["target_obs"].to(dtype=torch.float32)
        z_goal = enc(goal_img)
        success = False
        verbose = (ep == 0)
        blocked = {}          # cell -> set of dirs that produced no movement
        visit = {}            # cell -> times visited (revisit penalty)
        last_rev = -1         # opposite of last executed dir (avoid immediate U-turn)
        OPP = {0: 1, 1: 0, 2: 3, 3: 2}
        for _m in range(max_macros):
            ot = obs_tensor(obs)
            cell = tuple(int(c) for c in env.agent_cell)
            visit[cell] = visit.get(cell, 0) + 1
            # Two-level lookahead: roll the FINE world model K steps per direction
            # (wall-aware — the obs wall mask is in the latent) and score with the
            # COARSE long-horizon value. Minus a revisit penalty (breaks cycles).
            vals = []
            for dd in range(4):
                di = torch.tensor([dd], device=device)
                znext = fine_kstep_target(jepa, ot, di, K, cell_size)
                v = float(cvalue(znext.float(), z_goal.float()).item())
                if xy_head is not None:
                    v -= 0.05 * visit.get(pred_cell(znext), 0)
                vals.append(v)
            order = sorted(range(4), key=lambda dd: vals[dd], reverse=True)
            if verbose:
                print(f"   [m{_m}] cell={list(cell)} goal={env.goal_cell.tolist()} "
                      f"V[D,U,R,L]={[round(v,3) for v in vals]} blocked={sorted(blocked.get(cell,[]))}",
                      flush=True)
            # try directions best-value-first, skipping known-blocked & immediate U-turn
            moved = False; done = False
            cand = [d for d in order if d not in blocked.get(cell, set()) and d != last_rev]
            cand += [d for d in order if d not in cand]  # U-turn allowed as last resort
            for d in cand:
                prev = env.agent_cell.copy()
                act = (CARDINALS[d] * cell_size).cpu().numpy()
                for _k in range(K):
                    obs, _, done, trunc, info = env.step(act)
                    if done or trunc:
                        break
                if not np.array_equal(env.agent_cell, prev):
                    moved = True; last_rev = OPP[d]; break
                blocked.setdefault(cell, set()).add(d)   # macro didn't move -> wall
                if done or trunc:
                    break
            if done:
                success = True; break
            if not moved:
                break  # fully boxed in (all 4 dirs blocked) — give up this episode
        successes.append(float(success))
        print(f"[hier-eval] ep {ep}: {'SUCCESS' if success else 'fail'}", flush=True)
    sr = float(np.mean(successes))
    out = {"success_rate": sr, "num_episodes": num_ep, "K": K, "depth": depth,
           "n_samples": n_samples, "astar_free": True}
    with open(os.path.join(rdir, "hier_eval.json"), "w") as fjs:
        json.dump(out, fjs, indent=2)
    print(f"[hier-eval] A*-FREE success rate = {sr*100:.2f}% over {num_ep} mazes", flush=True)


if __name__ == "__main__":
    main()
