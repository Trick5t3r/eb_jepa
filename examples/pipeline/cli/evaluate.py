"""Downstream evaluation entrypoint (generic). Runs the full comparison:
frozen-SSL vs random-encoder vs supervised end-to-end (+ any track baselines).

  python -m examples.pipeline.cli.evaluate --track synthetic \
         --ckpt /.../checkpoints/pipeline/run/latest.pth.tar
"""
import argparse
import json
import os

import torch
from omegaconf import OmegaConf

from examples.pipeline.config import load_config
from examples.pipeline.core.probe import frozen_probe, supervised_baseline
from examples.pipeline.core.registry import get_track, list_tracks


def _fmt(d):
    return " ".join(f"{k}={v:.4f}" for k, v in d.items())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", required=True, help=f"one of {list_tracks()}")
    ap.add_argument("--ckpt", required=True, help="SSL checkpoint (latest.pth.tar)")
    ap.add_argument("--config", default=None)
    ap.add_argument("--sup-epochs", type=int, default=10)
    ap.add_argument("--results-dir", default=None)
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    track = get_track(args.track)
    # prefer the cfg stored in the checkpoint so the encoder matches the SSL run
    state = torch.load(args.ckpt, map_location=device, weights_only=False)
    cfg = OmegaConf.create(state["cfg"]) if "cfg" in state else load_config(args.config, args.overrides)
    if args.overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(args.overrides))

    results = {}

    # 1) frozen SSL encoder
    enc = track.build_encoder(cfg).to(device)
    enc.load_state_dict(state["encoder"])
    results["frozen_ssl"] = track_metric = frozen_probe(enc, track, cfg, device)
    print(f"[eval] frozen_ssl   : {_fmt(results['frozen_ssl'])}", flush=True)

    # 2) random-encoder floor
    rnd = track.build_encoder(cfg).to(device)
    results["random"] = frozen_probe(rnd, track, cfg, device)
    print(f"[eval] random       : {_fmt(results['random'])}", flush=True)

    # 3) supervised end-to-end (direct prediction)
    results["supervised"] = supervised_baseline(track, cfg, device, epochs=args.sup_epochs)
    print(f"[eval] supervised   : {_fmt(results['supervised'])}", flush=True)

    # 4) track-specific baselines (DLinear, FNO, ...), if any
    for name, m in track.extra_baselines(cfg, device).items():
        results[name] = m
        print(f"[eval] {name:<13}: {_fmt(m)}", flush=True)

    rdir = args.results_dir or os.environ.get("RESULTS_DIR")
    if rdir:
        os.makedirs(rdir, exist_ok=True)
        with open(os.path.join(rdir, f"eval_{args.track}.json"), "w") as f:
            json.dump({"track": args.track, "task": track.task_type, "results": results}, f, indent=2)
        print(f"[eval] saved -> {rdir}/eval_{args.track}.json", flush=True)


if __name__ == "__main__":
    main()
