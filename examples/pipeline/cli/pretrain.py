"""SSL pretraining entrypoint (generic, track-agnostic).

  python -m examples.pipeline.cli.pretrain --track synthetic \
         --config examples/pipeline/configs/base.yaml model.ssl=predictive optim.epochs=30
"""
import argparse

import torch

from examples.pipeline.config import load_config
from examples.pipeline.core.jepa import build_ssl
from examples.pipeline.core.registry import get_track, list_tracks
from examples.pipeline.core.trainer import pretrain


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", required=True, help=f"one of {list_tracks()}")
    ap.add_argument("--config", default=None)
    ap.add_argument("overrides", nargs="*", help="dotlist, e.g. model.ssl=vicreg")
    args = ap.parse_args()

    cfg = load_config(args.config, args.overrides)
    torch.manual_seed(cfg.meta.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    track = get_track(args.track)

    encoder = track.build_encoder(cfg)
    ssl = build_ssl(cfg.model.ssl, encoder, cfg.model)
    print(f"[pretrain] track={args.track} ssl={cfg.model.ssl} "
          f"D={encoder.out_dim} params={sum(p.numel() for p in ssl.parameters())/1e6:.2f}M",
          flush=True)

    mk = lambda split, sh: torch.utils.data.DataLoader(
        track.build_pretrain_dataset(cfg, split), batch_size=cfg.data.batch_size,
        shuffle=sh, num_workers=cfg.data.num_workers, pin_memory=True,
        drop_last=True, persistent_workers=cfg.data.num_workers > 0)
    train_loader, val_loader = mk("train", True), mk("val", False)
    print(f"[pretrain] {len(train_loader.dataset)} train / {len(val_loader.dataset)} val windows",
          flush=True)

    pretrain(ssl, train_loader, val_loader, cfg, device, cfg.meta.ckpt_dir)


if __name__ == "__main__":
    main()
