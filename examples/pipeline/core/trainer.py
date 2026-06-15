"""Generic SSL pretraining loop. Modality-agnostic: works for both the predictive
JEPA (batch = one tensor x) and two-view VICReg (batch = (v1, v2)), because the SSL
module exposes a uniform ``compute_loss(batch) -> (loss, logs)`` and ``on_step_end()``.
GENERIC — students never edit this."""
import os
import time

import torch
from omegaconf import OmegaConf


def _to_device(batch, device):
    if torch.is_tensor(batch):
        return batch.to(device, non_blocking=True)
    return [b.to(device, non_blocking=True) for b in batch]


def pretrain(ssl, train_loader, val_loader, cfg, device, ckpt_dir):
    ssl = ssl.to(device)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    spe = len(train_loader)
    opt = torch.optim.AdamW(ssl.parameters(), lr=cfg.optim.lr,
                            weight_decay=cfg.optim.weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=cfg.optim.lr, total_steps=cfg.optim.epochs * spe,
        pct_start=0.1, anneal_strategy="cos")
    use_amp = bool(cfg.training.use_amp) and device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    os.makedirs(ckpt_dir, exist_ok=True)

    g = 0
    for epoch in range(cfg.optim.epochs):
        ssl.train(); t0 = time.time()
        for batch in train_loader:
            batch = _to_device(batch, device)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast(device.type, enabled=use_amp):
                loss, logs = ssl.compute_loss(batch)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            sched.step(); ssl.on_step_end(); g += 1
            if g % cfg.logging.log_every == 0:
                extra = " ".join(f"{k}={v:.3f}" for k, v in logs.items())
                print(f"e{epoch} s{g} loss={loss.item():.4f} {extra}", flush=True)

        ssl.eval(); vl, nb = 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                batch = _to_device(batch, device)
                with torch.amp.autocast(device.type, enabled=use_amp):
                    l, _ = ssl.compute_loss(batch)
                vl += l.item(); nb += 1
        print(f"[epoch {epoch}] {time.time()-t0:.0f}s val_loss={vl/max(nb,1):.4f}", flush=True)
        torch.save({"epoch": epoch, "encoder": ssl.encoder.state_dict(),
                    "cfg": OmegaConf.to_container(cfg, resolve=True)},
                   os.path.join(ckpt_dir, "latest.pth.tar"))
    print(f"[pretrain] done -> {ckpt_dir}/latest.pth.tar", flush=True)
    return os.path.join(ckpt_dir, "latest.pth.tar")
