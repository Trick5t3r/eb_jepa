"""Generic downstream-probe harness — the Track-5/6 comparison engine.

For a frozen encoder it extracts the pooled representation on train/test, fits a
linear head (LogReg for classification, Ridge for regression/forecasting), and
scores with the track's metric. It also provides the two reference baselines every
track compares against:

  * ``random``      a frozen untrained encoder (representation floor)
  * ``supervised``  encoder + linear head trained end-to-end (= direct prediction)

A track-specific extra baseline (DLinear for LTSF, FNO/U-Net for fields) lives in
the track module and is plugged via ``Track.extra_baselines`` (#TODO, optional).
GENERIC otherwise — students don't edit this.
"""
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler


@torch.no_grad()
def _features(encoder, ds, device, bs=512):
    encoder.eval()
    loader = torch.utils.data.DataLoader(ds, batch_size=bs, shuffle=False, num_workers=8)
    X, Y = [], []
    for x, y in loader:
        X.append(encoder.represent(x.to(device, non_blocking=True)).cpu().numpy())
        Y.append(y.numpy())
    return np.concatenate(X), np.concatenate(Y)


def frozen_probe(encoder, track, cfg, device):
    tr = track.build_eval_dataset(cfg, "train")
    te = track.build_eval_dataset(cfg, "test")
    Xtr, ytr = _features(encoder, tr, device)
    Xte, yte = _features(encoder, te, device)
    tt = track.task_type
    if tt == "classification":
        sc = StandardScaler().fit(Xtr)
        clf = LogisticRegression(max_iter=3000, C=1.0, class_weight="balanced")
        clf.fit(sc.transform(Xtr), ytr)
        proba = clf.predict_proba(sc.transform(Xte))
        prob = proba[:, 1] if track.n_classes == 2 else proba
        return track.metric(yte, prob)
    # regression / forecasting: ridge on raw features
    n = len(ytr)
    Ytr = ytr.reshape(n, -1)
    reg = Ridge(alpha=10.0).fit(Xtr, Ytr)
    pred = reg.predict(Xte).reshape(yte.shape)
    return track.metric(yte, pred)


def supervised_baseline(track, cfg, device, epochs=10):
    """Encoder + linear head trained end-to-end on the labels (direct prediction)."""
    from torch import nn
    enc = track.build_encoder(cfg).to(device)
    tr = track.build_eval_dataset(cfg, "train")
    te = track.build_eval_dataset(cfg, "test")
    x0, y0 = tr[0]
    is_cls = track.task_type == "classification"
    out_dim = track.n_classes if is_cls else int(np.prod(np.asarray(y0).shape))
    head = nn.Linear(enc.out_dim, out_dim).to(device)
    opt = torch.optim.AdamW(list(enc.parameters()) + list(head.parameters()),
                            lr=1e-3, weight_decay=1e-5)
    loader = torch.utils.data.DataLoader(tr, batch_size=64, shuffle=True,
                                         num_workers=8, drop_last=True)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3,
                                                total_steps=epochs * len(loader), pct_start=0.1)
    lossf = nn.CrossEntropyLoss() if is_cls else nn.MSELoss()
    for ep in range(epochs):
        enc.train(); head.train()
        for x, y in loader:
            x = x.to(device); y = y.to(device)
            opt.zero_grad(set_to_none=True)
            out = head(enc.represent(x))
            loss = lossf(out, y) if is_cls else lossf(out, y.reshape(y.shape[0], -1).float())
            loss.backward(); opt.step(); sched.step()
        print(f"  [supervised] epoch {ep} loss={loss.item():.4f}", flush=True)
    # eval
    enc.eval(); head.eval()
    tel = torch.utils.data.DataLoader(te, batch_size=512, shuffle=False, num_workers=8)
    P, Y = [], []
    with torch.no_grad():
        for x, y in tel:
            out = head(enc.represent(x.to(device)))
            P.append((torch.softmax(out, -1) if is_cls else out).cpu().numpy()); Y.append(y.numpy())
    P, Y = np.concatenate(P), np.concatenate(Y)
    if is_cls:
        prob = P[:, 1] if track.n_classes == 2 else P
        return track.metric(Y, prob)
    return track.metric(Y, P.reshape(Y.shape))
