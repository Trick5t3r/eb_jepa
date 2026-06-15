# JEPA pipeline — modular scaffold for the hackathon tracks

A reusable, track-agnostic pipeline that factors out everything the track PoCs
(finance, LTSF, EEG, audio, point clouds, PDE fields) have in common:
**data → SSL JEPA → frozen-feature probe vs baselines**. The generic engine is
fully implemented; each track only fills a small, clearly-marked surface.

## What's generic (implemented — don't edit)

```
examples/pipeline/core/
  transforms.py   generic 1D augmentations for the two-view objective
  encoders.py     Conv1dEncoder (represent() + frames()); pointers for 2D/PointNet/mel
  jepa.py         PredictiveJEPA (RNNPredictor + EMA target + VCLoss)  &  TwoViewVICReg
  trainer.py      AMP SSL training loop (works for both objectives)
  probe.py        frozen / random / supervised probe harness
  metrics.py      classification / regression / forecasting metrics
  registry.py     @register_track / get_track
config.py         yaml + CLI-override loader
cli/pretrain.py   python -m examples.pipeline.cli.pretrain --track ...
cli/evaluate.py   python -m examples.pipeline.cli.evaluate  --track ... --ckpt ...
```

Both SSL objectives are built on the **eb_jepa core primitives** (`RNNPredictor`,
`VCLoss`, `VICRegLoss`, `Projector`) — the pipeline is a thin, faithful orchestration
of the repo, not a reimplementation.

## What a track provides (the `# TODO` surface)

A track is a subclass of `tracks/base.py:Track`. Minimum:

| hook | what you implement |
|---|---|
| `task_type` (+ `n_classes`) | classification / regression / forecasting |
| `in_channels` | number of input channels (sizes the default encoder) |
| `build_pretrain_dataset(cfg, split)` | unlabeled windows for SSL |
| `build_eval_dataset(cfg, split)` | `(x, y)` pairs for the probe |

Optional overrides: `build_encoder` (2D/PointNet/mel), `metric` (custom), `augment`
(SO(3) / SpecAugment / masking), `extra_baselines` (DLinear, FNO, …).

- **`tracks/template_track.py`** — copy this to start; every hook is a `# TODO` with a
  pointer to the matching reference worktree (fintime / ltsf / eeg_ssl / …).
- **`tracks/synthetic_track.py`** — a complete, runnable example on in-memory synthetic
  data (no files needed); used for the smoke test.

## Add a track in 4 steps

```bash
cp examples/pipeline/tracks/template_track.py examples/pipeline/tracks/my_track.py
# 1. fill the TODOs (data loaders, task_type, in_channels[, encoder/metric])
# 2. @register_track("my_track") and add the import to tracks/__init__.py
# 3. pretrain
python -m examples.pipeline.cli.pretrain --track my_track model.ssl=predictive optim.epochs=30
# 4. evaluate (frozen-SSL vs random vs supervised vs your baselines)
python -m examples.pipeline.cli.evaluate --track my_track --ckpt <ckpt>/latest.pth.tar
```

## Run on SLURM

```bash
TRACK=synthetic SSL=predictive sbatch cluster/pipeline.sbatch smoke
TRACK=synthetic SSL=predictive EPOCHS=30 sbatch cluster/pipeline.sbatch pretrain
TRACK=synthetic SSL=predictive sbatch cluster/pipeline.sbatch eval
```

## Choosing the SSL objective

- `model.ssl=predictive` — latent prediction + anti-collapse (needs `frames()`; best
  when there is temporal/sequence structure to predict).
- `model.ssl=vicreg` — two augmented views + invariance + anti-collapse (needs only
  `represent()` and an `augment`; best for modalities with strong augmentations:
  audio, point clouds, EEG).
