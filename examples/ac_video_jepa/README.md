# Action-Conditioned Video JEPA

An action-conditioned JEPA world model: an Impala encoder maps observations to latents,
an `RNNPredictor` predicts the **next latent given the action**, and a planner reaches
goals by minimizing energy in latent space (no pixel reconstruction). Demonstrated on the
**Two Rooms** environment.

The full write-up — architecture, dataset/pipeline, training, and planning — lives in
**[`two_rooms/README.md`](two_rooms/README.md)**.

## Quick start
```bash
python -m examples.ac_video_jepa.main --fname examples/ac_video_jepa/two_rooms/cfgs/train.yaml
```

## Layout
```
examples/ac_video_jepa/
  main.py / eval.py     # shared trainer / eval (planning-as-energy-minimization)
  two_rooms/            # the Two Rooms use-case: cfgs/, assets/, README
```
