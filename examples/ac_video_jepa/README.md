# Action-Conditioned Video JEPA

An action-conditioned JEPA world model: an encoder maps observations to latents, an
`RNNPredictor` predicts the **next latent given the action**, and a planner reaches
goals by minimizing energy in latent space (no pixel reconstruction). The same
stack serves **two use-cases**, selected by `env_name` in the config:

| use-case | what | guide |
|---|---|---|
| **Two Rooms** | the original example: a dot navigating two rooms; generation, training, planning | [`two_rooms/README.md`](two_rooms/README.md) |
| **Maze** | the same stack on procedurally A\*-generated mazes, plus a learned hierarchy for **A\*-free** navigation (baseline → Level 1 → Level 2, all modular) | [`maze/README.md`](maze/README.md) |

## Layout (mirrors `eb_jepa/datasets/`)
Shared code stays at the top; everything env-specific lives in a subfolder.
```
examples/ac_video_jepa/
  main.py            # SHARED trainer  (env_name: two_rooms | maze)
  eval.py            # SHARED eval / planning launcher
  cfgs/              # SHARED configs (some, e.g. planning_mppi.yaml, are used by both)
  two_rooms/         # Two Rooms: README + assets
  maze/              # Maze: scripts (subgoal/cotrain/random/...), READMEs (overview, value, hierarchical)
```
Why `main.py`/`eval.py`/`cfgs/` are shared and not duplicated: the trainer is
env-agnostic (the dataset/env is dispatched by `env_name`), exactly like
`eb_jepa/datasets/utils.py` sits above `datasets/two_rooms/` and `datasets/maze/`;
and several configs (e.g. `planning_mppi.yaml`) are referenced by both use-cases.

## Quick start
```bash
# Two Rooms (see two_rooms/README.md)
python -m examples.ac_video_jepa.main examples/ac_video_jepa/cfgs/train.yaml

# Maze (see maze/README.md): train the world model, then A*-free navigation
python -m examples.ac_video_jepa.main examples/ac_video_jepa/cfgs/train_maze_aux.yaml \
    --meta.model_folder=$EBJEPA_CKPTS/maze/exp_value
python -m examples.ac_video_jepa.maze.main_subgoal <fine_ckpt> <out_dir> 4 12
python -m examples.ac_video_jepa.maze.eval_subgoal <fine_ckpt> <out_dir>/subgoal.pth.tar \
       results/maze_subgoal 32 4 0.05 32 4 10
```

The shared model architecture (Impala encoder, `RNNPredictor`, regularizers,
planning-as-energy-minimization) is documented in **[`two_rooms/README.md`](two_rooms/README.md)**.
