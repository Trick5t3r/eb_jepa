# Maze Pipeline — Implementation Plan

## Source files (provided)

| File | Role |
|------|------|
| `maze_generator.py` | DFS random maze generator → 2D array (0=wall, 1=path) |
| `maze_solver.py` | A* solver → list of (row, col) positions + discrete actions (0=up, 1=down, 2=left, 3=right) |
| `dataset_builder.py` | RGB frame renderer + .npz/.mp4 serialiser |
| `test.py` | Shape validation |

---

## New files — `eb_jepa/datasets/maze/`

| File | Role |
|------|------|
| `maze_generator.py` | Copy of provided DFS generator |
| `maze_solver.py` | Copy of provided A* solver |
| `normalizer.py` | `MazeNormalizer` — same interface as `two_rooms/normalizer.py` |
| `maze_dataset.py` | `MazeDataset`, `MazeDatasetConfig`, `MazeSample` |
| `env.py` | `MazeEnv(gym.Env)` — eval environment for MPPI/CEM planner |
| `data_config.yaml` | Default config (maze_height=21, maze_width=21, cell_size=3, img_size=63, …) |

---

## Data format (mirrors two_rooms)

### Images — 2 channels, 63×63 px
- Maze: 21×21 cells × cell_size=3 → 63×63 pixels
- **Channel 0** — agent position: Gaussian dot (std ≈ 1.5 px) centred on the agent's cell
- **Channel 1** — maze structure: 0=path, 1=wall, 0.5=goal cell

### Tensors returned by `MazeDataset.__getitem__`

| Field | Shape | Description |
|-------|-------|-------------|
| `states` | `(2, T, 63, 63)` | 2-channel frames, float normalised |
| `actions` | `(2, T)` | 2D direction vectors (dr, dc) ∈ {(−1,0),(1,0),(0,−1),(0,1)} |
| `locations` | `(2, T)` | Pixel-space agent position (row_px, col_px) |
| `wall_x` | `(1,)` | Dummy (set to 0) — keeps WallSample-compatible 5-field interface |
| `door_y` | `(1,)` | Dummy (set to 0) |

`MazeSample` is a `NamedTuple` with the same 5 fields as `WallSample` so the training loop unpacks identically:
```python
x, a, loc, wall_x, door_y = next(loader_iter)
```

### `__getitem__` generation logic
1. Generate a random DFS maze (21×21)
2. Solve A* from (1,1) to (19,19)
3. If path length < `sample_length + 1`: retry (up to 50 times)
4. Sample a random contiguous window of `sample_length + 1` steps
5. Render `sample_length` frames (wall channel precomputed once per maze, agent channel per step)
6. Normalise states via `MazeNormalizer.normalize_state`, locations via `normalize_location`
7. Return `MazeSample`

---

## Eval environment — `MazeEnv(gym.Env)`

Matches the `DotWall` interface used by `MPPIPlanner` / `CEMPlanner`:

| Method | Behaviour |
|--------|-----------|
| `reset(location=None)` | Generate new maze, place agent at (1,1), goal at (H-2,W-2). Returns `(obs, info)` where `info` contains `dot_position`, `target_position`, `target_obs` |
| `step(action)` | MPPI passes a continuous 2D vector; discretise to nearest cardinal direction; move if target cell is a path, else stay. Returns `(obs, reward, done, truncated, info)` |
| `eval_state(goal_pos, curr_pos)` | Success if Euclidean pixel distance < threshold (≈ 1 cell = 3 px) |
| `get_target_obs()` | 2-channel image with agent rendered at goal cell |
| `action_space` | `Box(low=-1, high=1, shape=(2,))` |
| `observation_space` | `Box(low=0, high=1, shape=(2, 63, 63))` |
| `.normalizer` | `MazeNormalizer` instance (needed by `xy_head` and `GCAgent`) |

---

## Minimal changes to existing files

### `eb_jepa/datasets/utils.py`
Add `elif env_name == "maze":` branch in `init_data` — loads `MazeDatasetConfig` and returns `(loader, val_loader, config, None)`.

### `examples/ac_video_jepa/main.py`
Replace the hardcoded `DotWall` import in `env_creator` with a dispatch on `cfg.data.env_name`:
```python
def env_creator():
    from eb_jepa.datasets.utils import create_env
    return create_env(cfg.data.env_name, env_config, **eval_cfg_dict.get("env", {}))
```
Add `create_env(env_name, config, **kwargs)` helper to `utils.py`.

### New config files
- `examples/ac_video_jepa/maze/cfgs/train_maze.yaml` — same model/optim as `train.yaml` but `env_name=maze`, `dobs=2`, `img_size` implicit from `data_config.yaml`
- `examples/ac_video_jepa/maze/cfgs/eval_maze.yaml` — `env_name=maze`, `n_allowed_steps=200`

---

## What stays unchanged

- Model architecture (ImpalaEncoder + RNNPredictor)
- Training loop (JEPA, VC_IDM_Sim_Regularizer, AdamW)
- MPPI/CEM planner (works with ±1 magnitude actions)
- All two_rooms code (parallel env, untouched)

---

## Key config parameters (`data_config.yaml`)

```yaml
maze_height: 21
maze_width: 21
cell_size: 3
img_size: 63          # maze_height * cell_size
sample_length: 17
min_path_length: 30   # reject mazes with path shorter than this
agent_std: 1.5        # Gaussian dot std (pixels)
size: 100000
val_size: 10000
batch_size: 64
normalize: true
device: cpu
```
