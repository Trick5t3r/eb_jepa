# Hierarchical maze JEPA — A*-FREE navigation (trained with A*, evaluated without)

Goal: **make the maze solvable at eval with NO A*** (no waypoints, no A* prior, no
A* fallback), via a two-level hierarchy. A* is used **only as a training teacher**.

> **Result.** A learned high-level **SubgoalPredictor** (supervised on A* waypoints)
> + a low-level reacher (the frozen fine world model, which is wall-aware) navigates
> 21×21 mazes **A*-FREE at 31.25 %** (16 held-out mazes), vs **0 %** for greedy
> planning without the hierarchy. Co-training the two levels jointly *hurt* (→12.5 %).

## The idea (feudal / closed-loop subgoals)

Planning straight to a far global goal fails (0 %): the value/horizon can't span a
50-cell maze. The fix is a **two-level** decomposition:

- **High level — `SubgoalPredictor(z_current, goal_xy) → next waypoint position`**
  (`eb_jepa/hierarchical.py`). The state latent `z_current` encodes the *whole*
  maze (the wall mask is in the obs image) plus the agent position, so the predictor
  can learn to route. Trained **supervised on A* trajectories**: label = the A*
  position `N=4` cells ahead (`loc[t+N]`). MSE loss. A* only teaches; at eval the
  predictor proposes the waypoints itself.
- **Low level — reach the waypoint** with the **frozen fine world model**, which is
  *wall-aware* (it was trained with `wall_bump_prob`, so it predicts "stay" into a
  wall). At each step we roll the fine WM 1 step per cardinal, pick the one whose
  probe-decoded position is closest to the waypoint, with **execution-feedback
  blocked-skip** (a direction that doesn't move the agent is blacklisted at that
  cell) + no-immediate-U-turn. Fully closed-loop: the subgoal is re-predicted every
  step.

This is exactly "trained with A*, evaluated without": the high level distils A*'s
routing into a learned model; the low level is the proven wall-aware world model.

## Files
- `eb_jepa/hierarchical.py` — `SubgoalPredictor` (+ a `CoarsePredictor` variant)
- `examples/ac_video_jepa/main_subgoal.py` — train the SubgoalPredictor (frozen fine WM)
- `examples/ac_video_jepa/eval_subgoal.py` — A*-FREE closed-loop eval
- `examples/ac_video_jepa/main_cotrain.py` — joint fine-tune (shared latent) phase
- `scripts/maze_subgoal.sbatch`, `maze_cotrain.sbatch`

## Results (`results/maze_hierarchical/`)

| variant (16 held-out 21×21 mazes, **no A* at eval**) | A*-free success |
|---|---|
| greedy, no hierarchy (for reference) | **0 %** |
| **SubgoalPredictor, frozen fine WM (12 ep)** | **31.25 %** |
| co-training (encoder+predictor+probe+subgoal, shared latent, 8 ep) | 12.50 % |

**What worked.** The hierarchy turns 0 % → 31.25 % A*-free. The high level learns
A*-like routing (waypoint MSE 0.108 → 0.059), the low level follows it with the
wall-aware fine WM. The agent genuinely crosses 21×21 mazes with **zero A* at eval**.

**What didn't: co-training.** Jointly fine-tuning the shared latent (low encoder LR
2e-4 + the three losses) *degraded* it to 12.5 %. The logs show the **subgoal loss
rising during co-training** (0.084 → 0.091): the encoder drifts toward the dynamics
objective and breaks the warm-started subgoal head — interference, not the hoped-for
shared-representation alignment. The "non-shared bias" intuition is sound in theory,
but here the joint optimization needs a much gentler encoder LR / staged unfreezing
to avoid wrecking the high level. Honest negative.

**Caveat.** 16 episodes is a high-variance estimate (an early 4-maze smoke read 75 %).
31.25 % is the reliable 16-maze number; the controlled 0 %→31 % gain from the
hierarchy is the trustworthy signal.

## Reproduce
```bash
sbatch scripts/maze_subgoal.sbatch 4 full 12     # train high level + A*-free eval
sbatch scripts/maze_cotrain.sbatch 4 8           # co-training phase + eval
```
