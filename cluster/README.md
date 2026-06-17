# cluster/ — Utility scripts for the HTW DALIA cluster

Scripts for monitoring jobs, GPU usage, and logs. All read-only — they never submit or cancel anything.

Add to PATH for convenience (already done if you source `env.sh` with the snippet below):

```bash
export PATH="$EBJEPA_REPO/cluster:$PATH"   # $EBJEPA_REPO is set by env.sh
```

---

## Scripts

### `sq` — My jobs

```
sq
```

Shows your running and pending jobs, color-coded:
- **green** = RUNNING
- **yellow** = PENDING
- **red** = FAILED / CANCELLED / TIMEOUT

Prints a state summary at the bottom.

---

### `qall` — All jobs on defq

```
qall
```

Shows every job on the `defq` partition, sorted by user, with a per-user GPU/CPU/node summary at the bottom.

---

### `log` — View a job log

```
log [JOBID] [-f]
```

| Invocation | Behavior |
|------------|----------|
| `log` | Show the most recent job's stdout |
| `log 62022` | Show job 62022 stdout |
| `log -f` | Tail the most recent job's log (live) |
| `log 62022 -f` | Tail job 62022 live |

- Auto-discovers the log file path via `scontrol` (running jobs) or `sacct` (completed jobs).
- If stdout is empty, falls back to showing stderr automatically (useful for FAILED jobs).
- Works with both `slurm_test.sh` jobs and submitit-launched training jobs.

---

### `gpus` — GPU allocation per node

```
gpus
```

Shows a per-node table for all 18 `dalianvl` nodes:

| Column | Meaning |
|--------|---------|
| TOT | Total GPUs on the node (always 4 × GB200) |
| USED | Currently allocated |
| FREE | Available |
| STATE | SLURM node state (idle / mixed / allocated / drained) |
| CPU_LOAD | Current CPU load average |
| FREE_MEM | Free RAM in GB |

Color: green = fully free, yellow = partially used, red = fully allocated.

Shows total GPU counts (used / total / free) at the bottom.

---

### `users` — Resource usage per user

```
users
```

Shows GPU, CPU, and node counts per user on `defq`, sorted by GPU usage descending. Also shows job state counts at the bottom.

---

## Architecture variables (for training jobs)

`env.sh` exports:
```bash
EBJEPA_COMPUTE_ARCH=aarch64   # target arch for SLURM compute nodes
```

Override before sourcing to target a different arch:
```bash
export EBJEPA_COMPUTE_ARCH=x86_64 && source env.sh
```

This drives `COMPUTE_PYTHON` in `launch_sbatch.py`, which is the Python binary submitit will use on compute nodes.

---

## W&B toggle

```bash
# Disable W&B globally (before sourcing env.sh)
export WANDB_DISABLED=true && source env.sh

# Re-enable
export WANDB_DISABLED=false && source env.sh
```

Or per-run via config override:
```bash
python -m examples.launch_sbatch --example ac_video_jepa --single  # uses train.yaml default (log_wandb: true)
```
