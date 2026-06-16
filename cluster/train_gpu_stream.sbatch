#!/usr/bin/env bash
# Training + sequential eval — GPU stream data pipeline (no CPU workers).
# Data gen is fully GPU-side; 16 CPUs covers val_loader (4 workers) + main process.
# Two-phase: train (enable_plan_eval=false), then eval_only_mode on same checkpoint.

#SBATCH --partition=defq
#SBATCH --account=pdl17890
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=1:30:0
#SBATCH --job-name=eb_jepa_gpu_stream
#SBATCH --output=/lustre/work/pdl17890/%u/logs/gpu_stream_%j.out

set -e

REPO="${EBJEPA_REPO:-$SLURM_SUBMIT_DIR}"
source "$REPO/env.sh"
cd "$REPO"

PYTHON="$UV_PROJECT_ENVIRONMENT/bin/python3"
CFG="$REPO/examples/ac_video_jepa/cfgs/train_ex_reduction_of_walltime.yaml"

# Shared experiment folder so Phase 2 can load Phase 1's checkpoint.
# Format mirrors get_unified_experiment_dir: .../ac_video_jepa/{sweep}/{exp_name}_seed{seed}/
SEED=1
SWEEP="dev_$(date +%Y-%m-%d_%H-%M)"
EXP_NAME="impala_cov8_std16_simt12_idm1"
CKPT_BASE="${WORK}/checkpoints"
EXP_FOLDER="${CKPT_BASE}/ac_video_jepa/${SWEEP}/${EXP_NAME}_seed${SEED}"

echo "=== Phase 1: Training ==="
echo "    node        : $(hostname)"
echo "    GPU         : $CUDA_VISIBLE_DEVICES"
echo "    start       : $(date)"
echo "    exp_folder  : $EXP_FOLDER"

"$PYTHON" -m examples.ac_video_jepa.main \
    "$CFG" \
    --meta.model_folder="$EXP_FOLDER" \
    --data.pipeline.mode=stream \
    --data.pipeline.backend=gpu \
    --data.pipeline.chunk_size=3840 \
    --data.num_workers=4 \
    --meta.exp_suffix=gpu_stream \
    --meta.seed=$SEED

echo ""
echo "=== Phase 2: Sequential eval (20 episodes) ==="
echo "    start: $(date)"

"$PYTHON" -m examples.ac_video_jepa.main \
    "$CFG" \
    --meta.model_folder="$EXP_FOLDER" \
    --data.pipeline.mode=online \
    --data.num_workers=4 \
    --meta.exp_suffix=gpu_stream \
    --meta.seed=$SEED \
    --meta.enable_plan_eval=True \
    --meta.eval_only_mode=True

echo ""
echo "=== Done: $(date) ==="
