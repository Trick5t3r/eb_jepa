#!/usr/bin/env bash
#SBATCH --partition=defq
#SBATCH --account=pdl17890
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=1:00:00
#SBATCH --job-name=hwt_eval_baseline
#SBATCH --output=/lustre/work/pdl17890/udl806719/logs/hwt_eval_baseline_%j.out

set -e

REPO=/lustre/home/extusers/udl806719/dev/eb_jepa
source "$REPO/env.sh"
cd "$REPO"

PYTHON="$UV_PROJECT_ENVIRONMENT/bin/python3"
CFG="$REPO/examples/ac_video_jepa/cfgs/train_ex_reduction_of_walltime.yaml"
SEED=1
EXP_FOLDER="/lustre/work/pdl17890/udl806719/checkpoints/ac_video_jepa/hwt_2026-06-02/impala_cov8_std16_simt12_idm1_baseline_seed${SEED}"

echo "=== hwt eval baseline ==="
echo "    node       : $(hostname)"
echo "    GPU        : $CUDA_VISIBLE_DEVICES"
echo "    start      : $(date)"
echo "    exp_folder : $EXP_FOLDER"

"$PYTHON" -m examples.ac_video_jepa.main \
    "$CFG" \
    --meta.model_folder="$EXP_FOLDER" \
    --data.pipeline.mode=online \
    --data.num_workers=4 \
    --meta.seed=$SEED \
    --meta.enable_plan_eval=True \
    --meta.eval_only_mode=True

echo "=== Done: $(date) ==="
