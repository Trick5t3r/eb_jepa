#!/usr/bin/env bash
#SBATCH --partition=defq
#SBATCH --account=pdl17890
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --time=2:00:00
#SBATCH --job-name=hwt_offline
#SBATCH --output=/lustre/work/pdl17890/udl806719/logs/hwt_offline_%j.out

set -e

REPO=/lustre/home/extusers/udl806719/dev/eb_jepa
source "$REPO/env.sh"
cd "$REPO"

PYTHON="$UV_PROJECT_ENVIRONMENT/bin/python3"
CFG="$REPO/examples/ac_video_jepa/cfgs/train_ex_reduction_of_walltime.yaml"
SEED=1
EXP_FOLDER="/lustre/work/pdl17890/udl806719/checkpoints/ac_video_jepa/hwt_2026-06-02/impala_cov8_std16_simt12_idm1_offline_seed${SEED}"
DSET_DIR="/lustre/work/pdl17890/udl806719/datasets/two_rooms_hwt_2026-06-02"

echo "=== hwt offline: Phase 1 — dataset generation ==="
echo "    node       : $(hostname)"
echo "    GPU        : $CUDA_VISIBLE_DEVICES"
echo "    start      : $(date)"
echo "    dset_dir   : $DSET_DIR"

"$PYTHON" -m eb_jepa.datasets.two_rooms.offline_dataset \
    --data-dir "$DSET_DIR" \
    --num-samples 1200000 \
    --backend gpu \
    --seed $SEED

echo ""
echo "=== hwt offline: Phase 2 — training ==="
echo "    start      : $(date)"
echo "    exp_folder : $EXP_FOLDER"

"$PYTHON" -m examples.ac_video_jepa.main \
    "$CFG" \
    --meta.model_folder="$EXP_FOLDER" \
    --data.pipeline.mode=offline \
    --data.pipeline.data_dir="$DSET_DIR" \
    --meta.seed=$SEED

echo "=== Done: $(date) ==="
