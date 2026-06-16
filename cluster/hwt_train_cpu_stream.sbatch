#!/usr/bin/env bash
#SBATCH --partition=defq
#SBATCH --account=pdl17890
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --time=2:00:00
#SBATCH --job-name=hwt_cpu_stream
#SBATCH --output=/lustre/work/pdl17890/%u/logs/hwt_cpu_stream_%j.out

set -e

REPO="${EBJEPA_REPO:-$SLURM_SUBMIT_DIR}"
source "$REPO/env.sh"
cd "$REPO"

PYTHON="$UV_PROJECT_ENVIRONMENT/bin/python3"
CFG="$REPO/examples/ac_video_jepa/cfgs/train_ex_reduction_of_walltime.yaml"
SEED=1
EXP_FOLDER="${WORK}/checkpoints/ac_video_jepa/hwt_2026-06-02/impala_cov8_std16_simt12_idm1_cpu_stream_seed${SEED}"

echo "=== hwt cpu stream ==="
echo "    node       : $(hostname)"
echo "    GPU        : $CUDA_VISIBLE_DEVICES"
echo "    start      : $(date)"
echo "    exp_folder : $EXP_FOLDER"

"$PYTHON" -m examples.ac_video_jepa.main \
    "$CFG" \
    --meta.model_folder="$EXP_FOLDER" \
    --data.pipeline.mode=stream \
    --data.pipeline.backend=cpu \
    --data.pipeline.chunk_size=3840 \
    --data.pipeline.num_gen_workers=24 \
    --data.num_workers=4 \
    --meta.seed=$SEED

echo "=== Done: $(date) ==="
