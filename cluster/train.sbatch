#!/usr/bin/env bash
# Submit an eb_jepa training job.
# Usage: sbatch cluster/train.sbatch [config_path] [--key=value overrides...]
# Default config: examples/ac_video_jepa/cfgs/train_fast.yaml
#
# Examples:
#   sbatch cluster/train.sbatch
#   sbatch cluster/train.sbatch examples/ac_video_jepa/cfgs/train.yaml
#   sbatch cluster/train.sbatch examples/ac_video_jepa/cfgs/train_fast.yaml --meta.seed=42

#SBATCH --partition=defq
#SBATCH --account=pdl17890
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --time=1:0:0
#SBATCH --job-name=eb_jepa_train
#SBATCH --output=/lustre/work/pdl17890/%u/logs/train_%j.out

set -e

REPO="${EBJEPA_REPO:-$SLURM_SUBMIT_DIR}"
source "$REPO/env.sh"
cd "$REPO"

CFG="${1:-examples/ac_video_jepa/cfgs/train_fast.yaml}"
shift 2>/dev/null || true   # remaining args are key=value overrides

echo "=== eb_jepa training ==="
echo "    config : $CFG"
echo "    node   : $(hostname)"
echo "    GPU    : $CUDA_VISIBLE_DEVICES"
echo "    overrides: $*"
echo ""

"$UV_PROJECT_ENVIRONMENT/bin/python3" -m examples.ac_video_jepa.main \
    --fname "$CFG" \
    "$@"
