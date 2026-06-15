#!/usr/bin/env bash
# Generic JEPA pipeline runner. Stage=$1: smoke | pretrain | eval
# Track + ssl method + config are passed via env: TRACK, SSL, CONFIG, CKPT_DIR.
#SBATCH --partition=defq
#SBATCH --account=pdl17890
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=1:30:00
#SBATCH --output=/lustre/work/pdl17890/udl806719/logs/pipeline_%x_%j.out
set -e

STAGE="${1:-smoke}"
TRACK="${TRACK:-synthetic}"
SSL="${SSL:-predictive}"
EPOCHS="${EPOCHS:-30}"
REPO=/lustre/work/pdl17890/udl806719/worktrees/pipeline_tracks
CKPT_DIR="${CKPT_DIR:-/lustre/work/pdl17890/udl806719/checkpoints/pipeline/${TRACK}_${SSL}}"
CONFIG="${CONFIG:-examples/pipeline/configs/base.yaml}"
export WANDB_MODE=disabled WANDB_DISABLED=true
source "$REPO/env.sh"; cd "$REPO"
export PYTHONPATH="$REPO:$PYTHONPATH"
PY="$UV_PROJECT_ENVIRONMENT/bin/python3"
CK="$CKPT_DIR/latest.pth.tar"
echo "=== STAGE=$STAGE TRACK=$TRACK SSL=$SSL node=$(hostname) $(date) ==="
$PY -c "import eb_jepa; print('eb_jepa ->', eb_jepa.__file__)"

case "$STAGE" in
  smoke)
    $PY -m examples.pipeline.cli.pretrain --track "$TRACK" --config "$CONFIG" \
        model.ssl="$SSL" optim.epochs=3 data.num_workers=4 meta.ckpt_dir="$CKPT_DIR"
    export RESULTS_DIR="$REPO/results/pipeline_smoke"
    $PY -m examples.pipeline.cli.evaluate --track "$TRACK" --ckpt "$CK" --sup-epochs 3
    ;;
  pretrain)
    $PY -m examples.pipeline.cli.pretrain --track "$TRACK" --config "$CONFIG" \
        model.ssl="$SSL" optim.epochs="$EPOCHS" meta.ckpt_dir="$CKPT_DIR"
    ;;
  eval)
    export RESULTS_DIR="$REPO/results/pipeline_${TRACK}_${SSL}"
    $PY -m examples.pipeline.cli.evaluate --track "$TRACK" --ckpt "$CK" --sup-epochs 10
    ;;
  *) echo "unknown stage $STAGE"; exit 1 ;;
esac
echo "=== STAGE=$STAGE DONE $(date) ==="
