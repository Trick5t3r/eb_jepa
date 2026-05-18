#!/usr/bin/env bash
# Source this file to set up the EB-JEPA environment variables.
# Usage: source env.sh
# In SLURM scripts: source "$(dirname "$0")/../env.sh"  (adjust path as needed)

# Work partition — override by setting EBJEPA_WORK before sourcing
WORK=${EBJEPA_WORK:-/lustre/work/pdl17890/udl806719}
ARCH=$(uname -m)   # x86_64 on login node, aarch64 on compute nodes

# uv binary (arch-specific, avoids Exec format error across node types)
export UV_INSTALL_DIR=$WORK/uv_bin/$ARCH
export PATH="$UV_INSTALL_DIR:$HOME/.local/bin:$PATH"

# uv cache and venv (arch-specific so x86_64 and aarch64 don't collide)
export UV_CACHE_DIR=$WORK/uv_cache/$ARCH
export UV_PROJECT_ENVIRONMENT=$WORK/venvs/eb_jepa_$ARCH

# EB-JEPA paths
export EBJEPA_CKPTS=${EBJEPA_CKPTS:-$WORK/checkpoints}
# export EBJEPA_DSETS=$WORK/datasets   # uncomment once datasets are downloaded
