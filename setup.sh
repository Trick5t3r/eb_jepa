#!/usr/bin/env bash
# One-shot setup script for EB-JEPA on the HTW cluster.
# Run once from the repo root: bash setup.sh
set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$REPO_ROOT/env.sh"

echo "=== EB-JEPA cluster setup ==="
echo "    Arch   : $ARCH"
echo "    Home   : $HOME"
echo "    Work   : $WORK"
echo "    venv   : $UV_PROJECT_ENVIRONMENT"
echo "    cache  : $UV_CACHE_DIR"
echo ""

# 1. Make cluster scripts executable and ensure they're in PATH via env.sh
chmod +x "$REPO_ROOT"/cluster/{sq,qall,log,gpus,users}

# 2. Create required directories in the work partition
mkdir -p "$UV_INSTALL_DIR" "$UV_CACHE_DIR" "$WORK/venvs" \
         "$WORK/checkpoints" "$WORK/logs"

# 3. Install uv for the current arch if not already present
if ! "$UV_INSTALL_DIR/uv" --version &>/dev/null; then
    echo ">>> Installing uv for $ARCH..."
    curl -LsSf https://astral.sh/uv/install.sh | UV_INSTALL_DIR="$UV_INSTALL_DIR" sh
    echo "    uv installed: $("$UV_INSTALL_DIR/uv" --version)"
else
    echo ">>> uv already installed: $(uv --version)"
fi

# 4. Pin Python version and install dependencies for the current arch
echo ""
echo ">>> Running uv sync for $ARCH (this downloads wheels — may take a few minutes)..."
cd "$REPO_ROOT"
uv sync --dev

# 5. Sync the aarch64 compute-node venv (only needed when running from the x86_64 login node)
#    Submits a short SLURM job so the aarch64 venv gets torch+cu128 instead of torch+cpu.
if [[ "$ARCH" == "x86_64" ]]; then
    COMPUTE_ARCH="${EBJEPA_COMPUTE_ARCH:-aarch64}"
    COMPUTE_UV_DIR="$WORK/uv_bin/$COMPUTE_ARCH"
    mkdir -p "$COMPUTE_UV_DIR" "$WORK/uv_cache/$COMPUTE_ARCH"
    echo ""
    echo ">>> Submitting $COMPUTE_ARCH venv sync job to SLURM..."
    SYNC_JOB=$(sbatch \
        --partition=defq --account=pdl17890 \
        --nodes=1 --ntasks=1 --cpus-per-task=4 \
        --time=0:30:0 --job-name=eb_jepa_setup \
        --output="$WORK/logs/setup_${COMPUTE_ARCH}_%j.out" \
        --parsable \
        --wrap="set -e
source $REPO_ROOT/env.sh
if ! \$UV_INSTALL_DIR/uv --version &>/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | UV_INSTALL_DIR=\$UV_INSTALL_DIR sh
fi
cd $REPO_ROOT && uv sync --dev
echo '$COMPUTE_ARCH venv ready: '\$UV_PROJECT_ENVIRONMENT")
    echo "    Job $SYNC_JOB submitted — monitor with: log $SYNC_JOB"
fi

# 6. W&B login
echo ""
echo ">>> W&B setup"
if grep -q "api.wandb.ai" "$HOME/.netrc" 2>/dev/null; then
    echo "    W&B already configured in ~/.netrc"
else
    echo -n "    Enter your W&B API key (leave blank to skip): "
    read -r WANDB_KEY
    if [ -n "$WANDB_KEY" ]; then
        uv run wandb login "$WANDB_KEY"
        echo "    W&B key saved to ~/.netrc"
    else
        echo "    Skipped. To enable later: uv run wandb login <key>"
        echo "    To disable cluster-wide: export WANDB_DISABLED=true before sourcing env.sh"
    fi
fi

echo ""
echo "=== Setup complete ==="
echo ""
echo "Add these lines to your ~/.bashrc for persistent configuration:"
echo ""
echo "  # EB-JEPA"
echo "  source $REPO_ROOT/env.sh"
echo ""
echo "Then run: source ~/.bashrc"
echo ""
echo "To verify: uv run pytest tests/ -v"
echo ""
echo "To disable W&B logging: export WANDB_DISABLED=true  (before sourcing env.sh)"
echo "To re-enable:            export WANDB_DISABLED=false (before sourcing env.sh)"
