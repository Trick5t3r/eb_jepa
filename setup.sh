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

# 1. Create required directories in the work partition
mkdir -p "$UV_INSTALL_DIR" "$UV_CACHE_DIR" "$WORK/venvs" \
         "$WORK/checkpoints" "$WORK/logs"

# 2. Install uv for the current arch if not already present
if ! "$UV_INSTALL_DIR/uv" --version &>/dev/null; then
    echo ">>> Installing uv for $ARCH..."
    curl -LsSf https://astral.sh/uv/install.sh | UV_INSTALL_DIR="$UV_INSTALL_DIR" sh
    echo "    uv installed: $("$UV_INSTALL_DIR/uv" --version)"
else
    echo ">>> uv already installed: $(uv --version)"
fi

# 3. Pin Python version and install dependencies
echo ""
echo ">>> Running uv sync (this downloads wheels — may take a few minutes)..."
cd "$REPO_ROOT"
uv sync --dev

# 4. W&B login
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
