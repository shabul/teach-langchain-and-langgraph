#!/usr/bin/env bash
# start.sh — set up and launch the local-doc-chat
#
# First run:   ./start.sh --dir ../.. --reindex
# Resume:      ./start.sh --dir ../..
# Custom:      ./start.sh --dir ~/my-notes --session notes --reindex
#
# All extra args are forwarded to main.py

set -euo pipefail

# ── 0. Move to this script's directory ────────────────────────────────────────
cd "$(dirname "$0")"

# ── 1. Ensure uv is installed ─────────────────────────────────────────────────
if ! command -v uv &>/dev/null; then
  echo "uv not found — installing via official installer..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  # Add uv to PATH for the rest of this script
  export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
fi

echo "uv $(uv --version)"

# ── 2. Sync dependencies (creates .venv if needed) ────────────────────────────
echo "Syncing dependencies..."
uv sync --quiet

# ── 3. Check LM Studio is reachable ───────────────────────────────────────────
BASE_URL="${LM_STUDIO_URL:-http://localhost:1234}"
if ! curl -sf --max-time 2 "${BASE_URL}/api/v1/models" -o /dev/null 2>/dev/null; then
  echo ""
  echo "⚠  WARNING: LM Studio server not reachable at ${BASE_URL}"
  echo "   Start LM Studio → load a model → enable the REST API server."
  echo "   Continuing anyway — you can still index files now and chat once it's up."
  echo ""
fi

# ── 4. Run ────────────────────────────────────────────────────────────────────
echo "Starting local-doc-chat..."
echo ""
uv run python main.py "$@"
