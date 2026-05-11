#!/usr/bin/env bash
# start.sh — set up and launch the local-doc-chat
#
# Drop your files into my-docs/, then:
#
#   First run / after adding files:   ./start.sh --reindex
#   Resume existing session:          ./start.sh
#   Different folder:                 ./start.sh --dir ~/my-notes --reindex
#   Named session:                    ./start.sh --session work
#
# All extra args are forwarded to main.py (default --dir is ./my-docs)

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

# ── 3. Check LM Studio; set up cloud fallback if needed ───────────────────────
BASE_URL="${LM_STUDIO_URL:-http://localhost:1234}"
LM_AVAILABLE=false

if curl -sf --max-time 2 "${BASE_URL}/api/v1/models" -o /dev/null 2>/dev/null; then
  LM_AVAILABLE=true
  echo "LM Studio: connected at ${BASE_URL}"
fi

if [ "$LM_AVAILABLE" = false ]; then
  echo ""
  echo "LM Studio is not reachable at ${BASE_URL}."

  # ── Create .env from example if missing ──
  if [ ! -f .env ]; then
    cp .env.example .env
    echo ""
    echo "  Created .env for you. Open it and add a cloud API key, then rerun."
    echo ""
    echo "  Edit the file:"
    echo "    nano .env"
    echo "    # or: code .env  |  vim .env  |  open -e .env"
    echo ""
    echo "  Uncomment ONE of these lines and fill in your key:"
    echo "    GOOGLE_API_KEY=AIza...   # aistudio.google.com/apikey"
    echo "    OPENAI_API_KEY=sk-..."
    echo ""
    echo "  Then rerun:  ./start.sh $*"
    exit 0
  fi

  # ── .env exists — check whether a key is actually set ──
  HAS_KEY=false
  if grep -qE "^OPENAI_API_KEY=.+" .env 2>/dev/null || \
     grep -qE "^GOOGLE_API_KEY=.+" .env 2>/dev/null; then
    HAS_KEY=true
  fi

  if [ "$HAS_KEY" = false ]; then
    echo ""
    echo "  .env exists but no API key is set."
    echo ""
    echo "  Open .env and uncomment + fill in one of:"
    echo "    GOOGLE_API_KEY=AIza...   # aistudio.google.com/apikey"
    echo "    OPENAI_API_KEY=sk-..."
    echo ""
    echo "  Then rerun:  ./start.sh $*"
    exit 0
  fi

  echo "  Cloud fallback active — key found in .env."
  echo ""
fi

# ── 4. Run ────────────────────────────────────────────────────────────────────
echo "Starting local-doc-chat..."
echo ""
uv run python main.py "$@"
