#!/usr/bin/env bash
set -euo pipefail

WITH_PUFFER="${1:-}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "Warning: this setup script is tuned for macOS. Continuing anyway."
fi

if [[ "$(uname -m)" != "arm64" ]]; then
  echo "Warning: expected native arm64 Apple Silicon. MLX PyPI wheels require native arm64 on macOS."
  echo "Current arch: $(uname -m)"
fi

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip wheel setuptools
python -m pip install -r requirements-mac.txt
python -m pip install -e .

if [[ "$WITH_PUFFER" == "--with-puffer" ]]; then
  python -m pip install -r requirements-pufferlib4.txt
fi

python scripts/smoke_test.py

echo ""
echo "Setup complete. Activate with:"
echo "  source .venv/bin/activate"
