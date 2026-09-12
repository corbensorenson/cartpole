#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_DIR="$ROOT_DIR/.conda-aligator"
cd "$ROOT_DIR"

if ! command -v conda >/dev/null 2>&1; then
  echo "Conda is required for the Aligator environment." >&2
  exit 1
fi

if [[ -x "$ENV_DIR/bin/python" ]]; then
  conda env update --prefix "$ENV_DIR" --file environment-aligator.yml --prune
else
  conda env create --prefix "$ENV_DIR" --file environment-aligator.yml
fi

"$ENV_DIR/bin/python" -m pip install -e .
"$ENV_DIR/bin/python" -c \
  'import aligator, crocoddyl, mujoco; print(aligator.__version__, crocoddyl.__version__, mujoco.__version__)'

echo "Aligator environment ready at $ENV_DIR"
