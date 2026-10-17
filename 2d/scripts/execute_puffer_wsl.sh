#!/usr/bin/env bash
# Run inside the dedicated distro; all output directories must be new.
set -euo pipefail
test "${WSL_DISTRO_NAME:-}" = Cartpole-Puffer5
cd /home/cartpole/cartpole-lab
export PATH=/usr/local/cuda-12.6/bin:/usr/local/bin:/usr/bin:/bin:/usr/lib/wsl/lib
source .venv/bin/activate
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export CUDA_HOME=/usr/local/cuda-12.6 NVCC_ARCH=sm_75
# Link the SDK's NVML stub; runtime resolves the Windows-provided WSL library.
export LIBRARY_PATH=/usr/local/cuda-12.6/lib64/stubs
if [ "${1:-}" = --evaluate ]; then
    shift
    exec python scripts/run_puffer_evaluation.py "$@"
fi
exec python scripts/run_puffer_training.py "$@"
