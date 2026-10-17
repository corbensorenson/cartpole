#!/usr/bin/env bash
# Run as root only inside the dedicated Cartpole-Puffer5 Ubuntu distribution.
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
test "$(id -u)" = 0
test "${WSL_DISTRO_NAME:-}" = Cartpole-Puffer5
test -e /usr/lib/wsl/lib/libcuda.so.1
mkdir -p /var/log/cartpole-setup
cd /var/log/cartpole-setup
apt-get update > apt-update.log 2>&1
apt-get install -y --no-install-recommends build-essential clang ccache pkg-config \
  curl ca-certificates gnupg git unzip libomp-dev libglfw3-dev libgl1-mesa-dev \
  libgl1-mesa-dri xvfb xauth cmake python3-dev python3-venv ffmpeg > apt-base.log 2>&1
curl -fL --retry 3 -o cuda-keyring_1.1-1_all.deb \
  https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sha256sum cuda-keyring_1.1-1_all.deb > cuda-keyring.sha256
dpkg -i cuda-keyring_1.1-1_all.deb
apt-get update > apt-cuda-update.log 2>&1
packages=(cuda-nvcc-12-6 cuda-cudart-dev-12-6 cuda-nvml-dev-12-6 cuda-profiler-api-12-6
          cuda-nvtx-12-6 libcublas-dev-12-6 libcurand-dev-12-6 libcusolver-dev-12-6
          libcusparse-12-6 libnvjitlink-12-6)
# Keep the GPU driver supplied by Windows. Only toolkit/libraries go in WSL.
apt-get --simulate install --no-install-recommends "${packages[@]}" > cuda-install-plan.log
if grep -E '^Inst (nvidia-driver|nvidia-dkms|cuda-drivers|linux-modules-nvidia)' cuda-install-plan.log; then
  echo 'Unexpected Linux GPU driver dependency' >&2; exit 1
fi
apt-get install -y --no-install-recommends "${packages[@]}" > apt-cuda.log 2>&1
apt-cache madison libnccl-dev > nccl-available.txt
version=$(awk '/cuda12.6/ {print $3; exit}' nccl-available.txt)
if [ -z "$version" ]; then echo 'No matching CUDA12.6 NCCL package' >&2; exit 1; fi
apt-get install -y --no-install-recommends "libnccl2=$version" "libnccl-dev=$version" > apt-nccl.log 2>&1
if ! id cartpole >/dev/null 2>&1; then useradd --create-home --shell /bin/bash cartpole; fi
mkdir -p /home/cartpole/cartpole-lab
cp -a /mnt/c/Users/corbe/Documents/ChatGPT/cartpole/{src,scripts,native,research,pyproject.toml,requirements-lock.txt} /home/cartpole/cartpole-lab/
chown -R cartpole:cartpole /home/cartpole/cartpole-lab
dpkg-query -W > packages.tsv
/usr/local/cuda-12.6/bin/nvcc --version
/usr/lib/wsl/lib/nvidia-smi
echo 'Linux toolkit and lab source setup complete'
