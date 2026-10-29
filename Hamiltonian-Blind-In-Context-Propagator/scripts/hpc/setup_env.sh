#!/usr/bin/env bash
# Run ONCE on the CINECA login node (compute nodes have no internet).
#   bash setup_env.sh            # venv at $HBICP_VENV
set -e
module purge
module load python/3.11.7
V=${HBICP_VENV:-/leonardo_work/IscrC_QuSALa/venv_hbicp12}
[ -d "$V" ] || python3 -m venv "$V"
source "$V/bin/activate"
pip install -U pip
# Leonardo driver is CUDA 12.2: the default (cu13) torch wheel does not run, use cu121
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r "$(dirname "$0")/../../requirements.txt"
python -c "import torch,h5py,scipy,yaml;print('torch',torch.__version__,'cuda build',torch.version.cuda)"
echo "venv ready: $V   (export HBICP_VENV=$V before sbatch if you use a different path)"
