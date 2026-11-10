#!/bin/bash
#SBATCH --job-name=hbicp_pub
#SBATCH --output=logs/publish_%j.out
#SBATCH --partition=boost_usr_prod
#SBATCH --account=iscrc_manqo-gp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=alessiopecilli@gmail.com
# Builds and commits the dashboard exports (no internet here); then on the login node: python scripts/hpc/publish.py
module purge
module load python/3.11.7
source "${HBICP_VENV:-/leonardo_work/IscrC_QuSALa/venv_hbicp12}/bin/activate"
cd "$SLURM_SUBMIT_DIR" || exit 1
OFFLINE=1 python scripts/hpc/publish.py
