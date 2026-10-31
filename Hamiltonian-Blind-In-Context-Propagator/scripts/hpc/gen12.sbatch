#!/bin/bash
#SBATCH --job-name=gen12
#SBATCH --output=logs/gen12_%j.out
#SBATCH --error=logs/gen12_%j.err
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --account=iscrc_manqo-gp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=15:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=alessiopecilli@gmail.com
# CPU-only n=12 dataset generation; rerunning regenerates from scratch if it was cut (the .ok marker is written at the end).
module purge; module load python/3.11.7
source "${HBICP_VENV:-/leonardo_work/IscrC_QuSALa/venv_hbicp12}/bin/activate"
cd "${HBICP_CODE:-$SLURM_SUBMIT_DIR}" || exit 1
mkdir -p logs state
STAGES=G bash scripts/hpc/pipeline.sh
