#!/bin/bash
#SBATCH --job-name=hlfair10
#SBATCH --output=logs/hlfair10_%j.out
#SBATCH --error=logs/hlfair10_%j.err
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --account=iscrc_manqo-gp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=alessiopecilli@gmail.com
# CPU-only fair-HL baseline at n=10 (5 trajectories, T=64): sbatch scripts/hpc/hlfair_n10.sbatch
module purge; module load python/3.11.7
source "${HBICP_VENV:-/leonardo_work/IscrC_QuSALa/venv_hbicp12}/bin/activate"
cd "${HBICP_CODE:-$SLURM_SUBMIT_DIR}" || exit 1
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
python -m eval.evaluate --data datasets/n10/test.h5 --T 64 --horizon 64 --baselines hl_fair_o0 hl_fair_fit \
    --max_traj 5 --out results/hl_fair_n10
