#!/bin/bash
#SBATCH --job-name=hbicp
#SBATCH --output=logs/hbicp_%j.out
#SBATCH --error=logs/hbicp_%j.err
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --account=iscrc_manqo-gp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=15:00:00
#SBATCH --signal=B:USR1@600
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=alessiopecilli@gmail.com
# STAGES (env, default "A B C") picks the stages, e.g.: sbatch --export=ALL,STAGES=D scripts/hpc/hbicp.sbatch
# Submit from the code folder (<repo>/Hamiltonian-Blind-In-Context-Propagator):  sbatch scripts/hpc/hbicp.sbatch
# The pipeline is idempotent. 10 min before the time limit SLURM sends USR1: the pipeline is stopped
# (at most ckpt_every=500 steps are lost) and the job resubmits itself, up to MAX_CHAIN times.

module purge
module load python/3.11.7
source "${HBICP_VENV:-/leonardo_work/IscrC_QuSALa/venv_hbicp12}/bin/activate"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4

CODE=${HBICP_CODE:-$SLURM_SUBMIT_DIR}
cd "$CODE" || exit 1
mkdir -p logs state
CHAIN=${CHAIN:-0}; MAX_CHAIN=${MAX_CHAIN:-6}
echo "=== JOB $SLURM_JOB_ID chain=$CHAIN $(date) on $(hostname) ==="
python -c "import torch;print('cuda', torch.cuda.is_available())"

setsid bash scripts/hpc/pipeline.sh &
PID=$!
trap 'echo "USR1: time limit close, stopping pipeline"; kill -TERM -- -$PID 2>/dev/null; wait $PID 2>/dev/null; STOPPED=1' USR1
wait $PID

TAG=$(echo ${STAGES:-A B C} | tr -d " ")
if [ -f state/done_$TAG ] || [ -f state/failed_$TAG ]; then
  echo "pipeline finished (see state/)"; exit 0
fi
if [ "$CHAIN" -lt "$MAX_CHAIN" ]; then
  echo "resubmitting (chain $((CHAIN+1)))"
  sbatch --export=ALL,CHAIN=$((CHAIN+1)),HBICP_CODE="$CODE" scripts/hpc/hbicp.sbatch
else
  echo "MAX_CHAIN reached; resubmit by hand"
fi
