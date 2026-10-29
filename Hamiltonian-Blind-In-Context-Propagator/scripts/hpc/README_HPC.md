# HBICP on CINECA (Leonardo)

Three serial tests, requested after the professor's feedback (run order A, B, C; one job, auto-resume):

| Stage | Run | What it tests |
|---|---|---|
| A | `n8_resid_wide` (d=256, L=6, m=64, 20k steps, n=8) | is the limit the model capacity? |
| B | `n8_resid_long` (local model, 60k steps, n=8) | is the limit the training length? |
| C | `n10_resid` (hpc preset, 40k steps, n=10) | do the gains show up on larger systems? |

Plus the corrected fair-HL baseline (`results/hl_fair`, CPU, in the background). Each stage ends with
evaluation (`results/<name>/results.json`, T up to 127 / 128, horizon 64) and, for n=8, the train-vs-test k* diagnostic
(`results/<name>_traintest/`).

## One-time setup (login node)

```bash
cd /leonardo_work/IscrC_QuSALa
git clone https://github.com/Alessio-Pecilli/Hamiltonian-Blind-In-Context-Propagator.git hbicp
cd hbicp/Hamiltonian-Blind-In-Context-Propagator
bash scripts/hpc/setup_env.sh            # creates the venv, installs requirements (needs internet: login node only)
export HBICP_VENV=/leonardo_work/IscrC_QuSALa/venv_hbicp12
```

## Test the logic first (login node, 2 minutes, no GPU)

```bash
SMOKE=1 STAGES="A" bash scripts/hpc/pipeline.sh     # tiny steps; then delete runs/ results/ state/ datasets/
```

## Run

```bash
sbatch scripts/hpc/hbicp.sbatch           # from this folder
squeue -u $USER
tail -f logs/hbicp_<jobid>.out runs/n8_resid_wide/train.log
```

- Mail to alessiopecilli@gmail.com at the job end/failure (SLURM `--mail-type`) and, if `mail` exists, a message at each stage end.
- Checkpoints: `runs/<name>/last.pt` every 500 steps, written atomically (tmp file + rename). Ten minutes before the 15 h limit
  SLURM sends USR1, the pipeline stops and the job resubmits itself (max 6 times). If it does not, just `sbatch` again: finished
  stages are skipped and trainings resume from `last.pt`.
- `STAGES="A B"` (env) to skip C, `STEPS_A=...` to change lengths.
- Marker files: `state/ALL_DONE` (everything ok) or `state/FAILED` (a stage failed, see `logs/` and `runs/<name>/train.log`).

## Publish (login node, needs internet)

Token (once, typed by you, not echoed, never committed): `umask 077; read -rs T; printf %s "$T" > ~/.hbicp_gh_token; unset T`

```bash
python scripts/hpc/publish.py      # n8_resid_wide / n8_resid_long to the dashboard
```
n10 results are in `results/n10_resid/results.json` and are not published by this script.

## Things to check on the cluster (not verifiable from outside)
- `--gres=gpu:1`, `--cpus-per-task=8`, partition/qos/account copied from the older project (`iscrc_manqo-gp`; `iscrc_qusala` expired on 2026-08-25); adapt if the budget differs.
- `module load python/3.11.7` and a venv with a CUDA torch build (setup_env.sh installs the pip wheel).
- That compute nodes can call `sbatch` (self-resubmission) and that `--mail-type` is enabled.
