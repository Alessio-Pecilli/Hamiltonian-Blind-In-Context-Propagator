#!/usr/bin/env bash
# Serial HBICP pipeline for CINECA (SMOKE=1: tiny run to test the logic on a laptop).
# Idempotent: finished stages (results/<name>/results.json) are skipped, interrupted trainings
# resume from runs/<name>/last.pt (written atomically every ckpt_every steps). Safe to re-run any time.
#   A  n8_resid_wide  residual anchor, d=256 L=6 model, 20k steps   (is it capacity?)
#   B  n8_resid_long  residual anchor, local model, 60k steps        (is it training length?)
#   C  n10_resid      residual anchor, hpc model, n=10 qubits        (do the gains appear at larger n?)
# A fair-HL baseline (CPU) runs in the background next to A/B/C.
set -o pipefail
cd "$(dirname "$0")/../.."
export PYTHONIOENCODING=utf-8
MAIL=${MAIL:-alessiopecilli@gmail.com}
W=${SLURM_CPUS_PER_TASK:-4}
STATE=state; mkdir -p "$STATE" runs results logs
STAGES=${STAGES:-"A B C"}
RES=(model.residual=true)
WIDE=(model.d=256 model.de=64 model.nh=8 model.L=6 model.m=64 model.K=64)
if [ "${SMOKE:-0}" = 1 ]; then
  MAIL=""; STEPS_A=${STEPS_A:-20}; STEPS_B=${STEPS_B:-30}; STEPS_C=${STEPS_C:-10}
  CK=(train.ckpt_every=10 train.eval_every=1000000 train.batch_size=2 train.T_max=32 train.H=8); EV=(--max_traj 2 --batch_size 2); HL_N=1; HL_T="16"
else
  STEPS_A=${STEPS_A:-20000}; STEPS_B=${STEPS_B:-60000}; STEPS_C=${STEPS_C:-40000}
  CK=(); EV=(); HL_N=${HL_N:-20}; HL_T="16 32 64"
fi

notify() {
  echo "[notify] $*"
  if [ -n "$MAIL" ] && command -v mail >/dev/null 2>&1; then echo "$*" | mail -s "HBICP: $*" "$MAIL" 2>/dev/null || true; fi
}
FAILED=0
fail() { echo "[FAIL] $*"; notify "FAIL $*"; FAILED=1; }

cur_step() {  # step stored in <dir>/last.pt; a corrupt checkpoint is set aside and counts as 0
  local p=$1/last.pt
  [ -f "$p" ] || { echo 0; return; }
  python -c "import sys,torch;print(torch.load(sys.argv[1],map_location='cpu',weights_only=False)['step'])" "$p" 2>/dev/null \
    || { mv "$p" "$p.bad"; echo 0; }
}

make_data() {  # make_data <target> <dir> <workers>
  [ -f "$2/.ok" ] && return 0
  bash scripts/make_datasets.sh "$1" "$3" && touch "$2/.ok"
}

stage() {  # stage <name> <preset> <steps> <test.h5> "<T list>" "<baselines>" [overrides...]
  local name=$1 preset=$2 steps=$3 test=$4 Ts=$5 bl=$6; shift 6
  local d=runs/$name res=results/$name s
  if [ -f "$res/results.json" ]; then echo "[$name] already done, skip"; return 0; fi
  mkdir -p "$d"
  s=$(cur_step "$d")
  if [ "$s" -lt "$steps" ]; then
    local a=(train.py --preset "$preset" --set "train.steps=$steps" train.seed=0 "train.out_dir=$d" "${CK[@]}" "$@")
    [ -f "$d/last.pt" ] && a+=(--resume "$d/last.pt")
    echo "[$name] $(date) train from step $s to $steps"
    python "${a[@]}" 2>&1 | tee -a "$d/train.log"
    [ "${PIPESTATUS[0]}" -eq 0 ] || { fail "$name train"; return 1; }
  fi
  s=$(cur_step "$d")
  [ "$s" -ge "$steps" ] || { fail "$name stopped at step $s < $steps"; return 1; }
  echo "[$name] $(date) eval"
  python -m eval.evaluate --ckpt "$d/last.pt" --data "$test" --T $Ts --horizon 64 --baselines $bl "${EV[@]}" --out "$res" 2>&1 | tee -a "$d/eval.log"
  [ "${PIPESTATUS[0]}" -eq 0 ] || { fail "$name eval"; return 1; }
  notify "$name finished ($steps steps)"
}

diag() {  # k* on train vs test windows (n=8 only)
  local name=$1 split out=results/$1_traintest
  for split in train test; do
    [ -f "$out/$split/results.json" ] || python -m eval.evaluate --ckpt "runs/$name/last.pt" --data "datasets/n8/$split.h5" \
      --T 8 16 32 64 --horizon 64 --baselines --max_traj 64 --batch_size 16 --out "$out/$split" >> "runs/$name/eval.log" 2>&1
  done
}

echo "=== pipeline start $(date) stages=$STAGES smoke=${SMOKE:-0} ==="
python -m pytest -q tests 2>&1 | tail -3

# fair HL baseline (CPU) in the background; independent of the models
HLPID=""
make_data local datasets/n8 "$W" || { fail "n8 datasets"; exit 1; }
if [ ! -f results/hl_fair/results.json ]; then
  ( OMP_NUM_THREADS=2 python -m eval.evaluate --data datasets/n8/test.h5 --T $HL_T --horizon 64 \
      --baselines hl_fair_o0 hl_fair_fit --max_traj "$HL_N" --out results/hl_fair > logs/hl_fair.log 2>&1 \
    && notify "hl_fair finished" || fail "hl_fair" ) &
  HLPID=$!
fi

for s in $STAGES; do
  case $s in
    A) stage n8_resid_wide local "$STEPS_A" datasets/n8/test.h5 "8 16 32 64 96 127" "persistence dmd dmd4 oracle hl_oracle" "${RES[@]}" "${WIDE[@]}" && diag n8_resid_wide ;;
    B) stage n8_resid_long local "$STEPS_B" datasets/n8/test.h5 "8 16 32 64 96 127" "persistence dmd dmd4 oracle hl_oracle" "${RES[@]}" && diag n8_resid_long ;;
    C)
      if [ "${SMOKE:-0}" = 1 ]; then
        [ -f datasets/n10/.ok ] || { for sp in train:1 val:2 test:3; do python -m data.generate --family tfim --dt 0.1 --workers 2 --n 10 --ntraj 4 --L_traj 193 --seed ${sp#*:} --out datasets/n10/${sp%:*}.h5 || fail "n10 smoke data"; done; touch datasets/n10/.ok; }
      else make_data hpc datasets/n10 "$W" || { fail "n10 datasets"; continue; }; fi
      stage n10_resid hpc "$STEPS_C" datasets/n10/test.h5 "8 16 32 64 128" "persistence dmd dmd4 oracle" "${RES[@]}" ;;
  esac
done

[ -n "$HLPID" ] && wait "$HLPID"
if [ "$FAILED" = 0 ]; then touch "$STATE/ALL_DONE"; notify "ALL DONE ($STAGES)"; else touch "$STATE/FAILED"; fi
echo "=== pipeline end $(date) failed=$FAILED ==="
exit $FAILED
