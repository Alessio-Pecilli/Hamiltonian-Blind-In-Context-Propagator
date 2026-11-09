#!/usr/bin/env bash
# Serial HBICP pipeline for CINECA (SMOKE=1: tiny run to test the logic on a laptop).
# Idempotent: finished stages (results/<name>/results.json) are skipped, interrupted trainings
# resume from runs/<name>/last.pt (written atomically every ckpt_every steps). Safe to re-run any time.
#   A  n8_resid_wide  residual anchor, d=256 L=6 model, 20k steps   (is it capacity?)
#   B  n8_resid_long  residual anchor, local model, 60k steps        (is it training length?)
#   C  n10_resid      residual anchor, hpc model, n=10 qubits        (do the gains appear at larger n?)
#   D  n10_small      residual anchor, SMALL model, n=10, same data/steps as C   (size vs n)
#   E  n12_small      residual anchor, SMALL model, n=12 (needs G first)
#   G  n12 datasets only (CPU, slow)
#   R  n10_small_rel  re-evaluation of n10_small (no training) with the relative metrics
#   F  n10_small_norm as D, with the amplitude-normalised loss (loss.amp_norm=true)
#   V  n10_small_varh as D, trained on 64 future steps with the horizon sampled in [8, 64]
#   S  n10_small_roll as D, roll out of 4 blocks of 16 steps with scheduled sampling, evaluated the same way;
#      plus n10_small_blk32: the plain model of D evaluated as a roll out of 2 blocks of 32 (no retraining)
#   P  n10_sets_roll  as S, but trained on 64 fixed Hamiltonians (125 initial states each) and tested on 20 new
#      ones (and on the 64 seen ones with new states); baselines: in-context DMD and a library of the 64 per-set
#      DMD operators mixed with weights fitted on the context. Plus n10_small_roll (8000 Hamiltonians) on the same test.
# A fair-HL baseline (CPU) runs in the background next to A/B/C.
set -o pipefail
cd "$(dirname "$0")/../.."
export PYTHONIOENCODING=utf-8
MAIL=${MAIL:-alessiopecilli@gmail.com}
W=${SLURM_CPUS_PER_TASK:-4}
STATE=state; mkdir -p "$STATE" runs results logs
STAGES=${STAGES:-"A B C"}
TAG=$(echo $STAGES | tr -d " ")   # one done/failed marker per set of stages
rm -f "$STATE/failed_$TAG"
RES=(model.residual=true)
SMALL=(model.d=128 model.de=32 model.nh=4 model.L=3 model.m=32 model.K=32)
WIDE=(model.d=256 model.de=64 model.nh=8 model.L=6 model.m=64 model.K=64)
if [ "${SMOKE:-0}" = 1 ]; then
  MAIL=""; STEPS_A=${STEPS_A:-20}; STEPS_B=${STEPS_B:-30}; STEPS_C=${STEPS_C:-10}; STEPS_D=${STEPS_D:-10}; STEPS_E=${STEPS_E:-10}; STEPS_F=${STEPS_F:-10}; STEPS_V=${STEPS_V:-10}; STEPS_S=${STEPS_S:-10}; STEPS_P=${STEPS_P:-10}
  CK=(train.ckpt_every=10 train.eval_every=1000000 train.batch_size=2 train.T_max=32 train.H=8); EV=(--max_traj 2 --batch_size 2); HL_N=1; HL_T="16"
else
  STEPS_A=${STEPS_A:-20000}; STEPS_B=${STEPS_B:-60000}; STEPS_C=${STEPS_C:-40000}; STEPS_D=${STEPS_D:-40000}; STEPS_E=${STEPS_E:-40000}; STEPS_F=${STEPS_F:-40000}; STEPS_V=${STEPS_V:-40000}; STEPS_S=${STEPS_S:-40000}; STEPS_P=${STEPS_P:-40000}
  CK=(); EV=(); HL_N=${HL_N:-20}; HL_T="16 32 64"
fi

notify() {
  echo "[notify] $*"
  if [ -n "$MAIL" ] && command -v mail >/dev/null 2>&1; then echo "$*" | mail -s "HBICP: $*" "$MAIL" 2>/dev/null || true; fi
}
FAILED=0
EVX=()  # extra eval arguments of the next stage (e.g. --block)
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
  python -m eval.evaluate --ckpt "$d/last.pt" --data "$test" --T $Ts --horizon 64 --baselines $bl "${EV[@]}" "${EVX[@]}" --out "$res" 2>&1 | tee -a "$d/eval.log"
  [ "${PIPESTATUS[0]}" -eq 0 ] || { fail "$name eval"; return 1; }
  notify "$name finished ($steps steps)"
}

reeval() {  # reeval <run> <out> <test.h5> "<T list>" "<baselines>": eval only, of a finished run
  local res=results/$2
  if [ -f "$res/results.json" ]; then echo "[$2] already done, skip"; return 0; fi
  [ -f "runs/$1/last.pt" ] || { fail "$2: runs/$1/last.pt missing"; return 1; }
  echo "[$2] $(date) eval of runs/$1"
  python -m eval.evaluate --ckpt "runs/$1/last.pt" --data "$3" --T $4 --horizon 64 --baselines $5 "${EV[@]}" "${EVX[@]}" --out "$res" 2>&1 | tee -a "runs/$1/eval_rel.log"
  [ "${PIPESTATUS[0]}" -eq 0 ] || { fail "$2 eval"; return 1; }
  notify "$2 finished"
}

n10data() {
  if [ "${SMOKE:-0}" = 1 ]; then
    [ -f datasets/n10/.ok ] || { for sp in train:1 val:2 test:3; do python -m data.generate --family tfim --dt 0.1 --workers 2 --n 10 --ntraj 4 --L_traj 193 --seed ${sp#*:} --out datasets/n10/${sp%:*}.h5 || return 1; done; touch datasets/n10/.ok; }
  else make_data hpc datasets/n10 "$W"; fi
}

diag() {  # k* on train vs test windows (n=8 only)
  local name=$1 split out=results/$1_traintest
  for split in train test; do
    [ -f "$out/$split/results.json" ] || python -m eval.evaluate --ckpt "runs/$name/last.pt" --data "datasets/n8/$split.h5" \
      --T 8 16 32 64 --horizon 64 --baselines --max_traj 64 --batch_size 16 --out "$out/$split" >> "runs/$name/eval.log" 2>&1
  done
}

gen12() {  # n=12 data (slow on CPU); skipped when already generated
  if [ "${SMOKE:-0}" = 1 ]; then
    [ -f datasets/n12/.ok ] && return 0
    for sp in train:1 val:2 test:3; do python -m data.generate --family tfim --dt 0.1 --workers 2 --n 12 --ntraj 4 --L_traj 193 --seed ${sp#*:} --out datasets/n12/${sp%:*}.h5 || return 1; done
    touch datasets/n12/.ok
  else make_data n12 datasets/n12 "$W"; fi
}

echo "=== pipeline start $(date) stages=$STAGES smoke=${SMOKE:-0} ==="
python -m pytest -q tests 2>&1 | tail -3

# fair HL baseline (CPU) in the background; independent of the models
HLPID=""
case " $STAGES " in *" A "*|*" B "*)
make_data local datasets/n8 "$W" || { fail "n8 datasets"; exit 1; }
if [ ! -f results/hl_fair/results.json ]; then
  ( OMP_NUM_THREADS=2 python -m eval.evaluate --data datasets/n8/test.h5 --T $HL_T --horizon 64 \
      --baselines hl_fair_o0 hl_fair_fit --max_traj "$HL_N" --out results/hl_fair > logs/hl_fair.log 2>&1 \
    && notify "hl_fair finished" || fail "hl_fair" ) &
  HLPID=$!
fi

esac

for s in $STAGES; do
  case $s in
    A) stage n8_resid_wide local "$STEPS_A" datasets/n8/test.h5 "8 16 32 64 96 127" "persistence dmd dmd4 oracle hl_oracle" "${RES[@]}" "${WIDE[@]}" && diag n8_resid_wide ;;
    B) stage n8_resid_long local "$STEPS_B" datasets/n8/test.h5 "8 16 32 64 96 127" "persistence dmd dmd4 oracle hl_oracle" "${RES[@]}" && diag n8_resid_long ;;
    C)
      if [ "${SMOKE:-0}" = 1 ]; then
        [ -f datasets/n10/.ok ] || { for sp in train:1 val:2 test:3; do python -m data.generate --family tfim --dt 0.1 --workers 2 --n 10 --ntraj 4 --L_traj 193 --seed ${sp#*:} --out datasets/n10/${sp%:*}.h5 || fail "n10 smoke data"; done; touch datasets/n10/.ok; }
      else make_data hpc datasets/n10 "$W" || { fail "n10 datasets"; continue; }; fi
      stage n10_resid hpc "$STEPS_C" datasets/n10/test.h5 "8 16 32 64 128" "persistence dmd dmd4 oracle" "${RES[@]}" ;;
    D)
      if [ "${SMOKE:-0}" = 1 ]; then
        [ -f datasets/n10/.ok ] || { for sp in train:1 val:2 test:3; do python -m data.generate --family tfim --dt 0.1 --workers 2 --n 10 --ntraj 4 --L_traj 193 --seed ${sp#*:} --out datasets/n10/${sp%:*}.h5 || fail "n10 smoke data"; done; touch datasets/n10/.ok; }
      else make_data hpc datasets/n10 "$W" || { fail "n10 datasets"; continue; }; fi
      stage n10_small hpc "$STEPS_D" datasets/n10/test.h5 "8 16 32 64 128" "persistence dmd dmd4 oracle" "${RES[@]}" "${SMALL[@]}" ;;
    G) gen12 || fail "n12 datasets" ;;
    R)
      n10data || { fail "n10 datasets"; continue; }
      reeval n10_small n10_small_rel datasets/n10/test.h5 "8 16 32 64 128" "persistence dmd dmd4 oracle" ;;
    F)
      n10data || { fail "n10 datasets"; continue; }
      stage n10_small_norm hpc "$STEPS_F" datasets/n10/test.h5 "8 16 32 64 128" "persistence dmd dmd4 oracle" "${RES[@]}" "${SMALL[@]}" loss.amp_norm=true ;;
    V)
      n10data || { fail "n10 datasets"; continue; }
      stage n10_small_varh hpc "$STEPS_V" datasets/n10/test.h5 "8 16 32 64 128" "persistence dmd dmd4 oracle" "${RES[@]}" "${SMALL[@]}"         train.var_horizon=true train.H=64 train.H_min=8 train.T_max=96 ;;  # train trajectories are 161 long: T_max + 64 must fit
    S)
      n10data || { fail "n10 datasets"; continue; }
      EVX=(--block 32); reeval n10_small n10_small_blk32 datasets/n10/test.h5 "8 16 32 64 128" "persistence dmd dmd4 oracle"
      EVX=(--block 16); stage n10_small_roll hpc "$STEPS_S" datasets/n10/test.h5 "8 16 32 64 128" "persistence dmd dmd4 oracle" "${RES[@]}" "${SMALL[@]}"         train.H=16 train.rollout_blocks=4 train.T_max=96
      EVX=() ;;
    P)
      if [ "${SMOKE:-0}" = 1 ]; then
        [ -f datasets/n10sets/.ok ] || { for sp in train:161:1:4 val:161:2:2 test:192:3:2 test_seen:192:4:4; do IFS=: read f L s k <<< "$sp"
          python -m data.generate --dt 0.1 --n 10 --ntraj 8 --L_traj $L --seed $s --n_sets $k --set_seed $([ $f = test_seen ] && echo 1 || echo $s) --out datasets/n10sets/$f.h5; done; touch datasets/n10sets/.ok; } || { fail "sets smoke data"; continue; }
      else make_data sets datasets/n10sets "$W" || { fail "n10sets datasets"; continue; }; fi
      [ -f datasets/n10sets/dmd_lib.npy ] || python -m eval.baselines --train datasets/n10sets/train.h5 --out datasets/n10sets/dmd_lib.npy || { fail "dmd lib"; continue; }
      export HBICP_DMD_LIB=$PWD/datasets/n10sets/dmd_lib.npy
      D10S=(data.train=datasets/n10sets/train.h5 data.val=datasets/n10sets/val.h5 data.test=datasets/n10sets/test.h5)
      EVX=(--block 16)
      stage n10_sets_roll hpc "$STEPS_P" datasets/n10sets/test.h5 "8 16 32 64 128" "persistence dmd dmd_lib" "${RES[@]}" "${SMALL[@]}"         train.H=16 train.rollout_blocks=4 train.T_max=96 "${D10S[@]}"         && reeval n10_sets_roll n10_sets_roll_seen datasets/n10sets/test_seen.h5 "8 16 32 64 128" "persistence dmd dmd_lib"
      [ -f runs/n10_small_roll/last.pt ] && reeval n10_small_roll n10_small_roll_on_sets datasets/n10sets/test.h5 "8 16 32 64 128" ""
      EVX=() ;;
    E)
      gen12 || { fail "n12 datasets"; continue; }
      stage n12_small hpc "$STEPS_E" datasets/n12/test.h5 "8 16 32 64 128" "persistence dmd dmd4" "${RES[@]}" "${SMALL[@]}"         data.train=datasets/n12/train.h5 data.val=datasets/n12/val.h5 data.test=datasets/n12/test.h5 ;;
  esac
done

[ -n "$HLPID" ] && wait "$HLPID"
if [ "$FAILED" = 0 ]; then touch "$STATE/done_$TAG"; notify "ALL DONE ($STAGES)"; else touch "$STATE/failed_$TAG"; fi
echo "=== pipeline end $(date) failed=$FAILED ==="
exit $FAILED
