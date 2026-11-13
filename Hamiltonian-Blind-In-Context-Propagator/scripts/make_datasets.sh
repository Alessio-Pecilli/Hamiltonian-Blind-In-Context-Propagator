#!/usr/bin/env bash
# Datasets of spec Sec. 2.6.  Usage: bash scripts/make_datasets.sh [local|hpc|sets|mix|n12|transfer] [workers]
# Note: the spec's HPC config (T up to 128, H = 32) needs L_traj >= T + H + 1 = 161, one more
# than the L_traj = 160 of Sec. 2.1; test sets use 192 so that k can reach 64 at T = 128.
set -euo pipefail
WHAT=${1:-local}
W=${2:-4}
G="python -m data.generate --family tfim --dt 0.1 --workers $W"

case "$WHAT" in
  local)
    $G --n 8 --ntraj 1000 --L_traj 160 --seed 1 --out datasets/n8/train.h5
    $G --n 8 --ntraj 100  --L_traj 160 --seed 2 --out datasets/n8/val.h5
    $G --n 8 --ntraj 100  --L_traj 192 --seed 3 --out datasets/n8/test.h5
    ;;
  local5k)  # larger train set (5000 Hamiltonians), same val/test as local
    $G --n 8 --ntraj 5000 --L_traj 160 --seed 1 --out datasets/n8/train5k.h5
    ;;
  hpc)
    $G --n 10 --ntraj 8000 --L_traj 161 --seed 11 --out datasets/n10/train.h5
    $G --n 10 --ntraj 500  --L_traj 161 --seed 12 --out datasets/n10/val.h5
    $G --n 10 --ntraj 500  --L_traj 192 --seed 13 --out datasets/n10/test.h5
    ;;
  sets)  # n=10, 64 fixed Hamiltonians x 125 initial states; test on 20 new Hamiltonians and on the 64 seen ones (new states)
    $G --n 10 --ntraj 8000 --L_traj 161 --seed 41 --n_sets 64 --set_seed 41 --out datasets/n10sets/train.h5
    $G --n 10 --ntraj 500  --L_traj 161 --seed 42 --n_sets 20 --set_seed 42 --out datasets/n10sets/val.h5
    $G --n 10 --ntraj 500  --L_traj 192 --seed 43 --n_sets 20 --set_seed 43 --out datasets/n10sets/test.h5
    $G --n 10 --ntraj 512  --L_traj 192 --seed 44 --n_sets 64 --set_seed 41 --out datasets/n10sets/test_seen.h5
    ;;
  mix)  # n=10, six families in turn (tfim, xxz, heisenberg, lr_ising and the chains of the older project): 2000 Hamiltonians
        # each in training; one mixed test file plus one test file per family (new Hamiltonians everywhere)
    M="python -m data.generate --dt 0.1 --workers $W --n 10"
    $M --family mix --ntraj ${MIX_TRAIN:-12000} --L_traj 161 --seed 51 --out datasets/n10mix/train.h5
    $M --family mix --ntraj 600 --L_traj 161 --seed 52 --out datasets/n10mix/val.h5
    $M --family mix --ntraj 600 --L_traj 192 --seed 53 --out datasets/n10mix/test.h5
    k=0
    for fam in tfim xxz heisenberg lr_ising chain chain_dis; do
      k=$((k + 1))
      $M --family $fam --ntraj 300 --L_traj 192 --seed $((60 + k)) --out datasets/n10mix/test_$fam.h5
    done
    ;;
  n12)  # sizes overridable: N12_TRAIN (default 3000), CPU-heavy (~1 trajectory / 6-8 s with 8 workers)
    $G --n 12 --ntraj ${N12_TRAIN:-3000} --L_traj 161 --seed 31 --out datasets/n12/train.h5
    $G --n 12 --ntraj 100 --L_traj 161 --seed 32 --out datasets/n12/val.h5
    $G --n 12 --ntraj 300 --L_traj 192 --seed 33 --out datasets/n12/test.h5
    ;;
  transfer)
    $G --n 12 --ntraj 300 --L_traj 192 --seed 21 --out datasets/n12/test.h5
    $G --n 14 --ntraj 300 --L_traj 192 --seed 22 --out datasets/n14/test.h5
    python -m data.generate --family lr_ising --dt 0.1 --workers $W --n 10 --ntraj 300 \
        --L_traj 192 --seed 23 --out datasets/lr_ising_n10/test.h5
    ;;
  *) echo "unknown target $WHAT"; exit 1 ;;
esac
