#!/usr/bin/env bash
# Datasets of spec Sec. 2.6.  Usage: bash scripts/make_datasets.sh [local|hpc|transfer] [workers]
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
  hpc)
    $G --n 10 --ntraj 8000 --L_traj 161 --seed 11 --out datasets/n10/train.h5
    $G --n 10 --ntraj 500  --L_traj 161 --seed 12 --out datasets/n10/val.h5
    $G --n 10 --ntraj 500  --L_traj 192 --seed 13 --out datasets/n10/test.h5
    ;;
  transfer)
    $G --n 12 --ntraj 300 --L_traj 192 --seed 21 --out datasets/n12/test.h5
    $G --n 14 --ntraj 300 --L_traj 192 --seed 22 --out datasets/n14/test.h5
    python -m data.generate --family lr_ising --dt 0.1 --workers $W --n 10 --ntraj 300 \
        --L_traj 192 --seed 23 --out datasets/lr_ising_n10/test.h5
    ;;
  *) echo "unknown target $WHAT"; exit 1 ;;
esac
