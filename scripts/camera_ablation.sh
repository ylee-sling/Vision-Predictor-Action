#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
#
# Single-camera vs dual-camera ablation for the paper.
#
# Every run uses identical settings except --camera-keys:
#     1cam: --camera-keys agentview_rgb                    (the paper's single frame I_t)
#     2cam: --camera-keys agentview_rgb eye_in_hand_rgb    (third-person + wrist view)
# for each training seed in SEEDS, then the same evaluation protocol for every checkpoint
# (TRIALS episodes per task = all of LIBERO's initial states at 50, the same evaluation seed), the
# Eq. 17 latency of both encoders, and a summary table (scripts/summarize_ablation.py).
#
# Finished steps are skipped when the script is run again (a run with policy_final.pt is not
# retrained; an evaluation whose eval_metrics.json says "complete" is not repeated), so it can be
# restarted after an interruption. An interrupted training run starts again from scratch; to
# continue it instead, run train.py --resume <its latest checkpoint> by hand first.
#
# Relative paths (DATA, RUNS, RESULTS) are relative to the folder you run the script from.
#
# Usage (from anywhere):
#     DATA=/path/to/LIBERO/datasets/libero_10 bash scripts/camera_ablation.sh
#     DATA=... SEEDS="0" TRIALS=20 bash scripts/camera_ablation.sh          # a quick first pass
#     DATA=... TRAIN_ARGS="--init-encoder dinov2-small --encoder-lr 1e-4 --amp bf16" bash scripts/camera_ablation.sh
#
# Settings (environment variables):
#     DATA        LIBERO demonstration files or folder (required)
#     SUITE       LIBERO suite to evaluate on                     (default libero_10)
#     SEEDS       training seeds                                  (default "0 1 2")
#     TRIALS      evaluation episodes per task                    (default 50)
#     EVAL_SEED   evaluation seed, the same for every run         (default 0)
#     RUNS        training output root                            (default runs/camera_ablation)
#     RESULTS     evaluation output root                          (default eval_results/camera_ablation)
#     TRAIN_ARGS  extra train.py arguments, identical for both settings
#     EVAL_ARGS   extra eval.py arguments, identical for both settings
#     BENCH_DEVICE  device for bench_latency.py (default: cuda if available, else mps on macOS, else cpu)
#     SKIP_BENCH  set to 1 to skip the latency benchmark
#     PYTHON      Python interpreter                              (default python)

set -euo pipefail

: "${DATA:?set DATA to the LIBERO demonstration files or folder, e.g. DATA=~/LIBERO/datasets/libero_10}"
SUITE="${SUITE:-libero_10}"
SEEDS="${SEEDS:-0 1 2}"
TRIALS="${TRIALS:-50}"
EVAL_SEED="${EVAL_SEED:-0}"
RUNS="${RUNS:-runs/camera_ablation}"
RESULTS="${RESULTS:-eval_results/camera_ablation}"
TRAIN_ARGS="${TRAIN_ARGS:-}"
EVAL_ARGS="${EVAL_ARGS:-}"
SKIP_BENCH="${SKIP_BENCH:-0}"
PY="${PYTHON:-python}"

REPO="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$RUNS" "$RESULTS"

cameras_for() {
  case "$1" in
    1cam) echo "agentview_rgb" ;;
    2cam) echo "agentview_rgb eye_in_hand_rgb" ;;
    *) echo "unknown setting $1" >&2; return 1 ;;
  esac
}

for seed in $SEEDS; do
  for setting in 1cam 2cam; do
    run="$RUNS/${SUITE}_${setting}_s${seed}"
    out="$RESULTS/${SUITE}_${setting}_s${seed}"
    # shellcheck disable=SC2046,SC2086  # word splitting of the camera list and the extra arguments is intended
    if [[ -f "$run/policy_final.pt" ]]; then
      echo "[skip] $run is trained"
    else
      echo "[train] $setting seed $seed -> $run"
      "$PY" "$REPO/train.py" --data "$DATA" --out "$run" --seed "$seed" --camera-keys $(cameras_for "$setting") $TRAIN_ARGS
    fi
    if grep -q '"status": "complete"' "$out/eval_metrics.json" 2>/dev/null; then
      echo "[skip] $out is evaluated"
    else
      echo "[eval] $setting seed $seed -> $out"
      # shellcheck disable=SC2086
      "$PY" "$REPO/eval.py" --checkpoint "$run/policy_final.pt" --suite-name "$SUITE" \
        --num-trials-per-task "$TRIALS" --seed "$EVAL_SEED" --out "$out" $EVAL_ARGS
    fi
  done
done

if [[ "$SKIP_BENCH" != "1" ]]; then
  if [[ -z "${BENCH_DEVICE:-}" ]]; then
    if command -v nvidia-smi >/dev/null 2>&1; then BENCH_DEVICE=cuda
    elif [[ "$(uname -s)" == "Darwin" ]]; then BENCH_DEVICE=mps
    else BENCH_DEVICE=cpu; fi
  fi
  first_seed="${SEEDS%% *}"
  for setting in 1cam 2cam; do
    config="$RUNS/${SUITE}_${setting}_s${first_seed}/config.json"
    echo "[bench] $setting on $BENCH_DEVICE (model sizes from $config)"
    "$PY" "$REPO/bench_latency.py" --device "$BENCH_DEVICE" --config-json "$config" > "$RESULTS/bench_${setting}.txt"
  done
fi

"$PY" "$REPO/scripts/summarize_ablation.py" "$RESULTS" --json "$RESULTS/summary.json" | tee "$RESULTS/summary.md"
