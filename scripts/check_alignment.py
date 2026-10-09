# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
"""
check_alignment.py -- is frame k of a LIBERO demonstration recorded before or after ``actions[k]``?

``dataset.py`` pairs frame t (image I_t, proprioception S_t) with the action chunk
A_t = (actions[t+1], ..., actions[t+H]) (``ACTION_OFFSET = 1``), because LIBERO's
``create_dataset.py`` calls ``env.step(actions[j])`` and only then records the observation as frame j.
This script checks that convention on your own files, without trusting the recorder's code.

How: every demonstration also stores ``states``, the flattened simulator state recorded *before*
``actions[j]`` (filtered with the same indices as the frames). The robot's 7 joint angles appear in
that state vector. If frame k was recorded after ``actions[k]``, its ``joint_states`` equal the joint
angles of ``states[k + 1]`` (lag 1); if before, those of ``states[k]`` (lag 0). The script finds the
7 entries of the state vector that hold the joints (searching every offset) and reports, for each
lag, the mean absolute difference.

Verdict:
    lag 1 ~ 0, lag 0 clearly larger  ->  frame k is AFTER actions[k]; ACTION_OFFSET = 1 is correct.
    lag 0 ~ 0, lag 1 clearly larger  ->  frame k is BEFORE actions[k]; ACTION_OFFSET should be 0.

Usage (needs h5py and numpy only):
    python scripts/check_alignment.py data/libero_10                 # every file, first 5 demos each
    python scripts/check_alignment.py data/libero_10/X_demo.hdf5 --demos 50
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from typing import Dict, List, Tuple

import numpy as np


def lag_errors(joints: np.ndarray, states: np.ndarray, max_lag: int = 2) -> Tuple[int, Dict[int, float]]:
    """Locate the joint angles inside the state vectors and measure each lag.

    Args:
        joints: ``[T, 7]`` recorded ``joint_states``.
        states: ``[T, S]`` recorded flattened simulator states.
        max_lag: lags 0..max_lag are compared (``joints[k]`` vs ``states[k + lag]``).

    Returns:
        (offset of the 7 joint entries in the state vector, {lag: mean |joints[k] - states[k + lag, offset:offset+7]|}).
        The offset is the one with the smallest error at any lag.
    """
    t, n = joints.shape
    if states.shape[0] != t or states.shape[1] < n:
        raise ValueError(f"shapes do not fit: joints {joints.shape}, states {states.shape}")
    usable = t - max_lag
    if usable < 2:
        raise ValueError("demonstration too short")
    best_offset, best_err = 0, np.inf
    for offset in range(states.shape[1] - n + 1):
        for lag in range(max_lag + 1):
            err = float(np.abs(joints[:usable] - states[lag:lag + usable, offset:offset + n]).mean())
            if err < best_err:
                best_offset, best_err = offset, err
    errors = {lag: float(np.abs(joints[:usable] - states[lag:lag + usable, best_offset:best_offset + n]).mean())
              for lag in range(max_lag + 1)}
    return best_offset, errors


def _self_test() -> None:
    """Synthetic check: states built so that frame k equals state k + 1 (or k)."""
    rng = np.random.default_rng(0)
    q = np.cumsum(rng.normal(scale=0.05, size=(40, 7)), axis=0)        # a smooth joint trajectory
    states = np.concatenate([rng.normal(size=(40, 1)), q, rng.normal(size=(40, 9))], axis=1)  # time, qpos, rest
    offset, err = lag_errors(q[1:-1], states[:-2])                      # joints[k] = states[k + 1]
    assert offset == 1 and err[1] < 1e-12 < err[0], (offset, err)
    offset, err = lag_errors(q[:-1], states[:-1])                       # joints[k] = states[k]
    assert offset == 1 and err[0] < 1e-12 < err[1], (offset, err)
    print("check_alignment.py self-test passed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", nargs="*", help="LIBERO .hdf5 files or folders")
    parser.add_argument("--demos", type=int, default=5, help="demonstrations checked per file")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        _self_test()
        return
    if not args.paths:
        parser.error("give LIBERO .hdf5 files or folders")
    import h5py

    files: List[str] = []
    for p in args.paths:
        files += sorted(glob.glob(os.path.join(p, "*.hdf5"))) if os.path.isdir(p) else [p]
    votes = {0: 0, 1: 0}
    rows = []
    for path in files:
        with h5py.File(path, "r") as f:
            demos = sorted((k for k in f["data"] if k.startswith("demo")), key=lambda k: int(k.split("_")[-1]))
            for demo in demos[:args.demos]:
                g = f["data"][demo]
                if "states" not in g or "joint_states" not in g["obs"]:
                    print(f"{os.path.basename(path)}/{demo}: no states or joint_states, skipped")
                    continue
                offset, err = lag_errors(np.asarray(g["obs"]["joint_states"][()], dtype=np.float64),
                                         np.asarray(g["states"][()], dtype=np.float64))
                lag = min(err, key=err.get)
                votes[lag] = votes.get(lag, 0) + 1
                rows.append((os.path.basename(path)[:60], demo, offset, err))
    print(f"{'file':60s} {'demo':8s} {'off':>3s}  {'lag 0':>9s} {'lag 1':>9s} {'lag 2':>9s}")
    for name, demo, offset, err in rows:
        print(f"{name:60s} {demo:8s} {offset:3d}  {err[0]:9.2e} {err[1]:9.2e} {err[2]:9.2e}")
    total = sum(votes.values())
    if total == 0:
        sys.exit("nothing checked")
    print(f"\nbest lag over {total} demonstrations: " + ", ".join(f"lag {k}: {v}" for k, v in sorted(votes.items())))
    if votes.get(1, 0) == total:
        print("VERDICT: frame k is recorded AFTER actions[k] -> A_t = actions[t+1 ...] (ACTION_OFFSET = 1) is correct.")
    elif votes.get(0, 0) == total:
        print("VERDICT: frame k is recorded BEFORE actions[k] -> ACTION_OFFSET should be 0. Please report this.")
    else:
        print("VERDICT: mixed or unclear -- inspect the table (lag 2 best would mean an extra delay).")


if __name__ == "__main__":
    main()
