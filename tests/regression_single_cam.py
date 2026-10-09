# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
"""
regression_single_cam.py -- the single-camera setting must be bit-identical to the baseline tag.

Multi-camera support (``--camera-keys``, ``VPAConfig.num_views``) must not change anything when one
camera is used. This script checks that end to end:

    1. export the baseline (default: the git tag ``single-cam-baseline``) into a temporary folder;
    2. run ``tests/dump_reference_outputs.py`` against the baseline and against this working tree,
       each in its own process (CPU, one thread, fixed seeds);
    3. require every dumped tensor to be equal (``torch.equal``) and every other value to be equal:
       encoder, pipeline step / act, dataset samples, a short two-stage training run (weights and
       every logged loss), and fake-environment evaluation episodes;
    4. load the baseline's trained checkpoint with this tree's code and require the same weights and
       the same evaluation episodes (older checkpoints keep working).

If something differs, the baseline is dumped a second time: if the baseline does not even
reproduce itself, the difference comes from the environment (threads, library versions), not from
the code, and the script says so.

Requirements: git, torch, h5py, numpy (as for the self-tests). No network.

Usage (from the repository root):
    git tag single-cam-baseline 5729a11            # once, if the tag does not exist yet
    python tests/regression_single_cam.py          # exit code 0 = identical
    python tests/regression_single_cam.py --baseline <commit or tag> --keep
"""

from __future__ import annotations

import argparse
import io
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
from typing import Any, Dict, List, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
DUMP = os.path.join(HERE, "dump_reference_outputs.py")


def export_tree(ref: str, dst: str) -> None:
    """``git archive <ref>`` extracted into ``dst``."""
    proc = subprocess.run(["git", "-C", REPO, "archive", "--format=tar", ref], capture_output=True, check=False)
    if proc.returncode != 0:
        msg = proc.stderr.decode(errors="replace").strip()
        raise SystemExit(f"git archive {ref} failed: {msg}\n"
                         f"Create the baseline tag first: git tag single-cam-baseline 5729a11")
    os.makedirs(dst, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(proc.stdout)) as tar:
        try:
            tar.extractall(dst, filter="data")
        except TypeError:  # Python without extraction filters
            tar.extractall(dst)  # noqa: S202 -- our own repository's archive


def dump(tree: str, work: str, out: str, extra: List[str]) -> None:
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONHASHSEED="0", CUDA_VISIBLE_DEVICES="")
    cmd = [sys.executable, DUMP, "--tree", tree, "--work", work, "--out", out, *extra]
    print("$", " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, cwd=tree, env=env, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        sys.stdout.write(proc.stdout[-4000:])
        sys.stderr.write(proc.stderr[-8000:])
        raise SystemExit(f"dumping {tree} failed (exit code {proc.returncode})")
    print(proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else "(no output)", flush=True)


def load(path: str) -> Dict[str, Any]:
    import torch

    return torch.load(path, map_location="cpu", weights_only=True)


def same_bits(x: Any, y: Any) -> bool:
    """Bit-for-bit equality: same shape, dtype and bytes (NaN == NaN; -0.0 != 0.0).

    ``torch.equal`` is not enough: it treats NaN as unequal to itself, and the stage-1 checkpoint holds
    NaN placeholders (gamma*, mu_Z, mu_S, sigma_S before calibration).
    """
    import torch

    if x.shape != y.shape or x.dtype != y.dtype:
        return False
    xb = x.detach().contiguous().reshape(-1).view(torch.uint8)
    yb = y.detach().contiguous().reshape(-1).view(torch.uint8)
    return torch.equal(xb, yb)


def compare_tensors(a: Dict[str, Any], b: Dict[str, Any], keys: List[Tuple[str, str]]) -> List[str]:
    """Pairs (key in a, key in b) that are not bit-identical, with the size of the difference."""
    import torch

    problems = []
    for ka, kb in keys:
        if kb not in b:
            problems.append(f"missing: {kb}")
            continue
        x, y = a[ka], b[kb]
        if x.shape != y.shape or x.dtype != y.dtype:
            problems.append(f"{kb}: {tuple(x.shape)} {x.dtype} vs {tuple(y.shape)} {y.dtype}")
        elif not same_bits(x, y):
            if x.is_floating_point():
                diff = (x.double() - y.double()).abs()
                nan_mismatch = int((torch.isnan(x) != torch.isnan(y)).sum())
                worst = float(torch.nan_to_num(diff, nan=0.0).max()) if diff.numel() else 0.0
                problems.append(f"{kb}: values differ (max |diff| = {worst:.3g}, NaN mismatches = {nan_mismatch})")
            else:
                problems.append(f"{kb}: values differ ({int((x != y).sum())} elements)")
    return problems


def compare(base: Dict[str, Any], new: Dict[str, Any]) -> Tuple[List[str], int]:
    bt, nt = base["tensors"], new["tensors"]
    bo, no = base["objects"], new["objects"]
    problems = compare_tensors(bt, nt, [(k, k) for k in bt])
    for key, value in bo.items():
        if key not in no:
            problems.append(f"missing: {key}")
        elif no[key] != value:
            problems.append(f"{key}: values differ")
    # the baseline's checkpoint, loaded and evaluated by this tree
    state = [(k, k.replace("train/state/", "baseline_ckpt/state/", 1)) for k in bt if k.startswith("train/state/")]
    problems += [f"[baseline checkpoint] {p}" for p in compare_tensors(bt, nt, state)]
    extra_state = sorted(set(k for k in nt if k.startswith("baseline_ckpt/state/"))
                         - {kb for _, kb in state})
    problems += [f"[baseline checkpoint] unexpected tensor {k}" for k in extra_state]
    if no.get("baseline_ckpt/eval/episodes") != bo.get("eval/episodes"):
        problems.append("[baseline checkpoint] evaluation episodes differ")
    return problems, len(bt) + len(state) + len(bo) + 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--baseline", default="single-cam-baseline", help="git tag or commit of the baseline")
    parser.add_argument("--keep", action="store_true", help="keep the temporary folder (prints its path)")
    args = parser.parse_args()

    root = tempfile.mkdtemp(prefix="vpa_regression_")
    try:
        base_tree = os.path.join(root, "baseline_tree")
        export_tree(args.baseline, base_tree)
        base_work, new_work = os.path.join(root, "baseline_work"), os.path.join(root, "new_work")
        base_out, new_out = os.path.join(root, "baseline.pt"), os.path.join(root, "new.pt")
        dump(base_tree, base_work, base_out, [])
        dump(REPO, new_work, new_out,
             ["--baseline-checkpoint", os.path.join(base_work, "run", "policy_final.pt"),
              "--baseline-data", os.path.join(base_work, "data")])
        problems, checked = compare(load(base_out), load(new_out))
        if problems:
            again_work, again_out = os.path.join(root, "baseline_work2"), os.path.join(root, "baseline2.pt")
            dump(base_tree, again_work, again_out, [])
            first, again = load(base_out)["tensors"], load(again_out)["tensors"]
            unstable = compare_tensors(first, again, [(k, k) for k in first])
            print(f"\nFAIL: {len(problems)} of {checked} checks differ from {args.baseline}:")
            for p in problems[:40]:
                print("  -", p)
            if len(problems) > 40:
                print(f"  ... and {len(problems) - 40} more")
            if unstable:
                print(f"\nNote: the baseline does not reproduce itself here ({len(unstable)} tensors differ "
                      "between two baseline runs), so the environment is nondeterministic; fix that first.")
            sys.exit(1)
        print(f"\nPASS: single-camera outputs are bit-identical to {args.baseline} ({checked} checks), "
              "and the baseline's checkpoint loads and evaluates identically.")
    finally:
        if args.keep:
            print(f"kept {root}")
        else:
            shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    main()
