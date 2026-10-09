# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
"""
export_checkpoint.py -- a slim, shareable copy of a trained stage-2 checkpoint (e.g. for Hugging Face).

A training checkpoint (``policy_final.pt``) also stores the optimiser, the learning-rate schedule,
random-number states and the training machine's file paths, which are only needed to resume
training. This script writes, into ``--out``:

    <name>.pt            same format as a training checkpoint, minus the resume-only parts;
                         loads with ``train.load_pipeline`` and runs with ``eval.py --checkpoint``
                         exactly like the original (the script checks the weights are identical)
    <name>.safetensors   the model weights (pipeline state dict) in the safetensors format
    <name>.json          VPAConfig and the data / text-encoder record of the checkpoint

File paths in the record are reduced to file names (``eval.py`` finds LIBERO demonstration files
by name; pass ``--data-root`` there). Weights of a pre-trained CLIP text encoder are never stored:
they are downloaded from Hugging Face when the checkpoint is loaded.

Usage (from the repository root):
    python scripts/export_checkpoint.py runs/camera_ablation/libero_10_2cam_s0/policy_final.pt \
        --name vpa_libero10_2cam --out exports/vpa-libero10
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from typing import Any, Dict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

RESUME_ONLY = {"trainer": None, "momentum_encoder": None, "ema_momentum": None, "rng": {}}


def _names(value: Any) -> Any:
    if isinstance(value, str):
        return os.path.basename(value.rstrip("/"))
    if isinstance(value, list):
        return [_names(v) for v in value]
    return value


def slim_checkpoint(ckpt: Dict[str, Any]) -> Dict[str, Any]:
    """Copy of a training checkpoint without optimiser / RNG state and with file names instead of paths."""
    out = {key: ckpt[key] for key in ("format", "stage", "step", "config", "model")}
    extra = copy.deepcopy(ckpt["extra"])
    data = extra.get("data", {})
    if "files" in data:
        data["files"] = _names(data["files"])
    args = extra.get("args")
    if isinstance(args, dict):
        for key in ("data", "out", "resume", "init_from", "config_json", "init_encoder"):
            if args.get(key):
                args[key] = _names(args[key])
    out["extra"] = extra
    out.update(copy.deepcopy(RESUME_ONLY))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint", help="a stage-2 checkpoint written by train.py (e.g. policy_final.pt)")
    parser.add_argument("--name", required=True, help="base name of the exported files")
    parser.add_argument("--out", required=True, help="output folder")
    parser.add_argument("--no-verify", action="store_true", help="skip reloading both files and comparing weights")
    args = parser.parse_args()

    import torch

    from train import load_checkpoint, load_pipeline

    ckpt = load_checkpoint(args.checkpoint)
    if ckpt["stage"] != "policy":
        raise SystemExit(f"{args.checkpoint} is a stage-{ckpt['stage']!r} checkpoint; export the stage-2 one")
    os.makedirs(args.out, exist_ok=True)
    slim = slim_checkpoint(ckpt)
    pt_path = os.path.join(args.out, f"{args.name}.pt")
    torch.save(slim, pt_path)

    tensors = {k: v.detach().cpu().contiguous().clone() for k, v in slim["model"].items()}
    st_path = os.path.join(args.out, f"{args.name}.safetensors")
    from safetensors.torch import save_file

    save_file(tensors, st_path, metadata={"format": str(slim["format"]), "stage": str(slim["stage"]),
                                          "step": str(slim["step"]), "config": json.dumps(slim["config"])})
    json_path = os.path.join(args.out, f"{args.name}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"format": slim["format"], "stage": slim["stage"], "step": slim["step"],
                   "config": slim["config"], "extra": slim["extra"]}, f, indent=2, default=str)

    if not args.no_verify:
        original = load_pipeline(args.checkpoint).state_dict()
        exported = load_pipeline(pt_path).state_dict()
        if original.keys() != exported.keys() or not all(torch.equal(original[k], exported[k]) for k in original):
            raise SystemExit("verification FAILED: the exported checkpoint loads to different weights")
        from safetensors.torch import load_file

        loaded = load_file(st_path)
        if not all(torch.equal(loaded[k], tensors[k]) for k in tensors):
            raise SystemExit("verification FAILED: safetensors weights differ")
        print("verified: the exported checkpoint loads to identical weights")
    for path in (pt_path, st_path, json_path):
        print(f"{path}  {os.path.getsize(path) / 1e6:.1f} MB")
    print(f"original: {args.checkpoint}  {os.path.getsize(args.checkpoint) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
