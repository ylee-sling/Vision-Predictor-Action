# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
"""
dump_reference_outputs.py -- deterministic single-camera outputs of one source tree.

Used by ``regression_single_cam.py``, which runs this script once against the baseline tree (the
``single-cam-baseline`` tag) and once against the working tree, each in its own process, and
requires the two dumps to be bit-identical.

Only the single-camera API that exists in both trees is used (no ``camera_keys``, no
``num_views``), so the script also proves that this API did not change. Everything runs on the CPU
with one thread and fixed seeds. What is dumped:

    perception  E_psi weights, forward / encode_milestones / encode_siamese, E_psi_bar after EMA updates
    pipeline    three ``step`` outputs and six ``act`` outputs of the tiny pipeline
    dataset     every sample, milestone frames and frame blocks of a synthetic LIBERO-format set
                (native size and a resized copy)
    train       a 3 + 3-step two-stage run: final weights, every logged metric, the stage-1 checkpoint
    eval        fake-environment episodes of the trained checkpoint (plain and Eq. 15 overrides)

With ``--baseline-checkpoint`` (working tree only), the baseline's trained checkpoint is also loaded
and evaluated with this tree's code, to check that older checkpoints load and behave identically.

Usage (normally called by regression_single_cam.py):
    python tests/dump_reference_outputs.py --tree <source tree> --work <empty dir> --out dump.pt
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from typing import Any, Dict, List

# Wall-clock and path fields of episode records; everything else must match exactly.
VARYING_EPISODE_KEYS = ("wall_time_s", "step_latency_ms_mean", "step_latency_ms_p95", "video")
# Throughput fields of the training metrics.
VARYING_METRIC_KEYS = ("samples_per_s",)
TASK_NAMES = ("KITCHEN_SCENE1_put_the_bowl_on_the_plate", "LIVING_ROOM_SCENE2_stack_the_blocks")


def _import_tree(tree: str) -> Dict[str, Any]:
    """Import the VPA modules of ``tree`` (and nothing from anywhere else)."""
    tree = os.path.abspath(tree)
    sys.path.insert(0, tree)
    mods = {name: importlib.import_module(name) for name in ("perception", "pipeline", "dataset", "train")}
    # the evaluation with its fake environment: eval_fixed_horizon.py in the baseline, eval.py afterwards
    ev = importlib.import_module("eval_fixed_horizon")
    if not hasattr(ev, "_FakeEnv"):
        ev = importlib.import_module("eval")
    mods["eval"] = ev
    for name, mod in mods.items():
        if os.path.dirname(os.path.abspath(mod.__file__)) != tree:
            raise RuntimeError(f"{name} was imported from {mod.__file__}, not from {tree}")
    return mods


def _tensors(prefix: str, mapping: Dict[str, Any], out: Dict[str, Any]) -> None:
    import torch

    for key, value in mapping.items():
        if isinstance(value, torch.Tensor):
            out[f"{prefix}/{key}"] = value.detach().cpu().clone()


def dump_perception(m: Dict[str, Any], out: Dict[str, Any]) -> None:
    import torch

    perception = m["perception"]
    torch.manual_seed(0)
    enc = perception.VisionEncoder(image_size=(32, 32), patch_size=8, in_channels=3, width=32, depth=2,
                                   num_heads=4, latent_dim=16).eval()
    _tensors("perception/init", enc.state_dict(), out)
    g = torch.Generator().manual_seed(1)
    obs = torch.randn(2, 3, 32, 32, generator=g)
    goals = torch.randn(2, 3, 3, 32, 32, generator=g)
    with torch.no_grad():
        out["perception/z_t"] = enc(obs)
        out["perception/z_g"] = enc.encode_milestones(goals)
        z_t, z_g = enc.encode_siamese(obs, goals)
        out["perception/siamese_z_t"], out["perception/siamese_z_g"] = z_t, z_g
    target = perception.MomentumEncoder(enc, momentum=0.9)
    for _ in range(3):
        with torch.no_grad():
            for p in enc.parameters():
                p.add_(0.1 * torch.randn(p.shape, generator=g))
        target.update(enc)
    out["perception/target_z"] = target(obs)
    enc.train()
    obs_g = obs.clone().requires_grad_(True)
    ((enc(obs_g) - out["perception/target_z"]) ** 2).sum().backward()
    out["perception/grad_input"] = obs_g.grad.detach().clone()
    _tensors("perception/grad", {k: p.grad for k, p in enc.named_parameters()}, out)


def dump_pipeline(m: Dict[str, Any], out: Dict[str, Any], objects: Dict[str, Any]) -> None:
    import torch

    pipeline = m["pipeline"]
    torch.manual_seed(0)
    pipe = pipeline._tiny_pipeline(16, 2)
    _tensors("pipeline/init", pipe.state_dict(), out)
    g = torch.Generator().manual_seed(2)
    texts = ["move the cup", "throw the cup away", "push the block"]
    goals = torch.randn(3, 2, 3, 32, 32, generator=g)
    obs0 = torch.randn(3, 3, 32, 32, generator=g)
    goals[:, 0] = obs0                                   # milestone 1 is reached at the first step
    pipe.reset(texts, goals)
    evaluations: List[int] = []
    for i in range(3):
        obs = obs0 if i == 0 else torch.randn(3, 3, 32, 32, generator=g)
        state = torch.randn(3, 6, generator=g)
        noise = torch.randn(3, 16, 3, generator=g)
        step = pipe.step(obs, state, noise=noise)._asdict()
        evaluations.append(int(step.pop("num_sequential_evaluations")))
        _tensors(f"pipeline/step{i}", step, out)
    objects["pipeline/evaluations"] = evaluations
    pipe.reset(texts, goals)
    torch.manual_seed(3)                                 # act() draws its flow noise from the global RNG
    for i in range(6):
        res = pipe.act(torch.randn(3, 3, 32, 32, generator=g), torch.randn(3, 6, generator=g))
        out[f"pipeline/act{i}/action"] = res.action
        out[f"pipeline/act{i}/replanned"] = res.replanned


def _write_data(m: Dict[str, Any], data_dir: str) -> None:
    write = m["dataset"]._write_synthetic_libero
    os.makedirs(data_dir, exist_ok=True)
    write(os.path.join(data_dir, f"{TASK_NAMES[0]}_demo.hdf5"), (14, 11, 16, 12), seed=5)
    write(os.path.join(data_dir, f"{TASK_NAMES[1]}_demo.hdf5"), (13, 10, 15), seed=6, instruction=None)


def dump_dataset(m: Dict[str, Any], data_dir: str, out: Dict[str, Any], objects: Dict[str, Any]) -> None:
    dataset = m["dataset"]
    kwargs = dict(primitive_source="gripper", milestone_source="gripper", gripper_window=2)
    ds = dataset.LiberoHDF5Dataset(data_dir, 4, predictor_stride=2, **kwargs)
    objects["dataset/len"] = len(ds)
    objects["dataset/episodes"] = [[e.demo, e.length, e.task, list(e.milestones)] for e in ds.episodes]
    objects["dataset/image_shape"] = list(ds.image_shape)
    for i in range(len(ds)):
        _tensors(f"dataset/item{i}", ds[i], out)
    for e in range(len(ds.episodes)):
        out[f"dataset/milestones{e}"] = ds.milestone_frames(e)
        out[f"dataset/episode{e}"] = ds.episode_frames(e)
    blocks = dataset.LiberoFrameDataset(ds, block_size=5)
    for i in range(len(blocks)):
        _tensors(f"dataset/block{i}", blocks[i], out)
    small = dataset.LiberoHDF5Dataset(data_dir, 4, image_size=(8, 8), **kwargs)
    for i in (0, 5, len(small) - 1):
        _tensors(f"dataset/small{i}", small[i], out)
    out["dataset/small_milestones0"] = small.milestone_frames(0)
    for d in (ds, small, blocks.base):
        d.close()


def dump_train(m: Dict[str, Any], data_dir: str, work: str, out: Dict[str, Any], objects: Dict[str, Any]) -> str:
    import torch

    train = m["train"]
    cfg_path = os.path.join(work, "tiny.json")
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(train._TINY_CONFIG, f)
    run_dir = os.path.join(work, "run")
    args = train._test_args(data_dir, run_dir, cfg_path, ["--jepa-steps", "3", "--policy-steps", "3",
                                                          "--ckpt-every", "2"])
    pipe = train.run(args)
    train._close_logging()
    _tensors("train/state", pipe.state_dict(), out)
    with open(os.path.join(run_dir, "metrics.jsonl"), encoding="utf-8") as f:
        objects["train/metrics"] = [{k: v for k, v in json.loads(line).items() if k not in VARYING_METRIC_KEYS}
                                    for line in f]
    jepa = torch.load(os.path.join(run_dir, "jepa_final.pt"), map_location="cpu", weights_only=True)
    _tensors("train/jepa_final", jepa["model"], out)          # includes NaN placeholders (not yet calibrated)
    _tensors("train/jepa_final_momentum", jepa["momentum_encoder"], out)
    objects["train/jepa_final_step"] = int(jepa["step"])
    return os.path.join(run_dir, "policy_final.pt")


def _episodes(results: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [{k: v for k, v in ep.items() if k not in VARYING_EPISODE_KEYS}
            for task in results["tasks"] for ep in task["episodes"]]


def evaluate_checkpoint(m: Dict[str, Any], ckpt: str, data_dir: str, out_dir: str,
                        extra: List[str]) -> List[Dict[str, Any]]:
    """Fake-environment evaluation (the recipe of eval.py's self-test) of one checkpoint."""
    ev, train = m["eval"], m["train"]
    tasks = [ev._FakeTask(name=TASK_NAMES[0], language="pick up the red cup and place it on the plate",
                          problem_folder="fake", bddl_file="x.bddl", init_states_file="x.pruned_init"),
             ev._FakeTask(name=TASK_NAMES[1], language="stack the blocks", problem_folder="fake",
                          bddl_file="y.bddl", init_states_file="y.pruned_init")]
    suite = ev._FakeSuite(tasks, num_init_states=3)
    wait, max_steps = 2, 9

    def factory(task: Any) -> Any:
        i = TASK_NAMES.index(task.name)
        return ev._FakeEnv(os.path.join(data_dir, f"{task.name}_demo.hdf5"), "demo_0", i,
                           success_after=wait + 5 if i == 0 else None)

    args = ev.build_arg_parser().parse_args(
        ["--checkpoint", ckpt, "--device", "cpu", "--num-trials-per-task", "2", "--max-steps", str(max_steps),
         "--num-steps-wait", str(wait), "--out", out_dir, *extra])
    results = ev.evaluate(args, suite=suite, env_factory=factory, pipe=train.load_pipeline(ckpt))
    ev._close_logging()
    return _episodes(results)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tree", required=True, help="source tree whose modules are dumped")
    parser.add_argument("--work", required=True, help="empty working directory (data, runs, eval output)")
    parser.add_argument("--out", required=True, help="output file (torch.save of tensors and JSON-like objects)")
    parser.add_argument("--baseline-checkpoint", help="also load and evaluate this (baseline) checkpoint")
    parser.add_argument("--baseline-data", help="data folder the baseline checkpoint was trained on")
    args = parser.parse_args()

    m = _import_tree(args.tree)
    import torch

    torch.set_num_threads(1)                             # same reduction order in both processes
    os.makedirs(args.work, exist_ok=True)
    tensors: Dict[str, Any] = {}
    objects: Dict[str, Any] = {}
    data_dir = os.path.join(args.work, "data")
    _write_data(m, data_dir)

    dump_perception(m, tensors)
    dump_pipeline(m, tensors, objects)
    dump_dataset(m, data_dir, tensors, objects)
    ckpt = dump_train(m, data_dir, args.work, tensors, objects)
    objects["eval/episodes"] = evaluate_checkpoint(m, ckpt, data_dir, os.path.join(args.work, "eval"), [])
    objects["eval/episodes_eq15"] = evaluate_checkpoint(
        m, ckpt, data_dir, os.path.join(args.work, "eval_eq15"),
        ["--fixed-horizon", "3", "--beta", "40", "--min-horizon", "2"])
    if args.baseline_checkpoint:
        base_data = args.baseline_data or data_dir
        _tensors("baseline_ckpt/state", m["train"].load_pipeline(args.baseline_checkpoint).state_dict(), tensors)
        objects["baseline_ckpt/eval/episodes"] = evaluate_checkpoint(
            m, args.baseline_checkpoint, base_data, os.path.join(args.work, "eval_baseline_ckpt"), [])
    torch.save({"tensors": tensors, "objects": objects, "tree": os.path.abspath(args.tree)}, args.out)
    print(f"dumped {len(tensors)} tensors and {len(objects)} objects from {os.path.abspath(args.tree)}")


if __name__ == "__main__":
    main()
