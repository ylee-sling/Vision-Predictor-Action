# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
"""
bench_latency.py -- per-stage latency of one VPA decision step (preprint_261008.pdf, Eq. 17).

    T_VPA = t_E + t_h + t_P + K * t_l(H)                                              (Eq. 17)

t_E, t_h, t_P and t_l(H) are the wall-clock times of one evaluation of E_psi, pi_phi^h, P_omega and
v_theta. They are measured inside ``VPAInferencePipeline.step`` with forward pre/post hooks that
synchronise the device before each clock read, so every stage is timed on its own. t_l(H) is the
mean of the K v_theta evaluations of a run. Each (H, K) configuration gets warm-up runs and then
the median over the timed runs is reported. Every run asserts num_sequential_evaluations == K + 3
(Prop. 5.1).

The weights are random (VPAConfig placeholder sizes, dummy calibration statistics), so these are
random-weight timings of the architecture on this machine, not results.

The text encoder is not on the per-step chain (c_text is computed once per episode), so by default
an offline stand-in with d_c = 512 (the CLIP ViT-B/32 projection width) is used. Pass --clip to load
the real CLIP text tower instead (downloads openai/clip-vit-base-patch32 on first use).

--num-views V times the multi-camera encoder (V camera views per observation, ``VPAConfig.num_views``):
E_psi then runs its ViT body on V frames per step (one evaluation, so the depth is still K + 3); the
other stages are unchanged. Run it with 1 and with 2 to report t_E for both camera settings.

--config-json takes the model sizes from a JSON file instead of the VPAConfig placeholders: either a
JSON object of VPAConfig fields or the ``config.json`` a training run writes (its "config" entry), so
the timings use the architecture that was actually trained (image size, patch size, widths, num_views).
H and K are still swept; --num-views, if given, overrides the file.

Run:  python bench_latency.py [--runs 100] [--warmup 10] [--device mps|cuda|cpu] [--clip] [--num-views 1|2]
                              [--config-json runs/<run>/config.json]
Not part of the self-tests.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import statistics
import time
from typing import Any, Callable, Dict, List, Optional

import torch

from perception import TextEncoderWrapper, _ToyTextModel, _toy_tokenizer
from pipeline import VPAConfig, VPAInferencePipeline

HORIZONS = (8, 16, 32, 64, 128)
INTEGRATION_STEPS = (1, 2, 3)
STAGES = ("E_psi", "pi_phi_h", "P_omega", "v_theta")


def _pick_device(requested: str) -> torch.device:
    if requested == "mps" and not torch.backends.mps.is_available():
        print("MPS is not available; falling back to CPU")
        return torch.device("cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        print("CUDA is not available; falling back to CPU")
        return torch.device("cpu")
    return torch.device(requested)


def _synchronizer(device: torch.device) -> Callable[[], None]:
    if device.type == "mps":
        return torch.mps.synchronize
    if device.type == "cuda":
        return torch.cuda.synchronize
    return lambda: None


def _build(cfg: VPAConfig, device: torch.device, use_clip: bool) -> VPAInferencePipeline:
    if use_clip:
        text = TextEncoderWrapper(cfg.clip_model_name)
    else:
        text = TextEncoderWrapper(text_model=_ToyTextModel(projection_dim=512), tokenizer=_toy_tokenizer)
    pipe = VPAInferencePipeline.from_config(cfg, text_encoder=text).to(device)
    # Dummy deployment statistics: gamma* = 1, mu_Z = 0, mu_S = 0, sigma_S = 1.
    pipe.calibrate(1.0, torch.zeros(cfg.latent_dim), torch.zeros(cfg.proprio_dim), torch.ones(cfg.proprio_dim))
    return pipe


def _attach_timers(pipe: VPAInferencePipeline, sync: Callable[[], None]) -> Dict[str, List[float]]:
    """Forward pre/post hooks that sync the device and record each evaluation's wall time (s)."""
    times: Dict[str, List[float]] = {name: [] for name in STAGES}
    modules = (pipe.vision_encoder, pipe.selector, pipe.predictor, pipe.solver.vector_field)
    for name, module in zip(STAGES, modules, strict=True):
        start: List[float] = [0.0]

        def pre(_module, _inputs, start=start) -> None:
            sync()
            start[0] = time.perf_counter()

        def post(_module, _inputs, _output, name=name, start=start) -> None:
            sync()
            times[name].append(time.perf_counter() - start[0])

        module.register_forward_pre_hook(pre)
        module.register_forward_hook(post)
    return times


def load_config_values(path: str) -> Dict[str, Any]:
    """VPAConfig fields from a JSON object of fields or from a training run's ``config.json``."""
    with open(path, encoding="utf-8") as f:
        values = json.load(f)
    if "config" in values and isinstance(values["config"], dict):
        values = values["config"]
    known = {f.name for f in dataclasses.fields(VPAConfig)}
    unknown = sorted(set(values) - known)
    if unknown:
        raise ValueError(f"unknown VPAConfig fields in {path}: {unknown}")
    for key in ("image_size", "selector_hidden"):
        if key in values:
            values[key] = tuple(int(v) for v in values[key])
    return values


def bench(h: int, k: int, device: torch.device, runs: int, warmup: int, use_clip: bool,
          num_views: Optional[int] = None, base: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
    torch.manual_seed(0)
    values = dict(base or {})                            # default: VPAConfig placeholders
    values.update(horizon=h, integration_steps=k)
    if num_views is not None:
        values["num_views"] = num_views
    if values.get("min_horizon", VPAConfig.min_horizon) > h:   # keep H_min <= H when sweeping H
        values["min_horizon"] = h
    if values.get("predictor_stride", VPAConfig.predictor_stride) > values.get("min_horizon", VPAConfig.min_horizon):
        values["predictor_stride"] = values.get("min_horizon", VPAConfig.min_horizon)
    cfg = VPAConfig(**values)
    pipe = _build(cfg, device, use_clip)
    sync = _synchronizer(device)
    times = _attach_timers(pipe, sync)
    frame_shape = pipe.vision_encoder.frame_shape        # [C, H, W], or [V, C, H, W] with V cameras
    gen = torch.Generator().manual_seed(1)
    milestones = torch.randn(1, 2, *frame_shape, generator=gen).to(device)
    obs = torch.randn(1, *frame_shape, generator=gen).to(device)
    state = torch.randn(1, cfg.proprio_dim, generator=gen).to(device)
    pipe.reset("pick up the cup", milestones)

    per_run = {name: [] for name in STAGES}
    totals = []
    for run in range(warmup + runs):
        for v in times.values():
            v.clear()
        sync()
        t0 = time.perf_counter()
        out = pipe.step(obs, state)
        sync()
        total = time.perf_counter() - t0
        assert out.num_sequential_evaluations == k + 3, out.num_sequential_evaluations
        assert [len(times[s]) for s in STAGES] == [1, 1, 1, k]
        if run >= warmup:
            totals.append(total)
            for s in STAGES:
                per_run[s].append(statistics.fmean(times[s]))  # v_theta: mean of its K evaluations
    ms = {s: 1e3 * statistics.median(per_run[s]) for s in STAGES}
    ms["step"] = 1e3 * statistics.median(totals)
    ms["eq17"] = ms["E_psi"] + ms["pi_phi_h"] + ms["P_omega"] + k * ms["v_theta"]
    return ms


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--device", default="mps", choices=("mps", "cuda", "cpu"))
    parser.add_argument("--clip", action="store_true", help="use the real CLIP text encoder (network)")
    parser.add_argument("--num-views", type=int,
                        help="camera views per observation (VPAConfig.num_views; default 1 or the --config-json value)")
    parser.add_argument("--config-json", help="VPAConfig fields, or a training run's config.json")
    args = parser.parse_args()
    if args.num_views is not None and args.num_views < 1:
        parser.error("--num-views must be >= 1")
    sizes_from = load_config_values(args.config_json) if args.config_json else {}
    views = args.num_views if args.num_views is not None else int(sizes_from.get("num_views", 1))
    device = _pick_device(args.device)

    sizes = f"model sizes from {args.config_json}" if args.config_json else "VPAConfig placeholder sizes"
    print(f"RANDOM-WEIGHT TIMINGS, NOT RESULTS: untrained weights, {sizes}, batch 1.")
    print(f"torch {torch.__version__}, device {device}, median of {args.runs} runs after {args.warmup} warm-up, "
          f"{views} camera view(s)")
    print("Each stage is timed with a device sync before every clock read (Eq. 17 components, ms).\n")
    header = f"{'K':>2} {'H':>4} | {'t_E':>7} {'t_h':>7} {'t_P':>7} {'t_l(H)':>7} | {'Eq.17 sum':>9} {'step()':>8}"
    print(header)
    print("-" * len(header))
    results: Dict[tuple, Dict[str, float]] = {}
    with torch.inference_mode():
        for k in INTEGRATION_STEPS:
            for h in HORIZONS:
                r = bench(h, k, device, args.runs, args.warmup, args.clip, views, sizes_from)
                results[(k, h)] = r
                print(f"{k:>2} {h:>4} | {r['E_psi']:7.3f} {r['pi_phi_h']:7.3f} {r['P_omega']:7.3f} "
                      f"{r['v_theta']:7.3f} | {r['eq17']:9.3f} {r['step']:8.3f}")
            print()

    print("t_l(H) relative to H = 8 (same K):")
    for k in INTEGRATION_STEPS:
        base = results[(k, HORIZONS[0])]["v_theta"]
        ratios = "  ".join(f"H={h}: {results[(k, h)]['v_theta'] / base:4.2f}x" for h in HORIZONS)
        print(f"  K={k}: {ratios}")
    print("\nstep() includes non-network work (gather, conditioning, filter, pointer) and the sync overhead "
          "of the timing hooks, so it exceeds the Eq. 17 sum.")


if __name__ == "__main__":
    main()
