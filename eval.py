# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
"""
eval.py -- Closed-loop evaluation of a trained VPA policy on the LIBERO benchmark suites
(preprint_261008.pdf, Fig. 2, Secs. 3.2, 4.2, 4.4, 6).

One episode = one LIBERO task started from one of its fixed initial states:

    env.reset(); obs = env.set_init_state(s_0)         LIBERO's fixed initial states (trial % 50)
    ``--num-steps-wait`` zero actions                  physics settling, as in LIBERO's own evaluation
    pipe.reset(x, I_g^(1..M))                          c_text and z_g^(1..M) once per episode
    repeat until success or ``--max-steps`` environment steps:
        out = pipe.step(I_t, S_t)                      one decision step, K + 3 network evaluations
        execute the first H_t actions of A_hat_t       (Eq. 15; stop early on success / max steps)
    success = LIBERO's ``done`` flag, i.e. ``env._check_success()``

Observation mapping. The training HDF5 files were written by LIBERO's ``create_dataset.py`` from the
same simulator, so each training key is rebuilt from the raw environment observation:

    agentview_rgb   <- obs["agentview_image"]            eye_in_hand_rgb <- obs["robot0_eye_in_hand_image"]
    ee_pos          <- obs["robot0_eef_pos"]             ee_ori  <- quat2axisangle(obs["robot0_eef_quat"])
    ee_states       <- Concat(ee_pos, ee_ori)            gripper_states <- obs["robot0_gripper_qpos"]
    joint_states    <- obs["robot0_joint_pos"]

Frames go through exactly the transform of ``LiberoHDF5Dataset`` (180-degree rotation if the
checkpoint was trained with it, scaling to [0, 1], resize to the model's input size). The camera
renders at ``--camera-size`` (128, the resolution of the LIBERO datasets) before that resize.

Milestone frames. LIBERO provides no goal images, but the paper's goal g = (x, I_g^(1..M)) needs
them (Sec. 3.1). ``--milestone-source demo`` (default) takes them from a demonstration of the same
task with the rule the checkpoint was trained with (final frame, gripper releases or annotated
indices); trial k uses demonstration k mod N unless ``--milestone-demo`` fixes one.
``--milestone-source initial`` uses the first observation as the only milestone (M = 1); the
Eq. 9a test is then satisfied at the first decision step, so it is a fallback for missing data only.

Decisions where the paper is silent (do not change silently):
    * Success is LIBERO's success predicate. The Eq. 9a completion flag (task complete at m_t = M)
      is recorded but ends the episode only with ``--stop-on-task-complete`` (the paper's rule);
      a completion before LIBERO reports success is counted as premature (Sec. 6, item 5).
    * Actions are clipped to [-1, 1], the bounds of LIBERO's OSC_POSE controller.
    * The flow noise xi of every decision step comes from a CPU generator seeded per episode
      (seed + 1000 * task_id + trial), so results do not depend on the device or task subset.

Requirements: LIBERO (and its robosuite / MuJoCo stack). Headless Linux usually needs
``MUJOCO_GL=egl``. LIBERO asks for its dataset folder on the first import; run
``python -c "import libero.libero"`` once interactively to create ``~/.libero/config.yaml``.
``--save-video`` needs ``imageio`` with ``imageio-ffmpeg``.

Usage:
    python eval.py --checkpoint runs/libero10/policy_final.pt --suite-name libero_10
    python eval.py --checkpoint ... --suite-name libero_90 --task-ids 0 1 2 --out eval_results/shard0
    python eval.py --self-test        # offline, CPU: fake suite and environment, tiny trained model
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import json
import logging
import math
import os
import pickle
import sys
import tempfile
import time
from collections import Counter
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor

try:  # package import (e.g. `from vpa.eval import ...`)
    from .dataset import LiberoHDF5Dataset, _write_synthetic_libero
    from .pipeline import VPAConfig, VPAInferencePipeline
    from .train import _json_safe, load_checkpoint, load_pipeline, resolve_device, seed_everything
except ImportError:  # flat import (files side by side, `python eval.py`)
    from dataset import LiberoHDF5Dataset, _write_synthetic_libero
    from pipeline import VPAConfig, VPAInferencePipeline
    from train import _json_safe, load_checkpoint, load_pipeline, resolve_device, seed_everything

__all__ = [
    "SUITES",
    "FramePreprocessor",
    "DemoMilestones",
    "quat2axisangle",
    "proprio_from_obs",
    "run_episode",
    "evaluate",
    "build_arg_parser",
]

RESULTS_FORMAT = "vpa-eval-v1"
LOGGER = logging.getLogger("vpa.eval")
SUITES: Tuple[str, ...] = ("libero_10", "libero_spatial", "libero_object", "libero_goal", "libero_90")

# Training (HDF5) camera key -> raw LIBERO observation key.
CAMERA_OBS_KEYS: Dict[str, str] = {
    "agentview_rgb": "agentview_image",
    "eye_in_hand_rgb": "robot0_eye_in_hand_image",
}


# ---------------------------------------------------------------------------------------------
# Observation mapping (LIBERO create_dataset.py conventions)
# ---------------------------------------------------------------------------------------------
def quat2axisangle(quat: np.ndarray) -> np.ndarray:
    """(x, y, z, w) quaternion -> axis-angle vector, as ``robosuite.utils.transform_utils.quat2axisangle``.

    This is the conversion LIBERO's ``create_dataset.py`` used for ``ee_ori``. The input is not modified.

    Returns:
        ``[3]`` float64 (unit axis scaled by the angle in radians).
    """
    q = np.array(quat, dtype=np.float64).reshape(4)
    q[3] = min(1.0, max(-1.0, q[3]))
    den = np.sqrt(1.0 - q[3] * q[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (q[:3] * 2.0 * math.acos(q[3])) / den


def _ee_pos(obs: Dict[str, np.ndarray]) -> np.ndarray:
    return np.asarray(obs["robot0_eef_pos"], dtype=np.float64).reshape(-1)


def _ee_ori(obs: Dict[str, np.ndarray]) -> np.ndarray:
    return quat2axisangle(obs["robot0_eef_quat"])


PROPRIO_FROM_OBS: Dict[str, Callable[[Dict[str, np.ndarray]], np.ndarray]] = {
    "ee_pos": _ee_pos,
    "ee_ori": _ee_ori,
    "ee_states": lambda obs: np.concatenate([_ee_pos(obs), _ee_ori(obs)]),
    "gripper_states": lambda obs: np.asarray(obs["robot0_gripper_qpos"], dtype=np.float64).reshape(-1),
    "joint_states": lambda obs: np.asarray(obs["robot0_joint_pos"], dtype=np.float64).reshape(-1),
}


def proprio_from_obs(obs: Dict[str, np.ndarray], keys: Sequence[str]) -> np.ndarray:
    """S_t from a raw LIBERO observation, concatenated in the order of the training keys.

    Args:
        obs: raw observation dict of the LIBERO environment.
        keys: the checkpoint's ``proprio_keys`` (e.g. ee_pos, ee_ori, gripper_states, joint_states).

    Returns:
        ``[d_s]`` float32.
    """
    unknown = [k for k in keys if k not in PROPRIO_FROM_OBS]
    if unknown:
        raise KeyError(f"no LIBERO observation mapping for proprioception keys {unknown}; "
                       f"supported: {sorted(PROPRIO_FROM_OBS)}")
    return np.concatenate([PROPRIO_FROM_OBS[k](obs) for k in keys]).astype(np.float32)


class FramePreprocessor:
    """The training-time frame transform of ``LiberoHDF5Dataset`` applied to raw environment frames.

    ``__call__`` reuses ``LiberoHDF5Dataset._to_tensor`` itself, so evaluation frames are processed
    by the same code as training frames (180-degree rotation, scaling to [0, 1], antialiased resize).

    Args:
        rotate_180: rotate frames by 180 degrees.
        image_size: model input size (H_img, W_img).

    Shapes:
        ``[H, W, C]`` uint8 -> ``[C, H_img, W_img]`` float32
    """

    _to_tensor = LiberoHDF5Dataset._to_tensor

    def __init__(self, rotate_180: bool, image_size: Tuple[int, int]) -> None:
        self.rotate_180 = bool(rotate_180)
        self.image_size: Optional[Tuple[int, int]] = (int(image_size[0]), int(image_size[1]))

    def __call__(self, raw: np.ndarray) -> Tensor:
        if np.asarray(raw).ndim != 3:
            raise ValueError(f"expected one frame [H, W, C], got shape {np.asarray(raw).shape}")
        return self._to_tensor(raw)


def _normalise(text: str) -> str:
    return " ".join(str(text).lower().split())


# ---------------------------------------------------------------------------------------------
# Milestone frames I_g^(1..M) from demonstrations
# ---------------------------------------------------------------------------------------------
class DemoMilestones:
    """Milestone goal frames taken from a demonstration of the same task (see the module docstring).

    The demonstration file of task ``i`` is ``<task.name>_demo.hdf5``. It is searched for among the
    files the checkpoint was trained on, then under ``data_root`` (LIBERO layout
    ``<suite>/<task.name>_demo.hdf5`` or flat), then in LIBERO's configured datasets folder.

    Args:
        data_info: ``checkpoint["extra"]["data"]`` (training files and preprocessing record).
        cfg: model configuration (image size, H, nu).
        rotate_180: frame rotation (as for the observations).
        data_root: optional folder with LIBERO demonstration files.
        libero_datasets: optional LIBERO datasets folder (``get_libero_path("datasets")``).
        fixed_demo: use this demonstration index for every trial instead of cycling.
    """

    def __init__(
        self,
        data_info: Dict[str, Any],
        cfg: VPAConfig,
        rotate_180: bool,
        data_root: Optional[str] = None,
        libero_datasets: Optional[str] = None,
        fixed_demo: Optional[int] = None,
    ) -> None:
        self.data_info = data_info
        self.cfg = cfg
        self.rotate_180 = bool(rotate_180)
        self.roots = [r for r in (data_root, libero_datasets) if r]
        self.fixed_demo = fixed_demo
        self._cache: Dict[int, LiberoHDF5Dataset] = {}

    def find_file(self, suite: Any, task_id: int, task: Any) -> Optional[str]:
        name = f"{task.name}_demo.hdf5"
        for path in self.data_info.get("files", []):
            if os.path.basename(path) == name and os.path.isfile(path):
                return path
        relative = getattr(suite, "get_task_demonstration", None)
        for root in self.roots:
            candidates = [os.path.join(root, name)]
            if relative is not None:
                candidates.insert(0, os.path.join(root, relative(task_id)))
            for candidate in candidates:
                if os.path.isfile(candidate):
                    return candidate
        return None

    def _dataset(self, suite: Any, task_id: int, task: Any) -> LiberoHDF5Dataset:
        if task_id not in self._cache:
            path = self.find_file(suite, task_id, task)
            if path is None:
                raise FileNotFoundError(
                    f"no demonstration file '{task.name}_demo.hdf5' for task {task_id}; pass --data-root "
                    "(the LIBERO datasets folder) or use --milestone-source initial"
                )
            self._cache[task_id] = LiberoHDF5Dataset(
                path,
                self.cfg.horizon,
                predictor_stride=self.cfg.predictor_stride,
                image_size=(int(self.cfg.image_size[0]), int(self.cfg.image_size[1])),
                camera_key=self.data_info["camera_key"],
                proprio_keys=self.data_info["proprio_keys"],
                rotate_180=self.rotate_180,
                primitive_source="gripper",              # labels are not used here
                milestone_source=self.data_info["milestone_source"],
                gripper_window=self.data_info["gripper_window"],
            )
        return self._cache[task_id]

    def frames(self, suite: Any, task_id: int, task: Any, trial: int) -> Tuple[Tensor, Dict[str, Any]]:
        """Milestone frames ``[1, M, C, H_img, W_img]`` for one trial, and a record of their source."""
        ds = self._dataset(suite, task_id, task)
        episode = (self.fixed_demo if self.fixed_demo is not None else trial) % len(ds.episodes)
        ep = ds.episodes[episode]
        record = {
            "file": os.path.basename(ep.path),
            "demo": ep.demo,
            "milestone_frame_indices": list(ep.milestones),
            "instruction": ds.instructions[ep.task],
        }
        return ds.milestone_frames(episode).unsqueeze(0), record

    def release(self, task_id: int) -> None:
        ds = self._cache.pop(task_id, None)
        if ds is not None:
            ds.close()


# ---------------------------------------------------------------------------------------------
# LIBERO access (imported lazily so the rest of this file works without LIBERO)
# ---------------------------------------------------------------------------------------------
def import_libero() -> SimpleNamespace:
    """``benchmark``, ``get_libero_path`` and ``OffScreenRenderEnv`` from LIBERO, with clear errors."""
    try:
        from libero.libero import benchmark, get_libero_path
        from libero.libero.envs import OffScreenRenderEnv
    except EOFError as exc:  # LIBERO prompts for its dataset folder when ~/.libero/config.yaml is missing
        raise RuntimeError(
            "LIBERO asked for its dataset folder on first import. Run `python -c \"import libero.libero\"` "
            "once interactively (or set LIBERO_CONFIG_PATH to a folder with config.yaml), then retry."
        ) from exc
    except ImportError as exc:
        raise ImportError(
            "LIBERO is not installed; see https://github.com/Lifelong-Robot-Learning/LIBERO"
        ) from exc
    return SimpleNamespace(benchmark=benchmark, get_libero_path=get_libero_path,
                           OffScreenRenderEnv=OffScreenRenderEnv)


def make_suite(libero: SimpleNamespace, name: str, task_order_index: int = 0) -> Any:
    """``libero.libero.benchmark.get_benchmark(name)(task_order_index)``."""
    if name not in SUITES:
        raise ValueError(f"unknown suite {name!r}; choose from {SUITES}")
    return libero.benchmark.get_benchmark(name)(task_order_index)


def make_env(libero: SimpleNamespace, task: Any, camera_size: int, horizon: int, retries: int = 3) -> Any:
    """LIBERO ``OffScreenRenderEnv`` for one task (retried, since off-screen contexts can fail transiently)."""
    bddl = os.path.join(libero.get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env_args = {"bddl_file_name": bddl, "camera_heights": camera_size, "camera_widths": camera_size,
                "horizon": horizon}
    last: Optional[BaseException] = None
    for attempt in range(retries):
        try:
            return libero.OffScreenRenderEnv(**env_args)
        except Exception as exc:  # renderer / context creation errors
            last = exc
            LOGGER.warning("creating the environment for %s failed (attempt %d/%d): %s",
                           task.name, attempt + 1, retries, exc)
            time.sleep(2.0)
    raise RuntimeError(f"could not create the LIBERO environment for {task.name}") from last


def load_init_states(suite: Any, task_id: int, get_libero_path: Optional[Callable[[str], str]] = None) -> Any:
    """LIBERO's fixed initial states of one task.

    ``Benchmark.get_task_init_states`` calls ``torch.load`` with the default ``weights_only``; on
    torch >= 2.6 that rejects the numpy arrays in LIBERO's official init-state files, so they are
    then loaded again with ``weights_only=False`` (they are LIBERO's own asset files).
    """
    try:
        states = suite.get_task_init_states(task_id)
    except pickle.UnpicklingError:
        if get_libero_path is None:
            raise
        task = suite.get_task(task_id)
        path = os.path.join(get_libero_path("init_states"), task.problem_folder, task.init_states_file)
        LOGGER.info("loading LIBERO init states with weights_only=False: %s", path)
        states = torch.load(path, weights_only=False)
    if len(states) == 0:
        raise ValueError(f"task {task_id} has no initial states")
    return states


# ---------------------------------------------------------------------------------------------
# One episode
# ---------------------------------------------------------------------------------------------
class EnvironmentStepError(RuntimeError):
    """An exception raised by the simulator (not by the model); the episode is counted as a failure."""


def _env_call(fn: Callable[..., Any], *args: Any) -> Any:
    try:
        return fn(*args)
    except Exception as exc:
        raise EnvironmentStepError(f"{getattr(fn, '__name__', fn)} failed: {exc!r}") from exc


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


@dataclasses.dataclass
class EpisodeContext:
    """Fixed inputs of every episode of a run."""

    preprocess: FramePreprocessor
    camera_obs_key: str
    proprio_keys: Sequence[str]
    max_steps: int
    num_steps_wait: int
    stop_on_task_complete: bool
    device: torch.device
    record_video: bool = False
    rotate_view: bool = True


@torch.no_grad()
def run_episode(
    pipe: VPAInferencePipeline,
    env: Any,
    ctx: EpisodeContext,
    *,
    instruction: str,
    init_state: Any,
    milestones: Optional[Tensor],
    generator: torch.Generator,
) -> Tuple[Dict[str, Any], List[np.ndarray]]:
    """Roll out one episode with ``pipe.step`` decision steps and H_t-action execution prefixes.

    Args:
        pipe: calibrated pipeline.
        env: LIBERO environment (``reset``, ``set_init_state``, ``step``).
        ctx: run-wide settings.
        instruction: language instruction x.
        init_state: LIBERO initial simulator state.
        milestones: ``[1, M, C, H_img, W_img]`` goal frames, or None for the initial observation (M = 1).
        generator: CPU generator for the flow noise xi of every decision step.

    Returns:
        (episode record, list of ``[H, W, C]`` uint8 frames if ``ctx.record_video``).
    """
    t_start = time.perf_counter()
    action_dim, horizon = pipe.solver.action_dim, pipe.solver.horizon
    depth = pipe.solver.num_integration_steps + 3                                   # Prop. 5.1
    _env_call(env.reset)
    obs = _env_call(env.set_init_state, init_state)
    settle = np.zeros(action_dim, dtype=np.float64)
    for _ in range(ctx.num_steps_wait):                                             # physics settling
        obs, _, _, _ = _env_call(env.step, settle)

    video: List[np.ndarray] = []

    def view(o: Dict[str, np.ndarray]) -> np.ndarray:
        img = np.asarray(o[ctx.camera_obs_key])
        return np.ascontiguousarray(img[::-1, ::-1] if ctx.rotate_view else img)

    if milestones is None:                                                          # fallback, M = 1
        milestones = ctx.preprocess(obs[ctx.camera_obs_key]).unsqueeze(0).unsqueeze(0)
    pipe.reset(instruction, milestones.to(ctx.device))                              # c_text, z_g^(1..M)
    if ctx.record_video:
        video.append(view(obs))

    env_steps, decisions = 0, 0
    success, success_step, complete_step, stopped_by_pointer = False, None, None, False
    horizons: List[int] = []
    sigmas: List[float] = []
    latencies: List[float] = []
    primitives: Counter = Counter()
    while env_steps < ctx.max_steps and not success:
        frame = ctx.preprocess(obs[ctx.camera_obs_key]).unsqueeze(0).to(ctx.device)          # [1, C, H, W]
        state = torch.from_numpy(proprio_from_obs(obs, ctx.proprio_keys)).unsqueeze(0).to(ctx.device)  # [1, d_s]
        noise = torch.randn((1, horizon, action_dim), generator=generator).to(ctx.device)     # xi
        _synchronize(ctx.device)
        t0 = time.perf_counter()
        out = pipe.step(frame, state, noise=noise)
        _synchronize(ctx.device)
        latencies.append((time.perf_counter() - t0) * 1e3)
        if out.num_sequential_evaluations != depth:
            raise RuntimeError(f"decision step used {out.num_sequential_evaluations} sequential evaluations, "
                               f"expected K + 3 = {depth}")
        decisions += 1
        h_t = int(out.executed_horizon[0])                                                     # Eq. 15
        horizons.append(h_t)
        sigmas.append(float(out.sigma[0]))
        primitives[int(out.primitive[0])] += 1
        if complete_step is None and bool(out.task_complete[0]):                              # Eq. 9a at m_t = M
            complete_step = env_steps
            if ctx.stop_on_task_complete:
                stopped_by_pointer = True
                break
        prefix = out.action_chunk[0, :h_t].float().cpu().numpy().astype(np.float64)          # [H_t, d_a]
        for action in prefix:
            obs, _, done, _ = _env_call(env.step, np.clip(action, -1.0, 1.0))
            env_steps += 1
            if ctx.record_video:
                video.append(view(obs))
            if done:
                success, success_step = True, env_steps
                break
            if env_steps >= ctx.max_steps:
                break

    lat = np.asarray(latencies) if latencies else np.zeros(1)
    record = {
        "success": success,
        "env_steps": env_steps,
        "success_env_step": success_step,
        "decision_steps": decisions,
        "sequential_evaluations_per_step": depth,
        "num_milestones": int(milestones.shape[1]),
        "final_milestone_pointer": int(pipe.milestone_pointer[0]) if pipe.milestone_pointer is not None else None,
        "task_complete_env_step": complete_step,
        "premature_task_complete": complete_step is not None and (success_step is None or complete_step < success_step),
        "stopped_by_task_complete": stopped_by_pointer,
        "executed_horizon_mean": float(np.mean(horizons)) if horizons else None,
        "executed_horizon_min": int(min(horizons)) if horizons else None,
        "sigma_mean": float(np.mean(sigmas)) if sigmas else None,
        "primitive_counts": {str(k): v for k, v in sorted(primitives.items())},
        "step_latency_ms_mean": float(lat.mean()),
        "step_latency_ms_p95": float(np.percentile(lat, 95)),
        "wall_time_s": time.perf_counter() - t_start,
    }
    return record, video


def save_video(path: str, frames: List[np.ndarray], fps: int) -> Optional[str]:
    """Write an MP4 with imageio; returns the path, or None (with a warning) if that is not possible."""
    if not frames:
        return None
    try:
        import imageio.v2 as imageio
    except ImportError:
        LOGGER.warning("imageio is not installed; videos are disabled (pip install imageio imageio-ffmpeg)")
        return None
    try:
        imageio.mimwrite(path, frames, fps=fps)
    except Exception as exc:  # e.g. missing ffmpeg backend
        LOGGER.warning("could not write %s: %s", path, exc)
        return None
    return path


# ---------------------------------------------------------------------------------------------
# Benchmark loop
# ---------------------------------------------------------------------------------------------
def _close_logging() -> None:
    for handler in list(LOGGER.handlers):
        handler.close()
        LOGGER.removeHandler(handler)


def setup_logging(out_dir: str) -> logging.Logger:
    """Log to stdout and to ``<out_dir>/eval.log``."""
    _close_logging()
    LOGGER.setLevel(logging.INFO)
    LOGGER.propagate = False
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%Y-%m-%d %H:%M:%S")
    for handler in (logging.StreamHandler(sys.stdout), logging.FileHandler(os.path.join(out_dir, "eval.log"))):
        handler.setFormatter(fmt)
        LOGGER.addHandler(handler)
    return LOGGER


def _write_json(path: str, obj: Dict[str, Any]) -> None:
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def evaluate(
    args: argparse.Namespace,
    suite: Optional[Any] = None,
    env_factory: Optional[Callable[[Any], Any]] = None,
    pipe: Optional[VPAInferencePipeline] = None,
) -> Dict[str, Any]:
    """Evaluate a checkpoint on a LIBERO suite and write ``<out>/eval_metrics.json``.

    Args:
        args: parsed command-line arguments.
        suite: a LIBERO benchmark object (default: built from ``--suite-name`` through LIBERO).
        env_factory: ``task -> environment`` (default: LIBERO ``OffScreenRenderEnv``).
        pipe: an already loaded pipeline (default: ``load_pipeline(--checkpoint)``).

    Returns:
        the results dictionary that was written.
    """
    if args.num_trials_per_task < 1 or args.max_steps < 1 or args.num_steps_wait < 0:
        raise ValueError("--num-trials-per-task and --max-steps must be >= 1, --num-steps-wait >= 0")
    os.makedirs(args.out, exist_ok=True)
    logger = setup_logging(args.out)
    seed_everything(args.seed)
    device = resolve_device(args.device)

    # ---- model and the conventions it was trained with
    meta = load_checkpoint(args.checkpoint)
    if meta["stage"] != "policy":
        raise ValueError(f"{args.checkpoint} is a stage-{meta['stage']!r} checkpoint; closed-loop evaluation needs "
                         "the stage-2 policy checkpoint (policy_final.pt)")
    cfg = VPAConfig(**meta["config"])
    data_info = meta["extra"]["data"]
    if pipe is None:
        pipe = load_pipeline(args.checkpoint, device=str(device))
    if not (pipe.solver.is_calibrated and bool(pipe.tracker.frozen)):
        raise RuntimeError("the pipeline is not calibrated (gamma*, mu_Z, mu_S, sigma_S missing)")
    pipe.eval()
    rotate = bool(data_info["rotate_180"]) if args.rotate == "checkpoint" else args.rotate == "on"
    camera_key = data_info["camera_key"]
    if camera_key not in CAMERA_OBS_KEYS:
        raise KeyError(f"no LIBERO observation for camera key {camera_key!r}; supported: {sorted(CAMERA_OBS_KEYS)}")
    proprio_keys = list(data_info["proprio_keys"])
    unknown = [k for k in proprio_keys if k not in PROPRIO_FROM_OBS]
    if unknown:
        raise KeyError(f"no LIBERO observation mapping for proprioception keys {unknown}")
    ctx = EpisodeContext(
        preprocess=FramePreprocessor(rotate, tuple(cfg.image_size)),
        camera_obs_key=CAMERA_OBS_KEYS[camera_key],
        proprio_keys=proprio_keys,
        max_steps=args.max_steps,
        num_steps_wait=args.num_steps_wait,
        stop_on_task_complete=args.stop_on_task_complete,
        device=device,
        record_video=args.save_video,
        rotate_view=rotate,
    )

    # ---- benchmark
    libero = None
    if suite is None or env_factory is None:
        libero = import_libero()
    if suite is None:
        suite = make_suite(libero, args.suite_name, args.task_order_index)
    if env_factory is None:
        assert libero is not None
        env_horizon = args.max_steps + args.num_steps_wait + 50

        def env_factory(task: Any) -> Any:
            return make_env(libero, task, args.camera_size, env_horizon)

    n_tasks = int(suite.n_tasks)
    task_ids = list(range(n_tasks)) if not args.task_ids else list(args.task_ids)
    bad = [t for t in task_ids if not 0 <= t < n_tasks]
    if bad:
        raise ValueError(f"task ids {bad} are outside [0, {n_tasks})")
    milestone_source: Optional[DemoMilestones] = None
    if args.milestone_source == "demo":
        libero_datasets = None
        if libero is not None:
            try:
                libero_datasets = libero.get_libero_path("datasets")
            except Exception:  # missing key in a custom LIBERO config
                libero_datasets = None
        milestone_source = DemoMilestones(data_info, cfg, rotate, args.data_root, libero_datasets,
                                          args.milestone_demo)
    trained_on = {_normalise(x) for x in data_info.get("instructions", [])}
    get_path = libero.get_libero_path if libero is not None else None
    video_dir = os.path.join(args.out, "videos")
    if args.save_video:
        os.makedirs(video_dir, exist_ok=True)

    results: Dict[str, Any] = {
        "format": RESULTS_FORMAT,
        "status": "running",
        "checkpoint": os.path.abspath(args.checkpoint),
        "suite": args.suite_name,
        "task_order_index": args.task_order_index,
        "task_ids": task_ids,
        "num_trials_per_task": args.num_trials_per_task,
        "max_steps": args.max_steps,
        "num_steps_wait": args.num_steps_wait,
        "seed": args.seed,
        "device": str(device),
        "milestone_source": args.milestone_source,
        "milestone_rule": data_info["milestone_source"],
        "stop_on_task_complete": args.stop_on_task_complete,
        "rotate_180": rotate,
        "camera_key": camera_key,
        "camera_size": args.camera_size,
        "proprio_keys": proprio_keys,
        "config": _json_safe(dataclasses.asdict(cfg)),
        "started_at": _now(),
        "tasks": [],
    }
    out_path = os.path.join(args.out, "eval_metrics.json")
    logger.info("evaluating %s on %s: %d tasks x %d trials, max %d steps, K = %d, H = %d, device %s",
                args.checkpoint, args.suite_name, len(task_ids), args.num_trials_per_task, args.max_steps,
                cfg.integration_steps, cfg.horizon, device)
    t_run = time.perf_counter()
    total_success, total_trials = 0, 0
    try:
        for position, task_id in enumerate(task_ids):
            task = suite.get_task(task_id)
            instruction = str(task.language)
            init_states = load_init_states(suite, task_id, get_path)
            in_training = _normalise(instruction) in trained_on
            if not in_training:
                logger.warning("task %d (%s) is not among the checkpoint's training instructions", task_id, instruction)
            t_task = time.perf_counter()
            episodes: List[Dict[str, Any]] = []
            env = env_factory(task)
            try:
                for trial in range(args.num_trials_per_task):
                    episode_seed = args.seed + 1000 * task_id + trial
                    if hasattr(env, "seed"):
                        env.seed(episode_seed)
                    generator = torch.Generator().manual_seed(episode_seed)
                    milestones, source = (None, {"source": "initial observation"})
                    if milestone_source is not None:
                        milestones, source = milestone_source.frames(suite, task_id, task, trial)
                        if _normalise(source["instruction"]) != _normalise(instruction):
                            logger.warning("milestone demonstration instruction %r differs from the task's %r",
                                           source["instruction"], instruction)
                    init_index = trial % len(init_states)
                    try:
                        record, frames = run_episode(pipe, env, ctx, instruction=instruction,
                                                     init_state=init_states[init_index], milestones=milestones,
                                                     generator=generator)
                    except EnvironmentStepError as exc:  # simulator failure: a failed trial, fresh environment
                        logger.exception("task %d trial %d: environment error", task_id, trial)
                        record, frames = {"success": False, "error": str(exc)}, []
                        with contextlib.suppress(Exception):
                            env.close()
                        env = env_factory(task)
                    record.update({"trial": trial, "init_state_index": init_index, "seed": episode_seed,
                                   "milestones_from": source})
                    if args.save_video and frames:
                        tag = "success" if record["success"] else "failure"
                        record["video"] = save_video(
                            os.path.join(video_dir, f"task{task_id:02d}_trial{trial:02d}_{tag}.mp4"),
                            frames, args.video_fps)
                    episodes.append(record)
                    n_ok = sum(int(e["success"]) for e in episodes)
                    logger.info("[task %d/%d id %d] trial %d/%d %s | steps %s decisions %s H_t %.1f "
                                "latency %.2f ms | task success %d/%d",
                                position + 1, len(task_ids), task_id, trial + 1, args.num_trials_per_task,
                                "SUCCESS" if record["success"] else "failure", record.get("env_steps", "-"),
                                record.get("decision_steps", "-"), record.get("executed_horizon_mean") or 0.0,
                                record.get("step_latency_ms_mean", 0.0), n_ok, len(episodes))
            finally:
                env.close()
                if milestone_source is not None:
                    milestone_source.release(task_id)
            successes = sum(int(e["success"]) for e in episodes)
            total_success += successes
            total_trials += len(episodes)
            results["tasks"].append({
                "task_id": task_id,
                "name": str(task.name),
                "instruction": instruction,
                "in_training_data": in_training,
                "successes": successes,
                "trials": len(episodes),
                "success_rate": successes / max(1, len(episodes)),
                "premature_task_complete_rate": sum(int(bool(e.get("premature_task_complete"))) for e in episodes)
                / max(1, len(episodes)),
                "wall_time_s": time.perf_counter() - t_task,
                "episodes": episodes,
            })
            results.update(successes=total_success, trials=total_trials,
                           success_rate=total_success / max(1, total_trials),
                           wall_time_s=time.perf_counter() - t_run)
            _write_json(out_path, results)
            logger.info("task %d (%s): success rate %.3f (%d/%d) | running total %.3f (%d/%d)", task_id,
                        instruction, successes / max(1, len(episodes)), successes, len(episodes),
                        total_success / max(1, total_trials), total_success, total_trials)
        results["status"] = "complete"
    except KeyboardInterrupt:
        results["status"] = "interrupted"
        raise
    finally:
        results.update(successes=total_success, trials=total_trials,
                       success_rate=total_success / max(1, total_trials),
                       wall_time_s=time.perf_counter() - t_run, finished_at=_now())
        if results["status"] == "running":
            results["status"] = "failed"
        _write_json(out_path, results)
    logger.info("%s overall success rate %.3f (%d/%d) in %.1f min; results in %s", args.suite_name,
                results["success_rate"], total_success, total_trials, results["wall_time_s"] / 60.0, out_path)
    return results


# ---------------------------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Closed-loop evaluation of a trained VPA policy on LIBERO.")
    p.add_argument("--checkpoint", default="runs/libero10/policy_final.pt", help="stage-2 checkpoint")
    p.add_argument("--suite-name", choices=SUITES, default="libero_10")
    p.add_argument("--num-trials-per-task", type=int, default=20)
    p.add_argument("--max-steps", type=int, default=600, help="environment steps per episode (LIBERO default 600)")
    p.add_argument("--num-steps-wait", type=int, default=5, help="zero-action settling steps after reset")
    p.add_argument("--out", default="eval_results", help="output directory (eval_metrics.json, eval.log)")
    p.add_argument("--device", default="auto", help="auto, cuda, mps or cpu")
    p.add_argument("--save-video", action="store_true", help="write one MP4 per episode to <out>/videos")
    p.add_argument("--video-fps", type=int, default=20, help="LIBERO runs its controller at 20 Hz")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--task-ids", type=int, nargs="+", help="evaluate only these task indices (sharding)")
    p.add_argument("--task-order-index", type=int, default=0, help="LIBERO task order")
    p.add_argument("--camera-size", type=int, default=128, help="render resolution (LIBERO datasets use 128)")
    p.add_argument("--milestone-source", choices=("demo", "initial"), default="demo",
                   help="goal frames from a demonstration of the task, or the initial observation (M = 1)")
    p.add_argument("--data-root", help="folder with LIBERO demonstration files (<suite>/<task>_demo.hdf5)")
    p.add_argument("--milestone-demo", type=int, help="use this demonstration index for every trial")
    p.add_argument("--stop-on-task-complete", action="store_true",
                   help="end the episode when the Eq. 9a test holds at m_t = M (the paper's termination rule)")
    p.add_argument("--rotate", choices=("checkpoint", "on", "off"), default="checkpoint",
                   help="180-degree frame rotation (default: as in training)")
    p.add_argument("--self-test", action="store_true", help="run the offline self-test and exit")
    return p


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_arg_parser().parse_args(argv)
    if args.self_test:
        _self_test()
        return
    evaluate(args)


# ---------------------------------------------------------------------------------------------
# Self-test (run: python eval.py --self-test)
# ---------------------------------------------------------------------------------------------
def _axisangle_to_quat(vec: np.ndarray) -> np.ndarray:
    """Independent reference: q = (u sin(theta / 2), cos(theta / 2)), theta = ||v||, u = v / theta."""
    v = np.asarray(vec, dtype=np.float64)
    theta = float(np.linalg.norm(v))
    if theta == 0.0:
        return np.array([0.0, 0.0, 0.0, 1.0])
    return np.concatenate([v / theta * math.sin(theta / 2.0), [math.cos(theta / 2.0)]])


class _FakeTask(SimpleNamespace):
    pass


class _FakeSuite:
    """LIBERO benchmark stand-in: tasks, init states (= start frame index) and demo paths."""

    def __init__(self, tasks: List[_FakeTask], num_init_states: int) -> None:
        self.tasks = tasks
        self.n_tasks = len(tasks)
        self.num_init_states = num_init_states

    def get_task(self, i: int) -> _FakeTask:
        return self.tasks[i]

    def get_task_init_states(self, i: int) -> List[np.ndarray]:
        return [np.array([k]) for k in range(self.num_init_states)]

    def get_task_demonstration(self, i: int) -> str:
        return f"fake/{self.tasks[i].name}_demo.hdf5"


class _FakeEnv:
    """LIBERO environment stand-in replaying the raw observations of one synthetic demonstration.

    Observations are the stored frames and states converted back to LIBERO's raw keys (the end-effector
    orientation as a quaternion). ``history`` holds the actions of every episode (one list per
    ``reset``); each ``step`` advances one frame, and ``done`` becomes True after ``success_after``
    steps of an episode (settling steps included), if given. ``fail`` makes ``step`` raise.
    """

    def __init__(self, path: str, demo: str, task_index: int, success_after: Optional[int],
                 fail: bool = False) -> None:
        import h5py

        with h5py.File(path, "r") as f:
            obs = f["data"][demo]["obs"]
            self.data = {k: obs[k][()] for k in ("agentview_rgb", "eye_in_hand_rgb", "ee_pos", "ee_ori",
                                                  "gripper_states", "joint_states")}
        self.length = self.data["ee_pos"].shape[0]
        self.task_index = task_index
        self.success_after = success_after
        self.fail = fail
        self.history: List[List[np.ndarray]] = []
        self.t = 0
        self.closed = False

    def seed(self, seed: int) -> None:
        self.last_seed = seed

    def obs(self) -> Dict[str, np.ndarray]:
        t = self.t
        return {
            "agentview_image": self.data["agentview_rgb"][t],
            "robot0_eye_in_hand_image": self.data["eye_in_hand_rgb"][t],
            "robot0_eef_pos": self.data["ee_pos"][t],
            "robot0_eef_quat": _axisangle_to_quat(self.data["ee_ori"][t]),
            "robot0_gripper_qpos": self.data["gripper_states"][t],
            "robot0_joint_pos": self.data["joint_states"][t],
        }

    def reset(self) -> Dict[str, np.ndarray]:
        self.t = 0
        self.history.append([])
        return self.obs()

    def set_init_state(self, state: np.ndarray) -> Dict[str, np.ndarray]:
        self.t = int(state[0]) % self.length
        return self.obs()

    def step(self, action: np.ndarray) -> Tuple[Dict[str, np.ndarray], float, bool, Dict[str, Any]]:
        if self.fail:
            raise ValueError("simulated physics failure")
        self.history[-1].append(np.array(action, dtype=np.float64))
        self.t = min(self.t + 1, self.length - 1)
        done = self.success_after is not None and len(self.history[-1]) >= self.success_after
        return self.obs(), float(done), done, {}

    def close(self) -> None:
        self.closed = True


def _self_test() -> None:
    try:
        from .train import _TINY_CONFIG, _test_args
        from .train import _close_logging as _close_train_logging
        from .train import run as train_run
    except ImportError:
        from train import _TINY_CONFIG, _test_args
        from train import _close_logging as _close_train_logging
        from train import run as train_run

    # ---- quat2axisangle against the independent inverse; identity; clipping; input untouched
    rng = np.random.default_rng(0)
    for _ in range(200):
        axis = rng.normal(size=3)
        axis /= np.linalg.norm(axis)
        angle = float(rng.uniform(1e-3, math.pi - 1e-3))
        q = _axisangle_to_quat(axis * angle)
        q_copy = q.copy()
        assert np.allclose(quat2axisangle(q), axis * angle, atol=1e-9) and np.array_equal(q, q_copy)
    assert np.array_equal(quat2axisangle(np.array([0.0, 0.0, 0.0, 1.0])), np.zeros(3))
    assert np.array_equal(quat2axisangle(np.array([0.0, 0.0, 0.0, 1.0 + 1e-9])), np.zeros(3))

    with tempfile.TemporaryDirectory() as tmp:
        # ---- a tiny trained model on synthetic LIBERO-format data (train.py's self-test recipe)
        data_dir = os.path.join(tmp, "data")
        os.makedirs(data_dir)
        names = ("KITCHEN_SCENE1_put_the_bowl_on_the_plate", "LIVING_ROOM_SCENE2_stack_the_blocks")
        _write_synthetic_libero(os.path.join(data_dir, f"{names[0]}_demo.hdf5"), (14, 11, 16, 12), seed=5)
        _write_synthetic_libero(os.path.join(data_dir, f"{names[1]}_demo.hdf5"), (13, 10, 15), seed=6,
                                instruction=None)
        cfg_path = os.path.join(tmp, "tiny.json")
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(_TINY_CONFIG, f)
        train_out = os.path.join(tmp, "run")
        train_run(_test_args(data_dir, train_out, cfg_path, ["--jepa-steps", "2", "--policy-steps", "2"]))
        _close_train_logging()
        ckpt = os.path.join(train_out, "policy_final.pt")
        meta = load_checkpoint(ckpt)
        cfg = VPAConfig(**meta["config"])
        data_info = meta["extra"]["data"]
        demo_ds = [LiberoHDF5Dataset(os.path.join(data_dir, f"{n}_demo.hdf5"), cfg.horizon,
                                     image_size=tuple(cfg.image_size), milestone_source="gripper",
                                     primitive_source="gripper", gripper_window=2) for n in names]
        tasks = [_FakeTask(name=names[0], language="pick up the red cup and place it on the plate",
                           problem_folder="fake", bddl_file="x.bddl", init_states_file="x.pruned_init"),
                 _FakeTask(name=names[1], language="stack the blocks", problem_folder="fake",
                           bddl_file="y.bddl", init_states_file="y.pruned_init")]
        suite = _FakeSuite(tasks, num_init_states=3)
        wait, max_steps, success_steps = 2, 9, 5
        envs: List[_FakeEnv] = []
        failing = {"on": False}

        def factory(task: Any) -> _FakeEnv:
            i = names.index(task.name)
            env = _FakeEnv(os.path.join(data_dir, f"{task.name}_demo.hdf5"), "demo_0", i,
                           success_after=wait + success_steps if i == 0 else None, fail=failing["on"])
            envs.append(env)
            return env

        def parse(extra: Sequence[str]) -> argparse.Namespace:
            return build_arg_parser().parse_args(
                ["--checkpoint", ckpt, "--device", "cpu", "--num-trials-per-task", "2", "--max-steps",
                 str(max_steps), "--num-steps-wait", str(wait), *extra])

        # ---- observation mapping == training data (frames bit-exact, proprioception via the quaternion)
        assert data_info["camera_key"] == "agentview_rgb" and bool(data_info["rotate_180"])
        pre = FramePreprocessor(True, tuple(cfg.image_size))
        probe = factory(tasks[0])
        for t in (0, 5, probe.length - 1):
            probe.t = t
            o = probe.obs()
            assert torch.equal(pre(o["agentview_image"]), demo_ds[0].frame(0, t))
            assert np.allclose(proprio_from_obs(o, data_info["proprio_keys"]), demo_ds[0].proprioception(0)[t],
                               atol=1e-5)
        envs.clear()

        # ---- record what the loop feeds the pipeline
        pipe = load_pipeline(ckpt)
        calls: List[Dict[str, Any]] = []
        resets: List[Tensor] = []
        step_fn, reset_fn = pipe.step, pipe.reset

        def recording_step(frame: Tensor, state: Tensor, noise: Optional[Tensor] = None) -> Any:
            out = step_fn(frame, state, noise=noise)
            env = envs[-1]
            calls.append({"frame": frame.clone(), "state": state.clone(), "t": env.t, "out": out, "env": env,
                          "episode": len(env.history) - 1, "steps_before": len(env.history[-1])})
            return out

        def recording_reset(instruction: str, milestones: Tensor) -> None:
            resets.append(milestones.clone())
            reset_fn(instruction, milestones)

        pipe.step, pipe.reset = recording_step, recording_reset
        args = parse(["--out", os.path.join(tmp, "eval_a")])
        res = evaluate(args, suite=suite, env_factory=factory, pipe=pipe)
        with open(os.path.join(args.out, "eval_metrics.json"), encoding="utf-8") as f:
            saved = json.load(f)
        assert saved["status"] == "complete" and saved["trials"] == 4 and len(saved["tasks"]) == 2
        assert saved["tasks"][0]["success_rate"] == 1.0 and saved["tasks"][1]["success_rate"] == 0.0
        assert saved["success_rate"] == 0.5 == res["success_rate"]
        assert len(envs) == 2 and all(e.closed for e in envs)
        for task_res, i in zip(saved["tasks"], (0, 1), strict=True):
            ds = demo_ds[i]
            assert task_res["in_training_data"]
            for ep in task_res["episodes"]:
                assert ep["sequential_evaluations_per_step"] == cfg.integration_steps + 3
                assert ep["milestones_from"]["demo"] == ds.episodes[ep["trial"] % len(ds.episodes)].demo
                assert ep["env_steps"] == (success_steps if i == 0 else max_steps)
                assert ep["success"] == (i == 0) and ep["init_state_index"] == ep["trial"] % 3
        # milestones: the training rule's frames of demonstration (trial mod N), same transform
        assert len(resets) == 4
        for k, ms in enumerate(resets):
            i, trial = divmod(k, 2)
            assert torch.equal(ms, demo_ds[i].milestone_frames(trial % len(demo_ds[i].episodes)).unsqueeze(0))
        # frames / proprioception given to step() == the training transform of the env's current frame
        for c in calls:
            ds = demo_ds[c["env"].task_index]
            assert torch.equal(c["frame"][0], ds.frame(0, c["t"]))
            assert np.allclose(c["state"][0].numpy(), ds.proprioception(0)[c["t"]], atol=1e-5)
        # every episode: zero settling actions, then exactly the clipped H_t prefixes, truncated at the end
        for env in envs:
            assert len(env.history) == 2
            for episode, actions in enumerate(env.history):
                ep_calls = [c for c in calls if c["env"] is env and c["episode"] == episode]
                assert ep_calls[0]["steps_before"] == wait
                assert all(np.array_equal(a, np.zeros(cfg.action_dim)) for a in actions[:wait])
                expected: List[np.ndarray] = []
                for c in ep_calls:
                    assert c["steps_before"] == wait + len(expected)        # replanning after each prefix
                    h_t = int(c["out"].executed_horizon[0])
                    assert cfg.min_horizon <= h_t <= cfg.horizon
                    expected.extend(np.clip(c["out"].action_chunk[0, :h_t].double().numpy(), -1.0, 1.0))
                n = success_steps if env.task_index == 0 else max_steps
                assert len(actions) == wait + n and len(expected) >= n
                assert np.allclose(np.stack(actions[wait:]), np.stack(expected[:n]), rtol=0.0, atol=1e-12)

        # ---- determinism: the same seed reproduces every episode
        envs.clear()
        res_b = evaluate(parse(["--out", os.path.join(tmp, "eval_b")]), suite=suite, env_factory=factory, pipe=pipe)
        varying = ("wall_time_s", "step_latency_ms_mean", "step_latency_ms_p95")
        for ta, tb in zip(res["tasks"], res_b["tasks"], strict=True):
            for ea, eb in zip(ta["episodes"], tb["episodes"], strict=True):
                assert {k: v for k, v in ea.items() if k not in varying} == \
                       {k: v for k, v in eb.items() if k not in varying}

        # ---- initial-frame milestone + the paper's termination rule: Eq. 9a holds at the first decision
        envs.clear()
        res_c = evaluate(parse(["--out", os.path.join(tmp, "eval_c"), "--milestone-source", "initial",
                                "--stop-on-task-complete", "--task-ids", "1"]),
                         suite=suite, env_factory=factory, pipe=pipe)
        assert [t["task_id"] for t in res_c["tasks"]] == [1]
        for ep in res_c["tasks"][0]["episodes"]:
            assert ep["num_milestones"] == 1 and ep["task_complete_env_step"] == 0
            assert ep["stopped_by_task_complete"] and ep["premature_task_complete"]
            assert ep["decision_steps"] == 1 and ep["env_steps"] == 0 and not ep["success"]

        # ---- a simulator exception fails the trial, rebuilds the environment and the run continues
        envs.clear()
        failing["on"] = True
        res_d = evaluate(parse(["--out", os.path.join(tmp, "eval_d"), "--task-ids", "1"]),
                         suite=suite, env_factory=factory, pipe=pipe)
        failing["on"] = False
        eps = res_d["tasks"][0]["episodes"]
        assert res_d["status"] == "complete" and len(eps) == 2 and len(envs) == 3
        assert all(not e["success"] and "simulated physics failure" in e["error"] for e in eps)

        # ---- errors: a stage-1 checkpoint and unknown task ids are rejected
        pipe.step, pipe.reset = step_fn, reset_fn
        for bad_args in (["--checkpoint", os.path.join(train_out, "jepa_final.pt")], ["--task-ids", "7"]):
            try:
                evaluate(parse(["--out", os.path.join(tmp, "eval_err"), *bad_args]), suite=suite,
                         env_factory=factory, pipe=pipe)
                raise AssertionError(f"expected ValueError for {bad_args}")
            except ValueError:
                pass
        for ds in demo_ds:
            ds.close()
        _close_logging()
    print("eval.py self-test passed")


if __name__ == "__main__":
    main()
