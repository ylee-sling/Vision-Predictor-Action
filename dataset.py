# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
"""
dataset.py -- LIBERO demonstrations for offline VPA training (preprint_261008.pdf, Secs. 3-4).

One sample is one control step t of one demonstration. It carries exactly the quantities the
training objectives consume:

    image       I_t        ``[C, H_img, W_img]`` float32 in [0, 1]  E_psi input (Eqs. 1, 7)
    next_image  I_{t+nu}   ``[C, H_img, W_img]`` target frame of Eq. 11 (stride nu, Sec. 3.3)
                (both ``[V, C, H_img, W_img]`` with V >= 2 camera views, see "Camera views" below)
    proprio     S_t        ``[d_s]``   raw proprioception (Eq. 1; standardised later, Sec. 4.5)
    actions     A_t        ``[H, d_a]`` the next H commands after I_t (Eq. 3; regression target of Eq. 14)
    primitive   u_t        ``[]`` int64 primitive label of the next command (Eq. 10, Sec. 4.2, Fig. 3)
    episode, timestep, task ``[]`` int64 bookkeeping (latent-cache and c_text lookups in train.py)

LIBERO file layout (written by LIBERO's ``create_dataset.py``):
    data.attrs["problem_info"]          JSON string with "language_instruction"
    data/demo_<i>/actions               ``[T, 7]`` 6-D delta end-effector + gripper (+1 close, -1 open)
    data/demo_<i>/obs/agentview_rgb     ``[T, 128, 128, 3]`` uint8, rendered upside down
    data/demo_<i>/obs/eye_in_hand_rgb   ``[T, 128, 128, 3]`` uint8, rendered upside down
    data/demo_<i>/obs/ee_pos [T, 3], ee_ori [T, 3], ee_states [T, 6], gripper_states [T, 2],
                     joint_states [T, 7]

Frame / action alignment. ``create_dataset.py`` steps the simulator with ``actions[k]`` and then
records the resulting observation as frame k, so frame k shows the state *after* ``actions[k]``.
The command to issue from frame k is therefore ``actions[k + 1]``. A sample at frame t uses
A_t = (actions[t+1], ..., actions[t+H]) and the label of ``actions[t+1]`` (``ACTION_OFFSET = 1``).

Two inputs required by the paper are not part of LIBERO. The paper is silent on how they are
produced, so this module reads them from the file when present and otherwise uses an explicit,
documented stand-in:
    * primitive labels u_t -- ``data/demo_<i>/<primitive_key>`` (``[T]`` int), else
      ``gripper_phase_labels`` (reach / grasp / carry / release from the gripper command);
    * milestone frames I_g^(1..M) -- frame indices in ``data/demo_<i>/<milestone_key>``, else the
      final frame (M = 1, the single-stage case of Sec. 3.1), or ``gripper_milestone_indices`` on request.

Camera views. ``camera_keys`` names the cameras that form I_t. With one key (the default,
``agentview_rgb``) a frame is ``[C, H_img, W_img]``, exactly as before. With V >= 2 keys a frame is
the stack of the V views in the order of ``camera_keys``, ``[V, C, H_img, W_img]``, and so are
I_{t+nu}, the milestone frames and the frame blocks of the latent cache (``frame_shape``). Every view
goes through the same transform (rotation, scaling, resize).

Decisions where the paper is silent (do not change silently):
    * Chunks that run past the end of a demonstration repeat its final action
      (``chunk_padding="repeat"``); ``"drop"`` keeps only fully observed chunks.
    * Samples are restricted to t <= T - 1 - nu, so every sample has a real target frame I_{t+nu};
      since nu >= 1, the first command actions[t + 1] also always exists.
    * Frames are scaled to [0, 1] and not normalised further; LIBERO frames are rotated by 180 degrees
      (``rotate_180=True``), the convention used by common LIBERO pipelines. LIBERO renders the wrist
      camera upside down as well, so the rotation applies to every view.
    * With several cameras, all views must have the same stored frame shape, and I_t (like every
      milestone frame) is the tuple of the views in ``camera_keys`` order.

This file imports nothing from the VPA modules. ``python dataset.py`` runs the self-test
(needs ``h5py``; writes a small synthetic LIBERO-format file to a temporary directory).
"""

from __future__ import annotations

import copy
import json
import os
import random
import tempfile
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

__all__ = [
    "PRIMITIVE_NAMES",
    "DEFAULT_PROPRIO_KEYS",
    "DEFAULT_CAMERA_KEY",
    "Episode",
    "instruction_from_filename",
    "action_chunk",
    "gripper_phase_labels",
    "gripper_milestone_indices",
    "milestone_phase_by_index",
    "LiberoHDF5Dataset",
    "LiberoFrameDataset",
    "get_dataloader",
    "get_frame_loader",
]

# Labels produced by ``gripper_phase_labels`` (a stand-in; the paper does not define U).
REACH, GRASP, CARRY, RELEASE = 0, 1, 2, 3
PRIMITIVE_NAMES: Tuple[str, ...] = ("reach", "grasp", "carry", "release")

# Frame k of a LIBERO demonstration is recorded after actions[k] (see the module docstring).
ACTION_OFFSET = 1

# S_t = Concat(ee_pos, ee_ori, gripper_states, joint_states) -> d_s = 3 + 3 + 2 + 7 = 15 on LIBERO.
DEFAULT_PROPRIO_KEYS: Tuple[str, ...] = ("ee_pos", "ee_ori", "gripper_states", "joint_states")

# I_t of the single-camera setting (the third-person view).
DEFAULT_CAMERA_KEY = "agentview_rgb"

PathLike = Union[str, os.PathLike]
PrimitiveSource = Literal["auto", "hdf5", "gripper"]
MilestoneSource = Literal["auto", "hdf5", "final", "gripper"]


def _h5py() -> Any:
    """Import h5py lazily so the array helpers below work without it."""
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError("dataset.py needs h5py to read LIBERO files: pip install h5py") from exc
    return h5py


# ---------------------------------------------------------------------------------------------
# Array helpers (numpy only)
# ---------------------------------------------------------------------------------------------
def instruction_from_filename(path: PathLike) -> str:
    """Instruction x from a LIBERO file name.

    ``KITCHEN_SCENE3_turn_on_the_stove_demo.hdf5`` -> ``"turn on the stove"``: the extension and
    the ``_demo`` suffix are removed and every token up to the last one containing "SCENE" is dropped.
    """
    stem = os.path.basename(os.fspath(path))
    for suffix in (".hdf5", ".h5"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    if stem.endswith("_demo"):
        stem = stem[: -len("_demo")]
    words = [w for w in stem.split("_") if w]
    scene = [i for i, w in enumerate(words) if "SCENE" in w]
    if scene:
        words = words[scene[-1] + 1:]
    return " ".join(words)


def action_chunk(actions: np.ndarray, t: int, horizon: int) -> np.ndarray:
    """A_t = (a_t, a_{t+1}, ..., a_{t+H-1})  (Eq. 3).

    Steps past the last action a_{T-1} repeat a_{T-1} (see the module docstring).

    Args:
        actions: ``[T, d_a]`` actions of one demonstration.
        t: start step, 0 <= t < T.
        horizon: H.

    Returns:
        ``[H, d_a]`` (a new array).
    """
    length = actions.shape[0]
    if not 0 <= t < length:
        raise IndexError(f"t = {t} is outside [0, {length})")
    index = np.minimum(np.arange(t, t + horizon), length - 1)
    return actions[index]


def gripper_phase_labels(
    gripper_command: np.ndarray, window: int, close_positive: bool = True
) -> np.ndarray:
    """Primitive labels from the gripper command. A stand-in for annotated labels, not from the paper.

    closed_t = (command_t > 0), or (command_t < 0) if ``close_positive`` is False. Let s_t be the
    most recent step <= t at which ``closed`` changed value (s_t = 0 before the first change):

        reach   (0)  open,   and not (t - s_t < window after a close -> open switch)
        grasp   (1)  closed, t - s_t <  window
        carry   (2)  closed, t - s_t >= window
        release (3)  open,   t - s_t <  window, and the switch at s_t was close -> open

    Args:
        gripper_command: ``[T]`` gripper action (LIBERO: +1 close, -1 open).
        window: number of steps labelled grasp / release after a switch (>= 1).
        close_positive: sign convention of the command.

    Returns:
        ``[T]`` int64 labels in {0, 1, 2, 3}.
    """
    if window < 1:
        raise ValueError("window must be >= 1")
    cmd = np.asarray(gripper_command, dtype=np.float64).reshape(-1)
    closed = cmd > 0 if close_positive else cmd < 0
    labels = np.empty(cmd.shape[0], dtype=np.int64)
    switch, released = 0, False
    for t in range(cmd.shape[0]):
        if t > 0 and closed[t] != closed[t - 1]:
            switch, released = t, not bool(closed[t])
        recent = t - switch < window
        if closed[t]:
            labels[t] = GRASP if recent else CARRY
        else:
            labels[t] = RELEASE if (recent and released) else REACH
    return labels


def gripper_milestone_indices(labels: np.ndarray) -> Tuple[int, ...]:
    """Milestone frame indices from gripper-phase labels. A stand-in, not from the paper.

    Every closed -> open switch (an object has been put down) marks a milestone frame, and the
    final frame is always the last milestone.

    Args:
        labels: ``[T]`` output of ``gripper_phase_labels``.

    Returns:
        strictly increasing 0-based frame indices; the last one is T - 1.
    """
    labels = np.asarray(labels).reshape(-1)
    closed = (labels == GRASP) | (labels == CARRY)
    onsets = {t for t in range(1, labels.shape[0]) if closed[t - 1] and not closed[t]}
    return tuple(sorted(onsets | {labels.shape[0] - 1}))


def milestone_phase_by_index(milestone_indices: Sequence[int], length: int) -> np.ndarray:
    """1-based milestone pointer m_t from milestone frame indices, without the threshold test.

    m_t = 1 + #{k in 1..M-1 : idx_k < t}: frame idx_k still belongs to phase k, and phase k + 1
    starts on the next frame. This is the sequence Eq. 9a produces when ||z_t - z_g^(m_t)||_2 < tau
    holds exactly at the milestone frames and nowhere else; ``train.py`` uses it only when asked
    (``--milestone-phases index``) and otherwise applies Eq. 9a with tau*.

    Returns:
        ``[length]`` int64 in {1, ..., M}.
    """
    inner = np.asarray(list(milestone_indices)[:-1], dtype=np.int64)  # idx_1 .. idx_{M-1}
    t = np.arange(length, dtype=np.int64)
    return 1 + (inner[None, :] < t[:, None]).sum(axis=1).astype(np.int64)


def _sort_demo_keys(keys: Sequence[str]) -> List[str]:
    def key(name: str) -> Tuple[int, int, str]:
        tail = name.rsplit("_", 1)[-1]
        return (0, int(tail), name) if tail.isdigit() else (1, 0, name)

    return sorted(keys, key=key)


def _resolve_paths(paths: Union[PathLike, Sequence[PathLike]]) -> List[str]:
    """Files and directories (non-recursive ``*.hdf5`` / ``*.h5``) -> sorted list of files."""
    items = [paths] if isinstance(paths, (str, os.PathLike)) else list(paths)
    files: List[str] = []
    for item in items:
        p = os.fspath(item)
        if os.path.isdir(p):
            found = sorted(
                os.path.join(p, n) for n in os.listdir(p) if n.endswith((".hdf5", ".h5"))
            )
            if not found:
                raise FileNotFoundError(f"no .hdf5/.h5 files in {p}")
            files.extend(found)
        elif os.path.isfile(p):
            files.append(p)
        else:
            raise FileNotFoundError(p)
    if not files:
        raise FileNotFoundError("no dataset files given")
    return files


def _decode_attr(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.ndarray):
        return _decode_attr(value.item())
    return str(value)


# ---------------------------------------------------------------------------------------------
# Per-step dataset
# ---------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Episode:
    """One demonstration: ``path`` / ``data/<demo>``, T = ``length``, task index, milestone frames."""

    path: str
    demo: str
    length: int
    task: int
    milestones: Tuple[int, ...]   # 0-based frame indices of I_g^(1..M), strictly increasing


class LiberoHDF5Dataset(Dataset):
    """Per-control-step samples from LIBERO HDF5 demonstrations.

    Low-dimensional arrays (actions, proprioception, labels) are loaded into memory at
    construction; frames are read lazily from HDF5. File handles are opened per process, so the
    dataset works with any number of ``DataLoader`` workers and with the fork and spawn start methods.

    Args:
        paths: HDF5 files and/or directories containing them (non-recursive).
        horizon: chunk length H (= H_max).
        predictor_stride: nu >= 1; the JEPA target is I_{t+nu} (Sec. 3.3).
        image_size: (H_img, W_img) to resize to (bilinear, antialiased), or None for the native size.
        camera_keys: observation key(s) of I_t, e.g. ``agentview_rgb`` (default) or
            (``agentview_rgb``, ``eye_in_hand_rgb``); a string means one camera. The order is the
            view order.
        proprio_keys: observation keys concatenated into S_t.
        rotate_180: rotate frames by 180 degrees (LIBERO renders them upside down).
        primitive_source: "hdf5" (``primitive_key`` in every demo), "gripper" (``gripper_phase_labels``)
            or "auto" (hdf5 if every demo has the key, gripper if none has it, error if mixed).
        primitive_key: per-demo dataset of ``[T]`` integer labels.
        milestone_source: "hdf5" (``milestone_key`` in every demo), "final" (M = 1, last frame),
            "gripper" (``gripper_milestone_indices``) or "auto" (hdf5 if every demo has the key,
            final if none has it, error if mixed).
        milestone_key: per-demo dataset of increasing 0-based frame indices.
        gripper_window: ``window`` of ``gripper_phase_labels``.
        gripper_index: column of the gripper command in ``actions``.
        gripper_close_positive: sign convention of the gripper command.
        chunk_padding: "repeat" (pad past the episode end with the final action) or "drop".
        load_images: if False, samples carry no frames (used with a latent cache).
        episode_ids: global episode indices that contribute samples (default: all).
        camera_key: single-camera spelling of ``camera_keys`` (kept for existing callers); give one
            of the two, not both.

    Shapes of one sample (see the module docstring):
        image, next_image ``[C, H_img, W_img]`` (one camera) or ``[V, C, H_img, W_img]`` (V cameras);
        proprio ``[d_s]``; actions ``[H, d_a]``; primitive, episode, timestep, task ``[]`` int64.
    """

    def __init__(
        self,
        paths: Union[PathLike, Sequence[PathLike]],
        horizon: int,
        *,
        predictor_stride: int = 1,
        image_size: Optional[Tuple[int, int]] = None,
        camera_keys: Optional[Union[str, Sequence[str]]] = None,
        proprio_keys: Sequence[str] = DEFAULT_PROPRIO_KEYS,
        rotate_180: bool = True,
        primitive_source: PrimitiveSource = "auto",
        primitive_key: str = "primitive_labels",
        milestone_source: MilestoneSource = "auto",
        milestone_key: str = "milestone_indices",
        gripper_window: int = 10,
        gripper_index: int = -1,
        gripper_close_positive: bool = True,
        chunk_padding: Literal["repeat", "drop"] = "repeat",
        load_images: bool = True,
        episode_ids: Optional[Sequence[int]] = None,
        camera_key: Optional[str] = None,
    ) -> None:
        super().__init__()
        if horizon < 1:
            raise ValueError("horizon H must be >= 1")
        if predictor_stride < 1:
            raise ValueError("predictor stride nu must be >= 1")
        if primitive_source not in ("auto", "hdf5", "gripper"):
            raise ValueError(f"unknown primitive_source {primitive_source!r}")
        if milestone_source not in ("auto", "hdf5", "final", "gripper"):
            raise ValueError(f"unknown milestone_source {milestone_source!r}")
        if chunk_padding not in ("repeat", "drop"):
            raise ValueError(f"unknown chunk_padding {chunk_padding!r}")
        if not proprio_keys:
            raise ValueError("at least one proprioception key is required")
        if camera_keys is not None and camera_key is not None:
            raise ValueError("give camera_keys or camera_key, not both")
        keys = camera_keys if camera_keys is not None else (camera_key or DEFAULT_CAMERA_KEY)
        self.camera_keys: Tuple[str, ...] = (keys,) if isinstance(keys, str) else tuple(str(k) for k in keys)
        if not self.camera_keys:
            raise ValueError("at least one camera key is required")
        if len(set(self.camera_keys)) != len(self.camera_keys):
            raise ValueError(f"camera keys must be distinct, got {self.camera_keys}")
        self.horizon: int = int(horizon)
        self.predictor_stride: int = int(predictor_stride)
        self.image_size: Optional[Tuple[int, int]] = (
            None if image_size is None else (int(image_size[0]), int(image_size[1]))
        )
        self.proprio_keys: Tuple[str, ...] = tuple(proprio_keys)
        self.rotate_180: bool = bool(rotate_180)
        self.chunk_padding: str = chunk_padding
        self.load_images: bool = bool(load_images)
        self.gripper_window: int = int(gripper_window)
        self.files: List[str] = _resolve_paths(paths)
        self._handles: Dict[str, Any] = {}
        self._handle_pid: Optional[int] = None

        self.instructions: List[str] = []          # task index -> instruction x
        self.episodes: List[Episode] = []
        self._actions: List[np.ndarray] = []       # [T, d_a] float32
        self._proprio: List[np.ndarray] = []       # [T, d_s] float32
        self._labels: List[np.ndarray] = []        # [T] int64
        native_shape: Optional[Tuple[int, int, int]] = None
        raw_labels: List[Optional[np.ndarray]] = []
        raw_milestones: List[Optional[np.ndarray]] = []
        pending: List[Tuple[str, str, int, int]] = []  # (path, demo, T, task)

        h5py = _h5py()
        task_of: Dict[str, int] = {}
        for path in self.files:
            with h5py.File(path, "r") as f:
                if "data" not in f:
                    raise KeyError(f"{path}: missing group 'data'")
                data = f["data"]
                instruction = None
                if "problem_info" in data.attrs:
                    info = json.loads(_decode_attr(data.attrs["problem_info"]))
                    instruction = info.get("language_instruction")
                if not instruction:
                    instruction = instruction_from_filename(path)
                instruction = " ".join(str(instruction).split())
                task = task_of.setdefault(instruction, len(task_of))
                if task == len(self.instructions):
                    self.instructions.append(instruction)
                for demo in _sort_demo_keys([k for k in data.keys() if k.startswith("demo")]):
                    g = data[demo]
                    actions = np.asarray(g["actions"][()], dtype=np.float32)
                    if actions.ndim != 2:
                        raise ValueError(f"{path}/{demo}: actions must be [T, d_a], got {actions.shape}")
                    length = actions.shape[0]
                    obs = g["obs"]
                    for key in self.camera_keys:  # every view: present, [T, H, W, C], one frame shape
                        if key not in obs:
                            raise KeyError(f"{path}/{demo}: missing obs/{key}")
                        cam = obs[key]
                        if cam.ndim != 4 or cam.shape[0] != length:
                            raise ValueError(f"{path}/{demo}: obs/{key} must be [T, H, W, C]")
                        shape = (int(cam.shape[1]), int(cam.shape[2]), int(cam.shape[3]))
                        if native_shape is None:
                            native_shape = shape
                        elif shape != native_shape:
                            raise ValueError(f"{path}/{demo}: frame shape {shape} of obs/{key} differs "
                                             f"from {native_shape}")
                    parts = []
                    for key in self.proprio_keys:
                        if key not in obs:
                            raise KeyError(f"{path}/{demo}: missing obs/{key}")
                        arr = np.asarray(obs[key][()], dtype=np.float32).reshape(length, -1)
                        parts.append(arr)
                    self._actions.append(actions)
                    self._proprio.append(np.ascontiguousarray(np.concatenate(parts, axis=1)))
                    raw_labels.append(
                        np.asarray(g[primitive_key][()]).reshape(-1) if primitive_key in g else None
                    )
                    raw_milestones.append(
                        np.asarray(g[milestone_key][()]).reshape(-1) if milestone_key in g else None
                    )
                    pending.append((path, demo, length, task))
        if not pending:
            raise ValueError("the dataset files contain no demonstrations")
        assert native_shape is not None
        self.native_image_shape: Tuple[int, int, int] = native_shape  # (H, W, C) as stored

        dims = {a.shape[1] for a in self._actions}
        if len(dims) != 1:
            raise ValueError(f"inconsistent action dimensions {sorted(dims)}")
        dims = {s.shape[1] for s in self._proprio}
        if len(dims) != 1:
            raise ValueError(f"inconsistent proprioception dimensions {sorted(dims)}")

        self.primitive_source: str = self._resolve_source(primitive_source, raw_labels, "hdf5", "gripper",
                                                          primitive_key)
        self.milestone_source: str = self._resolve_source(milestone_source, raw_milestones, "hdf5", "final",
                                                          milestone_key)
        warned = False
        for i, (path, demo, length, task) in enumerate(pending):
            if self.primitive_source == "hdf5":
                labels = raw_labels[i]
                assert labels is not None
                if labels.shape[0] != length or not np.issubdtype(labels.dtype, np.integer) or labels.min() < 0:
                    raise ValueError(f"{path}/{demo}/{primitive_key}: expected [T] non-negative integers")
                labels = labels.astype(np.int64)
            else:
                if not warned and primitive_source == "auto":
                    warnings.warn(
                        f"no '{primitive_key}' in the demonstrations: using gripper_phase_labels "
                        f"{PRIMITIVE_NAMES} as primitive labels (a stand-in, not from the paper)",
                        stacklevel=2,
                    )
                    warned = True
                labels = gripper_phase_labels(
                    self._actions[i][:, gripper_index], self.gripper_window, gripper_close_positive
                )
            self._labels.append(labels)
            if self.milestone_source == "hdf5":
                ms = raw_milestones[i]
                assert ms is not None
                milestones = tuple(int(v) for v in ms)
            elif self.milestone_source == "gripper":
                milestones = gripper_milestone_indices(
                    gripper_phase_labels(self._actions[i][:, gripper_index], self.gripper_window,
                                         gripper_close_positive)
                )
            else:
                milestones = (length - 1,)
            if (len(milestones) < 1 or any(b <= a for a, b in zip(milestones[:-1], milestones[1:], strict=True))
                    or milestones[0] < 0 or milestones[-1] >= length):
                raise ValueError(f"{path}/{demo}: milestone indices must increase strictly within [0, T)")
            self.episodes.append(Episode(path, demo, length, task, milestones))

        lengths = np.array([e.length for e in self.episodes], dtype=np.int64)
        self.frame_offsets: np.ndarray = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
        self._num_primitives: int = (
            len(PRIMITIVE_NAMES) if self.primitive_source == "gripper"
            else int(max(int(lab.max()) for lab in self._labels)) + 1
        )
        self.episode_ids: np.ndarray = np.arange(len(self.episodes), dtype=np.int64)
        self._samples: np.ndarray = np.zeros((0, 2), dtype=np.int64)
        self._build_samples(range(len(self.episodes)) if episode_ids is None else episode_ids)

    @staticmethod
    def _resolve_source(
        requested: str, found: List[Optional[np.ndarray]], present: str, absent: str, key: str
    ) -> str:
        have = [x is not None for x in found]
        if requested == "auto":
            if all(have):
                return present
            if not any(have):
                return absent
            raise ValueError(f"'{key}' is present in some demonstrations but not in others")
        if requested == present and not all(have):
            raise KeyError(f"'{key}' is missing from {have.count(False)} demonstration(s)")
        return requested

    def _build_samples(self, episode_ids: Sequence[int]) -> None:
        ids = np.unique(np.asarray(list(episode_ids), dtype=np.int64))
        if ids.size and (ids[0] < 0 or ids[-1] >= len(self.episodes)):
            raise IndexError("episode id out of range")
        rows = []
        for e in ids.tolist():
            length = self.episodes[e].length
            last = length - 1 - self.predictor_stride            # I_{t+nu} must exist
            if self.chunk_padding == "drop":
                last = min(last, length - ACTION_OFFSET - self.horizon)  # actions[t+H] must exist
            if last >= 0:
                t = np.arange(last + 1, dtype=np.int64)
                rows.append(np.stack([np.full_like(t, e), t], axis=1))
        self.episode_ids = ids
        self._samples = np.concatenate(rows, axis=0) if rows else np.zeros((0, 2), dtype=np.int64)

    # ------------------------------------------------------------------ subsets
    def subset(self, episode_ids: Sequence[int]) -> "LiberoHDF5Dataset":
        """Shallow copy whose samples come only from ``episode_ids`` (global episode indices)."""
        other = copy.copy(self)
        other._handles, other._handle_pid = {}, None
        other._build_samples(episode_ids)
        return other

    def split_episodes(
        self, val_fraction: float, seed: int = 0
    ) -> Tuple["LiberoHDF5Dataset", "LiberoHDF5Dataset"]:
        """Episode-level train/validation split, stratified by task (no demonstration is shared).

        Each task contributes round(val_fraction * n_task) episodes to validation, keeping at
        least one training episode per task.
        """
        if not 0.0 <= val_fraction < 1.0:
            raise ValueError("val_fraction must lie in [0, 1)")
        rng = random.Random(seed)
        train, val = [], []
        by_task: Dict[int, List[int]] = {}
        for e in self.episode_ids.tolist():
            by_task.setdefault(self.episodes[e].task, []).append(e)
        for task in sorted(by_task):
            eps = by_task[task][:]
            rng.shuffle(eps)
            n_val = min(int(round(val_fraction * len(eps))), len(eps) - 1)
            val.extend(eps[:n_val])
            train.extend(eps[n_val:])
        return self.subset(train), self.subset(val)

    # ------------------------------------------------------------------ properties
    @property
    def proprio_dim(self) -> int:
        """d_s."""
        return int(self._proprio[0].shape[1])

    @property
    def action_dim(self) -> int:
        """d_a."""
        return int(self._actions[0].shape[1])

    @property
    def num_primitives(self) -> int:
        """Size of the label set: 4 for gripper labels, max label + 1 for labels read from the files."""
        return self._num_primitives

    @property
    def image_shape(self) -> Tuple[int, int, int]:
        """(C, H_img, W_img) of one camera view as returned by this dataset."""
        h, w, c = self.native_image_shape
        if self.image_size is not None:
            h, w = self.image_size
        return (c, h, w)

    @property
    def num_views(self) -> int:
        """V, the number of camera views per frame."""
        return len(self.camera_keys)

    @property
    def camera_key(self) -> str:
        """The camera of a single-camera dataset (kept for existing callers; use ``camera_keys``)."""
        if len(self.camera_keys) != 1:
            raise AttributeError(f"camera_key is defined for one camera only; this dataset has {self.camera_keys}")
        return self.camera_keys[0]

    @property
    def frame_shape(self) -> Tuple[int, ...]:
        """Shape of one frame I_t: ``(C, H_img, W_img)`` for one camera, ``(V, C, H_img, W_img)`` for V >= 2."""
        return self.image_shape if self.num_views == 1 else (self.num_views, *self.image_shape)

    @property
    def num_frames(self) -> int:
        """Total number of frames over all episodes (the length of the flat latent table)."""
        return int(self.frame_offsets[-1])

    def labels(self, episode: int) -> np.ndarray:
        """``[T]`` int64 primitive labels u_t of one episode."""
        return self._labels[episode]

    def proprioception(self, episode: int) -> np.ndarray:
        """``[T, d_s]`` float32 raw proprioception of one episode."""
        return self._proprio[episode]

    def actions(self, episode: int) -> np.ndarray:
        """``[T, d_a]`` float32 actions of one episode."""
        return self._actions[episode]

    # ------------------------------------------------------------------ statistics (Sec. 4.5)
    def proprio_statistics(self) -> Tuple[Tensor, Tensor]:
        """Dataset statistics (mu_S, sigma_S) over every frame of this dataset's episodes (Sec. 4.5).

        sigma_S is the unbiased per-dimension standard deviation (float64 accumulation).

        Returns:
            mu_S ``[d_s]``, sigma_S ``[d_s]`` (float32).
        """
        if self.episode_ids.size == 0:
            raise ValueError("no episodes")
        x = np.concatenate([self._proprio[e] for e in self.episode_ids.tolist()], axis=0).astype(np.float64)
        mean = x.mean(axis=0)
        std = x.std(axis=0, ddof=1) if x.shape[0] > 1 else np.zeros_like(mean)
        return torch.from_numpy(mean.astype(np.float32)), torch.from_numpy(std.astype(np.float32))

    # ------------------------------------------------------------------ frames
    def _file(self, path: str) -> Any:
        pid = os.getpid()
        if self._handle_pid != pid:          # new process (DataLoader worker): never reuse handles
            self._handles, self._handle_pid = {}, pid
        handle = self._handles.get(path)
        if handle is None:
            handle = _h5py().File(path, "r")
            self._handles[path] = handle
        return handle

    def _cameras(self, episode: int) -> List[Any]:
        """The HDF5 datasets of the views of one episode, in ``camera_keys`` order."""
        ep = self.episodes[episode]
        obs = self._file(ep.path)["data"][ep.demo]["obs"]
        return [obs[key] for key in self.camera_keys]

    def _views(self, episode: int, index: Union[int, slice]) -> Tensor:
        """Frames at ``index`` (one step or a slice of steps) of every view, through ``_to_tensor``.

        One camera: ``[C, H_img, W_img]`` or ``[n, C, H_img, W_img]``, exactly the stored frames.
        V >= 2 cameras: the views stacked in ``camera_keys`` order on a view axis,
        ``[V, C, H_img, W_img]`` or ``[n, V, C, H_img, W_img]``.
        """
        cams = self._cameras(episode)
        if len(cams) == 1:
            return self._to_tensor(cams[0][index])
        return torch.stack([self._to_tensor(cam[index]) for cam in cams], dim=-4)

    def _to_tensor(self, raw: np.ndarray) -> Tensor:
        """``[H, W, C]`` or ``[n, H, W, C]`` -> ``[C, H', W']`` or ``[n, C, H', W']`` float32."""
        x = np.asarray(raw)
        single = x.ndim == 3
        if single:
            x = x[None]
        scale = 255.0 if x.dtype == np.uint8 else 1.0
        if self.rotate_180:
            x = x[:, ::-1, ::-1]
        out = torch.from_numpy(np.ascontiguousarray(x)).permute(0, 3, 1, 2).to(torch.float32)
        if scale != 1.0:
            out = out / scale
        if self.image_size is not None and tuple(out.shape[-2:]) != self.image_size:
            out = F.interpolate(out, size=self.image_size, mode="bilinear", align_corners=False, antialias=True)
        return out[0] if single else out

    def frame(self, episode: int, t: int) -> Tensor:
        """Frame I_t of one episode: ``[*frame_shape]``, i.e. ``[C, H_img, W_img]`` or ``[V, C, H_img, W_img]``."""
        return self._views(episode, int(t))

    def episode_frames(self, episode: int, start: int = 0, stop: Optional[int] = None) -> Tensor:
        """Contiguous frames I_start .. I_{stop-1}: ``[n, *frame_shape]``."""
        stop = self.episodes[episode].length if stop is None else int(stop)
        return self._views(episode, slice(int(start), stop))

    def milestone_frames(self, episode: int) -> Tensor:
        """Milestone goal frames I_g^(1..M) of one episode, same views as I_t: ``[M, *frame_shape]``."""
        return torch.stack([self._views(episode, int(i)) for i in self.episodes[episode].milestones])

    def close(self) -> None:
        """Close the HDF5 handles opened by this process."""
        for handle in self._handles.values():
            try:
                handle.close()
            except Exception:  # closing is best effort
                pass
        self._handles = {}

    def __getstate__(self) -> Dict[str, Any]:
        state = self.__dict__.copy()
        state["_handles"], state["_handle_pid"] = {}, None   # h5py handles cannot be pickled
        return state

    # ------------------------------------------------------------------ Dataset protocol
    def __len__(self) -> int:
        return int(self._samples.shape[0])

    def sample_index(self, index: int) -> Tuple[int, int]:
        """(episode, t) of sample ``index``."""
        e, t = self._samples[index]
        return int(e), int(t)

    def __getitem__(self, index: int) -> Dict[str, Tensor]:
        e, t = self.sample_index(index)
        item: Dict[str, Tensor] = {
            "proprio": torch.from_numpy(self._proprio[e][t].copy()),                          # [d_s]
            # frame t shows the state after actions[t]; the commands to learn start at actions[t + 1]
            "actions": torch.from_numpy(action_chunk(self._actions[e], t + ACTION_OFFSET, self.horizon)),  # [H, d_a]
            "primitive": torch.tensor(int(self._labels[e][t + ACTION_OFFSET]), dtype=torch.long),         # []
            "episode": torch.tensor(e, dtype=torch.long),
            "timestep": torch.tensor(t, dtype=torch.long),
            "task": torch.tensor(self.episodes[e].task, dtype=torch.long),
        }
        if self.load_images:
            item["image"] = self.frame(e, t)                                                   # I_t
            item["next_image"] = self.frame(e, t + self.predictor_stride)                      # I_{t+nu}
        return item


# ---------------------------------------------------------------------------------------------
# Every frame, in flat order (latent cache, mu_Z of Sec. 4.5)
# ---------------------------------------------------------------------------------------------
class LiberoFrameDataset(Dataset):
    """Blocks of consecutive frames covering every frame of the given episodes exactly once.

    Item i is ``{"images": [n, *frame_shape], "flat_index": [n] int64}`` where ``flat_index`` is
    ``base.frame_offsets[episode] + t``. Use with ``get_frame_loader`` (each item is already a batch).

    Args:
        base: the per-step dataset that owns the files and preprocessing.
        episode_ids: global episode indices to cover (default: every episode of ``base.episodes``).
        block_size: maximum frames per item.
    """

    def __init__(
        self, base: LiberoHDF5Dataset, episode_ids: Optional[Sequence[int]] = None, block_size: int = 64
    ) -> None:
        super().__init__()
        if block_size < 1:
            raise ValueError("block_size must be >= 1")
        self.base = base.subset(base.episode_ids)    # own handle cache
        ids = range(len(base.episodes)) if episode_ids is None else episode_ids
        blocks = []
        for e in sorted({int(i) for i in ids}):
            length = base.episodes[e].length
            for start in range(0, length, block_size):
                blocks.append((e, start, min(start + block_size, length)))
        self.blocks: List[Tuple[int, int, int]] = blocks

    def __len__(self) -> int:
        return len(self.blocks)

    def __getitem__(self, index: int) -> Dict[str, Tensor]:
        e, start, stop = self.blocks[index]
        offset = int(self.base.frame_offsets[e])
        return {
            "images": self.base.episode_frames(e, start, stop),                       # [n, *frame_shape]
            "flat_index": torch.arange(offset + start, offset + stop, dtype=torch.long),  # [n]
        }


# ---------------------------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------------------------
def _seed_worker(worker_id: int) -> None:
    """Derive numpy / random seeds from the per-worker torch seed (reproducible augmentation hooks)."""
    seed = torch.initial_seed() % 2 ** 32
    np.random.seed(seed)
    random.seed(seed)


def get_dataloader(
    dataset: Dataset,
    batch_size: int,
    *,
    shuffle: bool = True,
    num_workers: int = 4,
    drop_last: bool = True,
    seed: int = 0,
    pin_memory: Optional[bool] = None,
    persistent_workers: Optional[bool] = None,
    prefetch_factor: int = 2,
    multiprocessing_context: Optional[str] = None,
) -> DataLoader:
    """DataLoader for ``LiberoHDF5Dataset`` (or a subset of it).

    The VICReg terms (Eqs. 18, 19) and the margin of Eq. 9b use the unbiased batch estimator, so a
    training batch needs B >= 2; ``drop_last=True`` (the default) keeps every batch at exactly
    ``batch_size``.

    Args:
        dataset: the dataset.
        batch_size: B >= 2.
        shuffle: reshuffle every epoch (seeded by ``seed``).
        num_workers: worker processes (0 loads in the main process).
        drop_last: drop the final incomplete batch.
        seed: seed of the shuffling generator; workers derive their seeds from it.
        pin_memory: default True when CUDA is available.
        persistent_workers: default True when ``num_workers > 0``.
        prefetch_factor: batches prefetched per worker (``num_workers > 0`` only).
        multiprocessing_context: e.g. "spawn" or "fork" (default: the platform default).

    Returns:
        ``DataLoader`` yielding dicts of batched tensors (``image [B, *frame_shape]``,
        ``actions [B, H, d_a]``, ``proprio [B, d_s]``, ...).
    """
    if batch_size < 2:
        raise ValueError("batch_size must be >= 2 (unbiased batch variance in Eqs. 9b, 18, 19)")
    if num_workers < 0:
        raise ValueError("num_workers must be >= 0")
    n = len(dataset)  # type: ignore[arg-type]
    if n == 0:
        raise ValueError("the dataset is empty")
    if drop_last and n < batch_size:
        raise ValueError(f"the dataset has {n} samples, fewer than one batch of {batch_size}")
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    kwargs: Dict[str, Any] = dict(
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
        pin_memory=torch.cuda.is_available() if pin_memory is None else bool(pin_memory),
        generator=generator,
        worker_init_fn=_seed_worker,
    )
    if num_workers > 0:
        kwargs["persistent_workers"] = True if persistent_workers is None else bool(persistent_workers)
        kwargs["prefetch_factor"] = int(prefetch_factor)
        if multiprocessing_context is not None:
            kwargs["multiprocessing_context"] = multiprocessing_context
    return DataLoader(dataset, **kwargs)


def get_frame_loader(
    frames: LiberoFrameDataset,
    *,
    num_workers: int = 4,
    pin_memory: Optional[bool] = None,
    multiprocessing_context: Optional[str] = None,
) -> DataLoader:
    """Ordered, non-shuffled loader over ``LiberoFrameDataset`` blocks (one block per iteration)."""
    kwargs: Dict[str, Any] = dict(
        batch_size=None,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available() if pin_memory is None else bool(pin_memory),
        worker_init_fn=_seed_worker,
    )
    if num_workers > 0 and multiprocessing_context is not None:
        kwargs["multiprocessing_context"] = multiprocessing_context
    return DataLoader(frames, **kwargs)


# ---------------------------------------------------------------------------------------------
# Synthetic LIBERO-format files (self-tests of dataset.py and train.py)
# ---------------------------------------------------------------------------------------------
def _synthetic_gripper(length: int) -> np.ndarray:
    """-1 (open) / +1 (closed) pattern with one grasp, or two for episodes of 12 steps or more."""
    cmd = -np.ones(length, dtype=np.float64)
    cmd[length // 6: length // 3 + 1] = 1.0
    if length >= 12:
        cmd[length // 2: (5 * length) // 6] = 1.0
    return cmd


def _write_synthetic_libero(
    path: str,
    lengths: Sequence[int],
    seed: int,
    image_hw: Tuple[int, int] = (16, 16),
    instruction: Optional[str] = "pick up the red cup and place it on the plate",
    primitive_labels: Optional[Sequence[np.ndarray]] = None,
    milestones: Optional[Sequence[Sequence[int]]] = None,
) -> None:
    """Write a small file with the LIBERO layout (random frames and low-dimensional data)."""
    h5py = _h5py()
    rng = np.random.default_rng(seed)
    with h5py.File(path, "w") as f:
        data = f.create_group("data")
        if instruction is not None:
            data.attrs["problem_info"] = json.dumps({"language_instruction": instruction, "problem_name": "x"})
        for i, length in enumerate(lengths):
            g = data.create_group(f"demo_{i}")
            actions = rng.uniform(-1.0, 1.0, size=(length, 7))
            actions[:, -1] = _synthetic_gripper(length)
            g.create_dataset("actions", data=actions)
            obs = g.create_group("obs")
            obs.create_dataset("agentview_rgb",
                               data=rng.integers(0, 256, size=(length, *image_hw, 3), dtype=np.uint8))
            obs.create_dataset("eye_in_hand_rgb",
                               data=rng.integers(0, 256, size=(length, *image_hw, 3), dtype=np.uint8))
            obs.create_dataset("ee_pos", data=rng.normal(size=(length, 3)))
            obs.create_dataset("ee_ori", data=rng.normal(size=(length, 3)))
            obs.create_dataset("gripper_states", data=rng.normal(size=(length, 2)))
            obs.create_dataset("joint_states", data=rng.normal(size=(length, 7)))
            if primitive_labels is not None:
                g.create_dataset("primitive_labels", data=np.asarray(primitive_labels[i], dtype=np.int64))
            if milestones is not None:
                g.create_dataset("milestone_indices", data=np.asarray(milestones[i], dtype=np.int64))


# ---------------------------------------------------------------------------------------------
# Self-test (run: python dataset.py)
# ---------------------------------------------------------------------------------------------
def _expect(exc: type, fn: Any) -> None:
    try:
        fn()
    except exc:
        return
    raise AssertionError(f"expected {exc.__name__}")


def _ref_phase_labels(cmd: np.ndarray, window: int) -> List[int]:
    """Independent reference for ``gripper_phase_labels``: switch times found first, then labelled."""
    closed = [c > 0 for c in cmd.tolist()]
    switches = [0] + [t for t in range(1, len(closed)) if closed[t] != closed[t - 1]]
    out = []
    for t in range(len(closed)):
        s = max(x for x in switches if x <= t)
        released = s > 0 and not closed[s]
        if closed[t]:
            out.append(1 if t - s < window else 2)
        else:
            out.append(3 if (t - s < window and released) else 0)
    return out


def _self_test() -> None:
    torch.manual_seed(0)
    # ---- array helpers against independent references
    assert instruction_from_filename("/x/KITCHEN_SCENE3_turn_on_the_stove_demo.hdf5") == "turn on the stove"
    assert instruction_from_filename("LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket"
                                     "_demo.hdf5").startswith("put both the cream")
    assert instruction_from_filename("pick_up_the_bowl_demo.hdf5") == "pick up the bowl"
    acts = np.arange(5 * 2, dtype=np.float32).reshape(5, 2)
    for t in range(5):
        chunk = action_chunk(acts, t, 4)
        ref = [acts[min(t + h, 4)].tolist() for h in range(4)]
        assert chunk.tolist() == ref and chunk.shape == (4, 2)
    _expect(IndexError, lambda: action_chunk(acts, 5, 4))
    rng = np.random.default_rng(3)
    for _ in range(50):
        cmd = np.where(rng.random(40) < 0.15, 1.0, -1.0)
        cmd = np.repeat(cmd, rng.integers(1, 4, size=40))
        w = int(rng.integers(1, 6))
        assert gripper_phase_labels(cmd, w).tolist() == _ref_phase_labels(cmd, w)
        assert gripper_phase_labels(-cmd, w, close_positive=False).tolist() == _ref_phase_labels(cmd, w)
    labels = gripper_phase_labels(_synthetic_gripper(18), 2)
    closed = [lab in (GRASP, CARRY) for lab in labels.tolist()]
    ref_ms = sorted({t for t in range(1, 18) if closed[t - 1] and not closed[t]} | {17})
    assert list(gripper_milestone_indices(labels)) == ref_ms and len(ref_ms) == 3
    for idx, length in (((4, 9, 13), 14), ((13,), 14), ((0, 5), 6)):
        phase = milestone_phase_by_index(idx, length).tolist()
        ref = [1 + sum(1 for k in idx[:-1] if k < t) for t in range(length)]
        assert phase == ref and min(phase) == 1 and max(phase) <= len(idx)

    with tempfile.TemporaryDirectory() as tmp:
        lengths_a, lengths_b = (12, 9, 15), (13, 8)
        file_a = os.path.join(tmp, "KITCHEN_SCENE1_open_the_drawer_demo.hdf5")
        file_b = os.path.join(tmp, "LIVING_ROOM_SCENE2_stack_the_blocks_demo.hdf5")
        _write_synthetic_libero(file_a, lengths_a, seed=1)
        _write_synthetic_libero(file_b, lengths_b, seed=2, instruction=None)
        h5py = _h5py()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            ds = LiberoHDF5Dataset(tmp, horizon=4, predictor_stride=2)
        assert any("gripper_phase_labels" in str(w.message) for w in caught)
        lengths = lengths_a + lengths_b
        assert ds.instructions == ["pick up the red cup and place it on the plate", "stack the blocks"]
        assert [e.task for e in ds.episodes] == [0, 0, 0, 1, 1]
        assert len(ds) == sum(t - 2 for t in lengths) and ds.num_frames == sum(lengths)
        assert ds.proprio_dim == 15 and ds.action_dim == 7 and ds.num_primitives == 4
        assert ds.image_shape == (3, 16, 16) and ds.primitive_source == "gripper"
        assert ds.milestone_source == "final" and [e.milestones for e in ds.episodes] == [(t - 1,) for t in lengths]

        # every sample against the raw file (independent read + manual layout conversion)
        with h5py.File(file_a, "r") as fa, h5py.File(file_b, "r") as fb:
            raw = [(fa, f"demo_{i}") for i in range(3)] + [(fb, f"demo_{i}") for i in range(2)]
            for index in range(len(ds)):
                e, t = ds.sample_index(index)
                f, demo = raw[e]
                g = f["data"][demo]
                item = ds[index]
                img = g["obs"]["agentview_rgb"][t]
                ref_img = np.stack([[[img[15 - r, 15 - c, ch] / 255.0 for c in range(16)] for r in range(16)]
                                    for ch in range(3)])
                assert np.allclose(item["image"].numpy(), ref_img, atol=1e-7)
                nxt = g["obs"]["agentview_rgb"][t + 2][::-1, ::-1].transpose(2, 0, 1) / 255.0
                assert np.allclose(item["next_image"].numpy(), nxt, atol=1e-7)
                ref_s = np.concatenate([g["obs"][k][t] for k in DEFAULT_PROPRIO_KEYS]).astype(np.float32)
                assert np.array_equal(item["proprio"].numpy(), ref_s)
                a = g["actions"][()]
                ref_a = np.stack([a[min(t + 1 + h, a.shape[0] - 1)] for h in range(4)]).astype(np.float32)
                assert np.array_equal(item["actions"].numpy(), ref_a) and item["actions"].shape == (4, 7)
                assert int(item["primitive"]) == _ref_phase_labels(a[:, -1], 10)[t + 1]
                assert int(item["task"]) == ds.episodes[e].task and int(item["episode"]) == e
                assert item["image"].dtype == torch.float32 and item["primitive"].dtype == torch.long

            # proprio statistics vs a float64 reference
            mu, sd = ds.proprio_statistics()
            allx = np.concatenate([np.concatenate([g["data"][d]["obs"][k][()] for k in DEFAULT_PROPRIO_KEYS], 1)
                                   for g, d in raw])
            assert np.allclose(mu.numpy(), allx.mean(0), rtol=1e-5, atol=1e-6)
            assert np.allclose(sd.numpy(), allx.std(0, ddof=1), rtol=1e-5, atol=1e-6)

        # drop padding, resize, no images, gripper milestones
        ds_drop = LiberoHDF5Dataset(tmp, horizon=4, predictor_stride=2, chunk_padding="drop",
                                    milestone_source="gripper", primitive_source="gripper")
        assert len(ds_drop) == sum(min(t - 3, t - 5) + 1 for t in lengths)   # t <= T-1-nu and t+H <= T-1
        assert all(e.milestones[-1] == e.length - 1 for e in ds_drop.episodes)
        ds_small = LiberoHDF5Dataset(tmp, horizon=4, image_size=(8, 8), load_images=True,
                                     primitive_source="gripper")
        assert ds_small[0]["image"].shape == (3, 8, 8) and ds_small.image_shape == (3, 8, 8)
        assert ds_small.milestone_frames(0).shape == (1, 3, 8, 8)
        assert ds_small.episode_frames(1, 2, 7).shape == (5, 3, 8, 8)
        ds_lat = LiberoHDF5Dataset(tmp, horizon=4, load_images=False, primitive_source="gripper")
        assert "image" not in ds_lat[0] and len(ds_lat) == sum(t - 1 for t in lengths)

        # several cameras: views stacked in camera_keys order, each view bit-identical to the
        # one-camera dataset of that key; everything that is not a frame is unchanged
        keys2 = ("agentview_rgb", "eye_in_hand_rgb")
        ds2 = LiberoHDF5Dataset(tmp, horizon=4, predictor_stride=2, camera_keys=keys2, primitive_source="gripper")
        ds_eye = LiberoHDF5Dataset(tmp, horizon=4, predictor_stride=2, camera_keys="eye_in_hand_rgb",
                                   primitive_source="gripper")
        assert ds.camera_keys == ("agentview_rgb",) and ds.num_views == 1 and ds.frame_shape == (3, 16, 16)
        assert ds2.camera_keys == keys2 and ds2.num_views == 2 and ds2.frame_shape == (2, 3, 16, 16)
        assert ds2.image_shape == (3, 16, 16) and ds_eye.frame_shape == (3, 16, 16) and len(ds2) == len(ds)
        for index in (0, 7, len(ds) - 1):
            one, two, eye = ds[index], ds2[index], ds_eye[index]
            assert two["image"].shape == (2, 3, 16, 16) and two["next_image"].shape == (2, 3, 16, 16)
            assert torch.equal(two["image"][0], one["image"]) and torch.equal(two["image"][1], eye["image"])
            assert torch.equal(two["next_image"][0], one["next_image"])
            assert torch.equal(two["next_image"][1], eye["next_image"])
            for key in ("proprio", "actions", "primitive", "episode", "timestep", "task"):
                assert torch.equal(two[key], one[key]), key
        with h5py.File(file_a, "r") as fa:      # the wrist view against the raw file (demo_1 = episode 1)
            raw_eye = fa["data"]["demo_1"]["obs"]["eye_in_hand_rgb"][4]
        ref_eye = np.stack([[[raw_eye[15 - r, 15 - c, ch] / 255.0 for c in range(16)] for r in range(16)]
                            for ch in range(3)])
        assert np.allclose(ds2.frame(1, 4)[1].numpy(), ref_eye, atol=1e-7)
        block2 = ds2.episode_frames(1, 2, 7)
        assert block2.shape == (5, 2, 3, 16, 16)
        assert torch.equal(block2[:, 0], ds.episode_frames(1, 2, 7))
        assert torch.equal(block2[:, 1], ds_eye.episode_frames(1, 2, 7))
        assert ds2.milestone_frames(0).shape == (1, 2, 3, 16, 16)
        assert torch.equal(ds2.milestone_frames(0)[:, 0], ds.milestone_frames(0))
        assert torch.equal(ds2.milestone_frames(0)[:, 1], ds_eye.milestone_frames(0))
        assert LiberoFrameDataset(ds2, block_size=5)[0]["images"].shape == (5, 2, 3, 16, 16)
        ds2_small = LiberoHDF5Dataset(tmp, horizon=4, camera_keys=keys2, image_size=(8, 8), primitive_source="gripper")
        assert ds2_small[0]["image"].shape == (2, 3, 8, 8) and ds2_small.frame_shape == (2, 3, 8, 8)
        ds2_rev = LiberoHDF5Dataset(tmp, horizon=4, camera_keys=keys2[::-1], primitive_source="gripper")
        assert torch.equal(ds2_rev.frame(0, 3), ds2.frame(0, 3).flip(0))     # camera_keys order = view order
        assert LiberoHDF5Dataset(tmp, horizon=4, camera_key="eye_in_hand_rgb",
                                 primitive_source="gripper").camera_keys == ("eye_in_hand_rgb",)  # old keyword
        _expect(KeyError, lambda: LiberoHDF5Dataset(tmp, horizon=4, camera_keys=("agentview_rgb", "nope")))
        _expect(ValueError, lambda: LiberoHDF5Dataset(tmp, horizon=4, camera_keys=("agentview_rgb",) * 2))
        _expect(ValueError, lambda: LiberoHDF5Dataset(tmp, horizon=4, camera_keys=()))
        _expect(ValueError, lambda: LiberoHDF5Dataset(tmp, horizon=4, camera_keys="agentview_rgb",
                                                      camera_key="agentview_rgb"))
        batch2 = next(iter(get_dataloader(ds2, batch_size=4, num_workers=0, seed=7, pin_memory=False)))
        assert batch2["image"].shape == (4, 2, 3, 16, 16) and batch2["next_image"].shape == (4, 2, 3, 16, 16)
        for d in (ds2, ds_eye, ds2_small, ds2_rev):
            d.close()

        # episode split: disjoint, stratified, covering
        train, val = ds.split_episodes(0.4, seed=0)
        tr, va = set(train.episode_ids.tolist()), set(val.episode_ids.tolist())
        assert not tr & va and tr | va == set(range(5))
        assert {ds.episodes[e].task for e in tr} == {0, 1}
        assert {train.sample_index(i)[0] for i in range(len(train))} == tr
        assert {val.sample_index(i)[0] for i in range(len(val))} == va

        # frame blocks cover every frame once, in flat order, and match frame()
        frames = LiberoFrameDataset(ds, block_size=5)
        seen = torch.cat([frames[i]["flat_index"] for i in range(len(frames))])
        assert seen.tolist() == list(range(ds.num_frames))
        blk = frames[2]
        e0 = int(np.searchsorted(ds.frame_offsets, int(blk["flat_index"][0]), side="right") - 1)
        t0 = int(blk["flat_index"][0]) - int(ds.frame_offsets[e0])
        assert torch.equal(blk["images"][0], ds.frame(e0, t0))

        # labels and milestones read from the files
        file_c = os.path.join(tmp, "annotated", "KITCHEN_SCENE9_annotated_demo.hdf5")
        os.makedirs(os.path.dirname(file_c))
        lab_c = [np.array([0, 0, 1, 1, 2, 2, 5, 5, 5, 5]), np.array([3, 3, 3, 4, 4, 4, 4, 4])]
        _write_synthetic_libero(file_c, (10, 8), seed=4, primitive_labels=lab_c, milestones=[(3, 9), (7,)])
        ds_c = LiberoHDF5Dataset(file_c, horizon=3)
        assert ds_c.primitive_source == "hdf5" and ds_c.num_primitives == 6
        assert ds_c.milestone_source == "hdf5" and [e.milestones for e in ds_c.episodes] == [(3, 9), (7,)]
        assert [int(ds_c[i]["primitive"]) for i in range(len(ds_c))] == lab_c[0][1:10].tolist() + lab_c[1][1:8].tolist()
        _expect(KeyError, lambda: LiberoHDF5Dataset(file_a, horizon=3, primitive_source="hdf5"))
        _expect(ValueError, lambda: LiberoHDF5Dataset([file_a, file_c], horizon=3))  # mixed annotation

        # loaders: B >= 2, drop_last, workers (fork or spawn), pickling with open handles
        _expect(ValueError, lambda: get_dataloader(ds, batch_size=1))
        _expect(ValueError, lambda: get_dataloader(ds_c.subset([1]), batch_size=64))
        import pickle
        ds.frame(0, 0)                              # opens a handle in this process
        clone = pickle.loads(pickle.dumps(ds))
        assert torch.equal(clone[3]["image"], ds[3]["image"])
        for workers in (0, 2):
            loader = get_dataloader(ds, batch_size=4, num_workers=workers, seed=7, pin_memory=False)
            batches = list(loader)
            assert len(batches) == len(ds) // 4
            b = batches[0]
            assert b["image"].shape == (4, 3, 16, 16) and b["next_image"].shape == (4, 3, 16, 16)
            assert b["actions"].shape == (4, 4, 7) and b["proprio"].shape == (4, 15)
            assert b["primitive"].shape == (4,) and b["primitive"].dtype == torch.long
            idx = sorted(int(b["episode"][i]) * 1000 + int(b["timestep"][i]) for i in range(4))
            assert len(set(idx)) == 4
        fl = list(get_frame_loader(frames, num_workers=2, pin_memory=False))
        assert torch.cat([x["flat_index"] for x in fl]).tolist() == list(range(ds.num_frames))
        for d in (ds, ds_drop, ds_small, ds_lat, ds_c, clone):
            d.close()
    print("dataset.py self-test passed")


if __name__ == "__main__":
    _self_test()
