# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
"""
train.py -- Offline training of the VPA modules (preprint_261008.pdf, Secs. 4.2-4.5, 5.2, Fig. 3).

The paper trains in stages, and the order matters: gamma* exists only once E_psi is frozen, and
the solver's conditioning (Eq. 12, Sec. 4.5) is standardised with gamma*. This script runs the
stages in sequence (``--stage all``) or one at a time.

Stage 1, "jepa" -- representation learning (Sec. 4.3, Fig. 3). E_psi and P_omega are trained
jointly; the text encoder is not involved. Per iteration n, on a batch of demonstration steps:

    z_t      = E_psi(I_t)                    online latents, the batch Z^(n)               (Eq. 7)
    z_{t+1}  = sg(E_psi_bar(I_{t+nu}))       momentum (EMA) target, no gradient            (Eq. 11)
    zhat^(i) = P_omega^(i)(z_t, u_t)         u_t = demonstration primitive label           (Eq. 10)
    L_JEPA   = E||zhat^(i) - z_{t+1}||_2^2 (each head) + lambda_v v(Z) + lambda_c c(Z)     (Eqs. 11, 18, 19)
    optimiser step on (psi, omega); then psi_bar <- m psi_bar + (1 - m) psi (EMA);
    then gamma_n = (1 - a_g) gamma_{n-1} + a_g sqrt((1/d) sum_j Var(Z^(n)_.j) + eps)       (Eq. 9b)

At the end E_psi and P_omega are frozen: gamma* = gamma_N, tau* = kappa sqrt(2d) gamma* (Eq. 9c).

Stage 2, "policy" -- alignment and warm start (Secs. 4.2, 4.5):

    1. every frame is encoded once by the frozen E_psi (latent cache); mu_Z is the mean latent
       over the training data and (mu_S, sigma_S) are the proprioception statistics;
    2. ``VPAInferencePipeline.calibrate(gamma*, mu_Z, mu_S, sigma_S)`` installs gamma* in the
       tracker, the predictor and the solver together;
    3. the milestone pointer m_t along every demonstration is obtained by Eq. 9a with tau*;
    4. per iteration, with E_psi and P_omega frozen:
       zhat_{t+1} = P_omega(z_t, u_t)                                                     (Eq. 10)
       e_t        = Concat(Embed(u_t), z~_t, zhat~_{t+1}, S~_t, c_text)                    (Eq. 12)
       L_solver   = E||v_theta(A^(rho), rho | e_t) - (A_t - xi)||_2^2,  rho ~ U[0, 1]      (Eqs. 13, 14)
       L_select   = CE(pi_phi^h(. | z_t, z_g^(m_t), c_text), u_t)                          (Sec. 4.2)

Decisions where the paper is silent (do not change silently):
    * pi_phi^h is trained in stage 2, on frozen latents: it needs c_text, and Fig. 3 states that
      the text encoder is not involved in stage 1.
    * During the warm start, Embed(u_t) and P_omega use the demonstration label u_t (teacher forcing).
    * Eq. 9a segments a demonstration with every frame treated as a decision step
      (``--milestone-phases index`` uses the milestone frame indices instead).
    * The EMA momentum is constant (default 0.996, the ``MomentumEncoder`` placeholder).
    * lambda_v = 25, lambda_c = 1, gamma = 1, AdamW, warm-up + cosine schedule and all step counts
      are placeholders.
    * sigma_S entries below 1e-6 (constant proprioception channels) are replaced by 1.
    * L_solver + w * L_select is minimised with one optimiser; the two losses share no parameters,
      and gradient clipping is applied per module, so they are optimised independently.
    * E_psi starts from random weights unless ``--init-encoder`` is given; then its first blocks are
      initialised from DINOv2 (``pretrained_encoder.py``) and stage 1 trains it with Eq. 11 as usual.
    * I_t is one camera (``--camera-keys agentview_rgb``, the default) or the tuple of several views
      (e.g. ``--camera-keys agentview_rgb eye_in_hand_rgb``). VPAConfig.num_views follows the data;
      every frame E_psi sees -- I_t, I_{t+nu}, milestones, the latent cache -- carries the same views,
      and the camera keys (in order) are stored in the checkpoint. Nothing else changes.

Usage:
    python train.py --data /path/to/libero_10 --out runs/libero10
    python train.py --data ... --out ... --stage policy --init-from runs/libero10/jepa_final.pt
    python train.py --data ... --out ... --resume runs/libero10/jepa_step0010000.pt
    python train.py --data ... --out runs/libero10_dino --init-encoder dinov2-small --encoder-lr 1e-4
    python train.py --data ... --out runs/libero10_2cam --camera-keys agentview_rgb eye_in_hand_rgb
    python train.py --self-test        # offline, CPU, synthetic LIBERO-format data (needs h5py)

The final checkpoint loads straight into the inference pipeline with ``load_pipeline(path)``.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import glob
import json
import logging
import math
import os
import random
import sys
import tempfile
import time
from collections import defaultdict
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

try:  # package import (e.g. `from vpa.train import ...`)
    from .dataset import (
        DEFAULT_CAMERA_KEY,
        DEFAULT_PROPRIO_KEYS,
        LiberoFrameDataset,
        LiberoHDF5Dataset,
        _write_synthetic_libero,
        get_dataloader,
        get_frame_loader,
        milestone_phase_by_index,
    )
    from .perception import MomentumEncoder, TextEncoderWrapper, _toy_tokenizer, _ToyTextModel
    from .pipeline import VPAConfig, VPAInferencePipeline
    from .predictor import VICRegLoss
    from .selector import NeuroSymbolicSelector
except ImportError:  # flat import (files side by side, `python train.py`)
    from dataset import (
        DEFAULT_CAMERA_KEY,
        DEFAULT_PROPRIO_KEYS,
        LiberoFrameDataset,
        LiberoHDF5Dataset,
        _write_synthetic_libero,
        get_dataloader,
        get_frame_loader,
        milestone_phase_by_index,
    )
    from perception import MomentumEncoder, TextEncoderWrapper, _toy_tokenizer, _ToyTextModel
    from pipeline import VPAConfig, VPAInferencePipeline
    from predictor import VICRegLoss
    from selector import NeuroSymbolicSelector

__all__ = [
    "JEPATrainer",
    "camera_keys_of",
    "PolicyData",
    "PolicyTrainer",
    "build_arg_parser",
    "encode_frames",
    "finish_jepa",
    "load_pipeline",
    "milestone_goal_index",
    "prepare_policy",
    "run",
]

CHECKPOINT_FORMAT = "vpa-train-v1"
LOGGER = logging.getLogger("vpa.train")


# ---------------------------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------------------------
def resolve_device(name: str) -> torch.device:
    """"auto" -> cuda, then mps, then cpu."""
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % 2 ** 32)
    torch.manual_seed(seed)


def _close_logging() -> None:
    for handler in list(LOGGER.handlers):
        handler.close()
        LOGGER.removeHandler(handler)


def setup_logging(out_dir: str) -> logging.Logger:
    """Log to stdout and to ``<out_dir>/train.log`` (handlers are replaced on every call)."""
    _close_logging()
    LOGGER.setLevel(logging.INFO)
    LOGGER.propagate = False
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%Y-%m-%d %H:%M:%S")
    for handler in (logging.StreamHandler(sys.stdout), logging.FileHandler(os.path.join(out_dir, "train.log"))):
        handler.setFormatter(fmt)
        LOGGER.addHandler(handler)
    return LOGGER


class MetricsWriter:
    """Append one JSON object per line to ``<out_dir>/metrics.jsonl``."""

    def __init__(self, out_dir: str) -> None:
        self.path = os.path.join(out_dir, "metrics.jsonl")

    def write(self, record: Dict[str, Any]) -> None:
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")


def _json_safe(obj: Any) -> Any:
    return json.loads(json.dumps(obj, default=str))


def count_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def _no_weight_decay(name: str, param: Tensor) -> bool:
    """Biases, norms, positional/class tokens and embeddings are not weight-decayed."""
    return (
        param.ndim <= 1
        or name.endswith("bias")
        or any(key in name for key in ("pos_embed", "cls_token", "embedding"))
    )


def make_optimizer(
    named_params: Sequence[Tuple[str, nn.Parameter]],
    lr: float,
    weight_decay: float,
    betas: Tuple[float, float],
    device: torch.device,
    lr_override: Optional[Callable[[str], Optional[float]]] = None,
) -> torch.optim.AdamW:
    """AdamW with weight decay on matrices only.

    ``lr_override(name)`` may return a different learning rate for a parameter (None: ``lr``). Groups are
    ordered by first appearance of each learning rate, decay before no-decay; without an override this is
    exactly the two groups [decay, no-decay] of earlier versions, so old optimiser states still load.
    """
    def lr_of(name: str) -> float:
        value = None if lr_override is None else lr_override(name)
        return lr if value is None else float(value)

    rates: List[float] = []
    for n, p in named_params:
        if p.requires_grad and lr_of(n) not in rates:
            rates.append(lr_of(n))
    groups = []
    for rate in rates:
        decay = [p for n, p in named_params if p.requires_grad and lr_of(n) == rate and not _no_weight_decay(n, p)]
        no_decay = [p for n, p in named_params if p.requires_grad and lr_of(n) == rate and _no_weight_decay(n, p)]
        groups += [{"params": ps, "weight_decay": wd, "lr": rate}
                   for ps, wd in ((decay, weight_decay), (no_decay, 0.0)) if ps]
    return torch.optim.AdamW(groups, lr=lr, betas=betas, fused=device.type == "cuda")


def make_scheduler(
    optimizer: torch.optim.Optimizer, warmup_steps: int, total_steps: int, min_lr_ratio: float
) -> torch.optim.lr_scheduler.LambdaLR:
    """Linear warm-up, then cosine decay to ``min_lr_ratio`` of the base rate."""

    def factor(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(1.0, max(0.0, progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def _clip(params: Sequence[Tensor], max_norm: float) -> float:
    """Clip the global gradient norm (no clipping if ``max_norm <= 0``); returns the norm before clipping."""
    norm = torch.nn.utils.clip_grad_norm_(params, max_norm if max_norm > 0 else float("inf"))
    return float(norm)


def _autocast(device: torch.device, amp: str) -> contextlib.AbstractContextManager:
    if amp == "none":
        return contextlib.nullcontext()
    if amp == "bf16" and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    raise ValueError("--amp bf16 is only supported on CUDA (MPS and CPU train in float32)")


def _cycle(loader: Any) -> Iterator[Tuple[int, Dict[str, Tensor]]]:
    epoch = 0
    while True:
        for batch in loader:
            yield epoch, batch
        epoch += 1


# ---------------------------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------------------------
def build_text_encoder(kind: str, model_name: str, toy_projection_dim: int = 12) -> TextEncoderWrapper:
    """Frozen text encoder producing c_text: pre-trained CLIP, or the offline stand-in for tests."""
    if kind == "clip":
        return TextEncoderWrapper(model_name)
    if kind == "toy":
        return TextEncoderWrapper(text_model=_ToyTextModel(projection_dim=toy_projection_dim), tokenizer=_toy_tokenizer)
    raise ValueError(f"unknown text encoder {kind!r}")


def init_vision_encoder(encoder: nn.Module, name_or_path: str) -> Dict[str, Any]:
    """Initialise E_psi from DINOv2 (``pretrained_encoder.py``, imported only when this option is used)."""
    try:
        from .pretrained_encoder import init_vision_encoder as _init
    except ImportError:
        from pretrained_encoder import init_vision_encoder as _init
    return _init(encoder, name_or_path)


_TUPLE_FIELDS = ("image_size", "selector_hidden")


def resolve_overrides(args: argparse.Namespace) -> Dict[str, Any]:
    """VPAConfig overrides from ``--config-json`` and the individual command-line flags."""
    overrides: Dict[str, Any] = {}
    if args.config_json:
        with open(args.config_json, encoding="utf-8") as f:
            overrides.update(json.load(f))
    known = {f.name for f in dataclasses.fields(VPAConfig)}
    unknown = sorted(set(overrides) - known)
    if unknown:
        raise ValueError(f"unknown VPAConfig fields in {args.config_json}: {unknown}")
    cli = {
        "horizon": args.horizon,
        "min_horizon": args.min_horizon,
        "integration_steps": args.integration_steps,
        "predictor_stride": args.predictor_stride,
        "num_primitives": args.num_primitives,
        "latent_dim": args.latent_dim,
        "patch_size": args.patch_size,
        "clip_model_name": args.clip_model,
    }
    overrides.update({k: v for k, v in cli.items() if v is not None})
    if args.image_size is not None:
        overrides["image_size"] = (args.image_size, args.image_size)
    for key in _TUPLE_FIELDS:
        if key in overrides:
            overrides[key] = tuple(int(v) for v in overrides[key])
    return overrides


def camera_keys_of(data_info: Dict[str, Any]) -> Tuple[str, ...]:
    """Camera keys (in view order) recorded in a checkpoint's ``extra["data"]``.

    Checkpoints written before multi-camera support store a single ``camera_key``.
    """
    keys = data_info.get("camera_keys")
    if keys is None:
        keys = [data_info.get("camera_key", DEFAULT_CAMERA_KEY)]
    return tuple(str(k) for k in keys)


def make_config(overrides: Dict[str, Any], ds: LiberoHDF5Dataset) -> VPAConfig:
    """VPAConfig = defaults <- overrides <- quantities fixed by the data (d_s, d_a, C, N_u >= labels)."""
    values = dict(overrides)
    if "image_size" not in values:  # default: the frames' native resolution
        values["image_size"] = (ds.native_image_shape[0], ds.native_image_shape[1])
    values["in_channels"] = ds.native_image_shape[2]
    if int(values.get("num_views", ds.num_views)) != ds.num_views:
        raise ValueError(f"num_views = {values['num_views']} but {ds.num_views} camera key(s) were given: "
                         f"{list(ds.camera_keys)}")
    values["num_views"] = ds.num_views
    values["proprio_dim"] = ds.proprio_dim
    values["action_dim"] = ds.action_dim
    values["num_primitives"] = int(values.get("num_primitives", ds.num_primitives))
    if values["num_primitives"] < ds.num_primitives:
        raise ValueError(f"num_primitives = {values['num_primitives']} but the labels use {ds.num_primitives}")
    return VPAConfig(**values)


def check_data_matches(cfg: VPAConfig, ds: LiberoHDF5Dataset) -> None:
    """The dataset must produce what the configured modules consume."""
    checks = [
        ("proprio_dim (d_s)", ds.proprio_dim, cfg.proprio_dim),
        ("action_dim (d_a)", ds.action_dim, cfg.action_dim),
        ("image channels", ds.image_shape[0], cfg.in_channels),
        ("camera views V", ds.num_views, cfg.num_views),
        ("image size", ds.image_shape[1:], tuple(cfg.image_size)),
        ("horizon H", ds.horizon, cfg.horizon),
        ("predictor stride nu", ds.predictor_stride, cfg.predictor_stride),
    ]
    for name, got, want in checks:
        if got != want:
            raise ValueError(f"data/config mismatch for {name}: data {got}, config {want}")
    if ds.num_primitives > cfg.num_primitives:
        raise ValueError(f"labels use {ds.num_primitives} primitives, config has {cfg.num_primitives}")


# ---------------------------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------------------------
def model_state(pipe: VPAInferencePipeline, include_text: bool) -> Dict[str, Tensor]:
    """Pipeline state (incl. gamma_bar, mu_Z, mu_S, sigma_S buffers); pre-trained CLIP weights are omitted."""
    return {k: v for k, v in pipe.state_dict().items() if include_text or not k.startswith("text_encoder.")}


def load_model_state(pipe: VPAInferencePipeline, state: Dict[str, Tensor], text_kind: str) -> None:
    missing, unexpected = pipe.load_state_dict(state, strict=False)
    allowed = [k for k in missing if k.startswith("text_encoder.")] if text_kind == "clip" else []
    if unexpected or len(allowed) != len(missing):
        raise RuntimeError(f"checkpoint does not match the model: missing {sorted(set(missing) - set(allowed))}, "
                           f"unexpected {sorted(unexpected)}")


def save_checkpoint(
    path: str,
    *,
    stage: str,
    step: int,
    cfg: VPAConfig,
    pipe: VPAInferencePipeline,
    trainer_state: Optional[Dict[str, Any]],
    extra: Dict[str, Any],
    momentum: Optional[MomentumEncoder] = None,
) -> None:
    """Write a checkpoint atomically (temporary file + rename)."""
    payload = {
        "format": CHECKPOINT_FORMAT,
        "stage": stage,
        "step": int(step),
        "config": dataclasses.asdict(cfg),
        "model": model_state(pipe, include_text=extra["text_encoder"]["kind"] != "clip"),
        "trainer": trainer_state,
        "momentum_encoder": None if momentum is None else momentum.encoder.state_dict(),
        "ema_momentum": None if momentum is None else momentum.momentum,
        "extra": extra,
        "rng": {
            "torch": torch.get_rng_state(),
            "python": random.getstate(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        },
    }
    tmp = f"{path}.tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)


def load_checkpoint(path: str) -> Dict[str, Any]:
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    if ckpt.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(f"{path} is not a {CHECKPOINT_FORMAT} checkpoint")
    return ckpt


def _restore_rng(ckpt: Dict[str, Any]) -> None:
    rng = ckpt.get("rng") or {}
    if "torch" in rng:
        torch.set_rng_state(rng["torch"])
    if "python" in rng:
        random.setstate(rng["python"])
    if rng.get("cuda") and torch.cuda.is_available() and len(rng["cuda"]) == torch.cuda.device_count():
        torch.cuda.set_rng_state_all(rng["cuda"])


def _prune(out_dir: str, stage: str, keep: int) -> None:
    if keep <= 0:
        return
    paths = sorted(glob.glob(os.path.join(out_dir, f"{stage}_step*.pt")))
    for old in paths[:-keep]:
        os.remove(old)


def load_pipeline(
    path: str, text_encoder: Optional[TextEncoderWrapper] = None, device: str = "cpu"
) -> VPAInferencePipeline:
    """Rebuild a ``VPAInferencePipeline`` from a training checkpoint.

    A stage-2 checkpoint is calibrated again with its stored gamma*, mu_Z, mu_S and sigma_S, so
    E_psi and P_omega are frozen and the pipeline is ready for ``reset`` / ``step`` / ``act``.

    Args:
        path: checkpoint written by this script.
        text_encoder: optional pre-built text encoder (default: rebuilt from the checkpoint record).
        device: target device.
    """
    ckpt = load_checkpoint(path)
    cfg = VPAConfig(**ckpt["config"])
    info = ckpt["extra"]["text_encoder"]
    if text_encoder is None:
        text_encoder = build_text_encoder(info["kind"], info["name"], info["embed_dim"])
    pipe = VPAInferencePipeline.from_config(cfg, text_encoder=text_encoder)
    load_model_state(pipe, ckpt["model"], info["kind"])
    if pipe.solver.is_calibrated and bool(pipe.tracker.frozen):
        pipe.calibrate(
            pipe.tracker.gamma_bar_star,
            pipe.solver.latent_mean.clone(),
            pipe.solver.proprio_mean.clone(),
            pipe.solver.proprio_std.clone(),
        )
    return pipe.to(device)


# ---------------------------------------------------------------------------------------------
# Stage 1: E_psi and P_omega (Sec. 4.3, Fig. 3, Eqs. 9b, 11, 18, 19)
# ---------------------------------------------------------------------------------------------
class JEPATrainer:
    """Optimisation state of stage 1: AdamW over (psi, omega), schedule, EMA target, step counter.

    Args:
        pipe: the pipeline whose ``vision_encoder`` (E_psi), ``predictor`` (P_omega) and ``tracker``
            (gamma_bar, Eq. 9b) are trained / updated.
        vicreg: the Eq. 11 objective (built by ``pipe.make_vicreg_loss`` so eps is shared with Eq. 9b).
        momentum: E_psi_bar.
        lr, weight_decay, betas, warmup_steps, total_steps, min_lr_ratio, grad_clip: optimiser settings.
        amp: "none" or "bf16" (CUDA only; the loss is always evaluated in float32).
        device: training device.
        encoder_lr: optional learning rate for E_psi's body (every encoder parameter except the latent
            head), e.g. lower than ``lr`` when E_psi starts from pre-trained weights. None: ``lr``.
    """

    def __init__(
        self,
        pipe: VPAInferencePipeline,
        vicreg: VICRegLoss,
        momentum: MomentumEncoder,
        *,
        lr: float,
        weight_decay: float,
        betas: Tuple[float, float],
        warmup_steps: int,
        total_steps: int,
        min_lr_ratio: float,
        grad_clip: float,
        amp: str,
        device: torch.device,
        encoder_lr: Optional[float] = None,
    ) -> None:
        pipe.check_vicreg_loss(vicreg)   # Eq. 9b and Eq. 18 share eps
        if bool(pipe.tracker.frozen):
            raise RuntimeError("gamma is frozen: stage 1 has already finished for this model")
        self.pipe, self.vicreg, self.momentum = pipe, vicreg, momentum
        self.momentum.eval()
        self.device, self.amp, self.grad_clip = device, amp, grad_clip
        named = (list(pipe.vision_encoder.named_parameters(prefix="vision_encoder"))
                 + list(pipe.predictor.named_parameters(prefix="predictor")))
        self.params: List[Tensor] = [p for _, p in named]

        def body_lr(name: str) -> Optional[float]:
            is_body = name.startswith("vision_encoder.") and not name.startswith("vision_encoder.head.")
            return encoder_lr if is_body else None

        self.encoder_lr = encoder_lr
        self.optimizer = make_optimizer(named, lr, weight_decay, betas, device,
                                        lr_override=None if encoder_lr is None else body_lr)
        self.scheduler = make_scheduler(self.optimizer, warmup_steps, total_steps, min_lr_ratio)
        self.step: int = 0

    def forward(self, batch: Dict[str, Tensor]) -> Tuple[Tensor, Dict[str, Tensor], Tensor]:
        """Eq. 11 on one batch.

        Args:
            batch: ``image [B, *frame_shape]`` (I_t), ``next_image [B, *frame_shape]`` (I_{t+nu}), where
                ``frame_shape`` is ``[C, H_img, W_img]``, or ``[V, C, H_img, W_img]`` with V camera views,
                ``primitive [B]`` (u_t).

        Returns:
            (L_JEPA scalar, dict of the invariance / variance / covariance terms, z_t ``[B, d]``).
        """
        images = batch["image"].to(self.device, non_blocking=True)            # I_t
        next_images = batch["next_image"].to(self.device, non_blocking=True)  # I_{t+nu}
        u_t = batch["primitive"].to(self.device, non_blocking=True)           # [B] labels
        with _autocast(self.device, self.amp):
            z_t = self.pipe.vision_encoder(images)                            # [B, d]  Z^(n), Eq. 7
            # Eq. 11 applies the prediction term to each head. P_omega(...) would also compute sigma,
            # which needs gamma* -- defined only after this stage -- so the heads are read directly.
            heads = self.pipe.predictor.forward_heads(z_t, u_t)               # [N_e, B, d]  Eq. 10
            z_target = self.momentum(next_images)                             # [B, d]  sg(E_psi_bar(I_{t+nu}))
        loss, terms = self.vicreg(z_t.float(), heads.float(), z_target.float())  # Eq. 11
        return loss, terms, z_t

    def train_step(self, batch: Dict[str, Tensor]) -> Dict[str, float]:
        """One optimisation step, then the EMA update of E_psi_bar and Eq. 9b on Z^(n)."""
        self.pipe.vision_encoder.train()
        self.pipe.predictor.train()
        loss, terms, z_t = self.forward(batch)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite L_JEPA at step {self.step}: {terms}")
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = _clip(self.params, self.grad_clip)
        self.optimizer.step()
        self.scheduler.step()
        self.momentum.update(self.pipe.vision_encoder)                        # psi_bar <- m psi_bar + (1-m) psi
        z_detached = z_t.detach().float()
        gamma_n = self.pipe.tracker.update_variance_margin(z_detached)        # Eq. 9b with Z^(n)
        self.step += 1
        record = {
            "loss": float(loss.detach()),
            "invariance": float(terms["invariance"]),
            "variance": float(terms["variance"]),
            "covariance": float(terms["covariance"]),
            "gamma_bar": gamma_n,
            "tau": self.pipe.tracker.threshold,                               # tau_n, Eq. 9c
            "latent_std": float(z_detached.std(dim=0).mean()),
            "grad_norm": grad_norm,
            "lr": self.optimizer.param_groups[-1]["lr"],                       # groups at the base rate come last
        }
        if self.encoder_lr is not None:
            record["lr_encoder"] = self.optimizer.param_groups[0]["lr"]
        return record

    @torch.no_grad()
    def evaluate(self, loader: Any, max_batches: int) -> Dict[str, float]:
        """Mean Eq. 11 terms on held-out demonstrations (no EMA or gamma update)."""
        self.pipe.vision_encoder.eval()
        self.pipe.predictor.eval()
        sums: Dict[str, float] = defaultdict(float)
        n = 0
        for i, batch in enumerate(loader):
            if i >= max_batches:
                break
            loss, terms, _ = self.forward(batch)
            sums["loss"] += float(loss)
            for key, value in terms.items():
                sums[key] += float(value)
            n += 1
        return {f"val_{k}": v / max(n, 1) for k, v in sums.items()}

    def state_dict(self) -> Dict[str, Any]:
        return {"optimizer": self.optimizer.state_dict(), "scheduler": self.scheduler.state_dict(), "step": self.step}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.optimizer.load_state_dict(state["optimizer"])
        self.scheduler.load_state_dict(state["scheduler"])
        self.step = int(state["step"])


def finish_jepa(pipe: VPAInferencePipeline) -> float:
    """End of stage 1 (Sec. 4.3): freeze E_psi and P_omega and fix gamma* = gamma_N; returns tau* (Eq. 9c)."""
    tau_star = pipe.tracker.freeze()
    pipe.vision_encoder.requires_grad_(False)
    pipe.predictor.requires_grad_(False)
    return tau_star


# ---------------------------------------------------------------------------------------------
# Stage 2 preparation: latent cache, mu_Z, calibration, milestone phases (Secs. 4.2, 4.5)
# ---------------------------------------------------------------------------------------------
@torch.no_grad()
def encode_frames(
    encoder: nn.Module, ds: LiberoHDF5Dataset, device: torch.device, num_workers: int, block_size: int
) -> Tensor:
    """z = E_psi(I) for every frame of every episode of ``ds``: ``[N_frames, d]`` float32 on the CPU.

    Row ``ds.frame_offsets[e] + t`` holds the latent of frame t of episode e.
    """
    encoder.eval()
    frames = LiberoFrameDataset(ds, block_size=block_size)
    loader = get_frame_loader(frames, num_workers=num_workers)
    latents = torch.empty(ds.num_frames, encoder.latent_dim)
    filled = torch.zeros(ds.num_frames, dtype=torch.bool)
    for item in loader:
        z = encoder(item["images"].to(device, non_blocking=True))           # [n, d]
        latents[item["flat_index"]] = z.float().cpu()
        filled[item["flat_index"]] = True
    if not bool(filled.all()):
        raise RuntimeError("some frames were not encoded")
    return latents


def episode_frame_index(ds: LiberoHDF5Dataset, episode_ids: Sequence[int]) -> Tensor:
    """Flat frame indices of the given episodes: ``[n]`` int64."""
    off = ds.frame_offsets
    parts = [torch.arange(int(off[e]), int(off[e + 1]), dtype=torch.long) for e in episode_ids]
    return torch.cat(parts) if parts else torch.zeros(0, dtype=torch.long)


@torch.no_grad()
def encode_instructions(text_encoder: TextEncoderWrapper, instructions: Sequence[str], device: torch.device,
                        batch_size: int = 256) -> Tensor:
    """c_text for every task: ``[N_tasks, d_c]`` on ``device``."""
    parts = [text_encoder(list(instructions[i:i + batch_size])).float()
             for i in range(0, len(instructions), batch_size)]
    return torch.cat(parts).to(device)


@torch.no_grad()
def milestone_goal_index(
    pipe: VPAInferencePipeline,
    ds: LiberoHDF5Dataset,
    latents: Tensor,
    mode: str,
    device: torch.device,
    chunk: int = 256,
) -> Tuple[Tensor, Dict[str, float]]:
    """Flat index of the active milestone frame z_g^(m_t) for every frame: ``[N_frames]`` int64.

    "threshold": m_t follows Eq. 9a with the frozen tau* along each demonstration (m_0 = 1, every
    frame is a decision step, the pointer used at t is m_t). Milestone frames are frames of the same
    demonstration, so z_g^(m) = E_psi(I_g^(m)) is read from the latent cache.
    "index": m_t = 1 + #{k < M : idx_k < t} (``milestone_phase_by_index``).

    Returns:
        goal index ``[N_frames]`` and statistics (fraction of completed episodes, fraction of frames
        whose threshold phase differs from the index phase).
    """
    offsets = ds.frame_offsets
    index_goal = torch.empty(ds.num_frames, dtype=torch.long)
    for e, ep in enumerate(ds.episodes):
        off = int(offsets[e])
        phase = milestone_phase_by_index(ep.milestones, ep.length)                       # [T], 1-based
        index_goal[off:off + ep.length] = torch.from_numpy(off + np.asarray(ep.milestones, dtype=np.int64)[phase - 1])
    if mode == "index":
        return index_goal, {"episodes": float(len(ds.episodes))}
    if mode != "threshold":
        raise ValueError(f"unknown milestone phase mode {mode!r}")
    if not bool(pipe.tracker.frozen):
        raise RuntimeError("Eq. 9a segmentation uses tau*; freeze gamma first")
    goal = torch.full((ds.num_frames,), -1, dtype=torch.long)
    by_m: Dict[int, List[int]] = defaultdict(list)
    for e, ep in enumerate(ds.episodes):
        by_m[len(ep.milestones)].append(e)
    completed = 0
    tracker = pipe.tracker
    for num_m, episodes in sorted(by_m.items()):
        for start in range(0, len(episodes), chunk):
            group = episodes[start:start + chunk]
            lengths = torch.tensor([ds.episodes[e].length for e in group], dtype=torch.long)
            t_max = int(lengths.max())
            z_seq = torch.zeros(len(group), t_max, latents.shape[1])
            ms = torch.empty(len(group), num_m, dtype=torch.long)                         # flat milestone index
            for i, e in enumerate(group):
                off, length = int(offsets[e]), ds.episodes[e].length
                z_seq[i, :length] = latents[off:off + length]
                ms[i] = off + torch.tensor(ds.episodes[e].milestones, dtype=torch.long)
            z_seq = z_seq.to(device)
            z_goals = latents[ms].to(device)                                              # [b, M, d]
            alive = lengths.to(device)
            history = torch.empty(len(group), t_max, dtype=torch.long, device=device)
            tracker.reset(num_m, len(group), device)
            for t in range(t_max):
                update = tracker.step(z_seq[:, t], z_goals, update_mask=alive > t)       # Eq. 9a with tau*
                history[:, t] = update.m_t
            assert tracker.task_complete is not None
            completed += int(tracker.task_complete.sum())
            history = history.cpu()
            for i, e in enumerate(group):
                off, length = int(offsets[e]), ds.episodes[e].length
                goal[off:off + length] = ms[i][history[i, :length] - 1]
    if bool((goal < 0).any()):
        raise RuntimeError("milestone segmentation left frames unassigned")
    stats = {
        "episodes": float(len(ds.episodes)),
        "completed_fraction": completed / max(1, len(ds.episodes)),
        "frames_ahead_of_index_phase": float((goal != index_goal).float().mean()),
    }
    return goal, stats


@dataclasses.dataclass
class PolicyData:
    """Frozen-encoder data for stage 2.

    Attributes:
        latents: ``[N_frames, d]`` E_psi(I) of every frame (row = ``frame_offsets[e] + t``).
        goal_index: ``[N_frames]`` int64 row of z_g^(m_t) for every frame.
        offsets: ``[E + 1]`` int64 frame offsets of the episodes.
        c_text: ``[N_tasks, d_c]`` instruction embeddings (training device).
    """

    latents: Tensor
    goal_index: Tensor
    offsets: Tensor
    c_text: Tensor

    def lookup(self, episode: Tensor, timestep: Tensor, device: torch.device) -> Tuple[Tensor, Tensor]:
        """(z_t ``[B, d]``, z_g^(m_t) ``[B, d]``) for a batch of (episode ``[B]``, t ``[B]``)."""
        dev = self.latents.device
        flat = self.offsets[episode.to(dev)] + timestep.to(dev)
        z_t = self.latents[flat]
        z_goal = self.latents[self.goal_index[flat]]
        return z_t.to(device, non_blocking=True), z_goal.to(device, non_blocking=True)


def _floor_std(std: Tensor, floor: float = 1e-6) -> Tuple[Tensor, List[int]]:
    bad = (std < floor) | ~torch.isfinite(std)
    return torch.where(bad, torch.ones_like(std), std), bad.nonzero().flatten().tolist()


def prepare_policy(
    args: argparse.Namespace,
    pipe: VPAInferencePipeline,
    ds: LiberoHDF5Dataset,
    train_ds: LiberoHDF5Dataset,
    device: torch.device,
    logger: logging.Logger,
) -> Tuple[PolicyData, Dict[str, float]]:
    """Freeze, encode, calibrate and segment (Secs. 4.2, 4.3, 4.5).

    If the solver already holds calibration statistics (resuming stage 2), they are kept.

    Returns:
        ``PolicyData`` and a dict of calibration values (gamma*, tau*, segmentation statistics).
    """
    if not bool(pipe.tracker.frozen):
        if not pipe.tracker.is_calibrated:
            raise RuntimeError("gamma has never been estimated; run stage 1 (--stage jepa) first")
        logger.warning("gamma is not frozen in this checkpoint; freezing it at gamma = %.6g",
                       float(pipe.tracker.gamma_bar))
        finish_jepa(pipe)
    gamma_star = pipe.tracker.gamma_bar_star
    pipe.vision_encoder.requires_grad_(False)
    pipe.predictor.requires_grad_(False)

    t0 = time.perf_counter()
    latents = encode_frames(pipe.vision_encoder, ds, device, args.num_workers, args.encode_block)
    logger.info("encoded %d frames with the frozen E_psi in %.1fs", ds.num_frames, time.perf_counter() - t0)
    train_rows = episode_frame_index(ds, train_ds.episode_ids.tolist())
    mu_z = latents[train_rows].double().mean(dim=0).float()                    # mu_Z (Sec. 4.5)
    mu_s, sd_s = train_ds.proprio_statistics()                                 # mu_S, sigma_S (Sec. 4.5)
    sd_s, floored = _floor_std(sd_s)
    if floored:
        logger.warning("proprioception dims %s have (near) zero variance; their sigma_S is set to 1", floored)
    if pipe.solver.is_calibrated:
        logger.info("solver already calibrated (resumed run): keeping its mu_Z, mu_S, sigma_S")
        mu_z = pipe.solver.latent_mean.detach().cpu().clone()
        mu_s = pipe.solver.proprio_mean.detach().cpu().clone()
        sd_s = pipe.solver.proprio_std.detach().cpu().clone()
    tau_star = pipe.calibrate(gamma_star, mu_z, mu_s, sd_s)                    # one gamma* everywhere

    c_text = encode_instructions(pipe.text_encoder, ds.instructions, device)   # [N_tasks, d_c]
    goal_index, seg = milestone_goal_index(pipe, ds, latents, args.milestone_phases, device)
    cache_device = device if args.cache_on_device else torch.device("cpu")
    data = PolicyData(
        latents=latents.to(cache_device),
        goal_index=goal_index.to(cache_device),
        offsets=torch.from_numpy(ds.frame_offsets).to(cache_device),
        c_text=c_text,
    )
    info = {"gamma_star": gamma_star, "tau_star": tau_star, **{f"milestones_{k}": v for k, v in seg.items()}}
    return data, info


# ---------------------------------------------------------------------------------------------
# Stage 2: selector and solver (Secs. 4.2, 4.4, 4.5, Eqs. 8, 10, 12-14)
# ---------------------------------------------------------------------------------------------
class PolicyTrainer:
    """Optimisation state of stage 2: AdamW over (phi, theta incl. Embed), schedule, step counter.

    Args:
        pipe: calibrated pipeline (E_psi, P_omega frozen).
        data: frozen-encoder data from ``prepare_policy``.
        selector_weight: w in L_solver + w * L_select.
        remaining arguments: as in ``JEPATrainer``.
    """

    def __init__(
        self,
        pipe: VPAInferencePipeline,
        data: PolicyData,
        *,
        lr: float,
        weight_decay: float,
        betas: Tuple[float, float],
        warmup_steps: int,
        total_steps: int,
        min_lr_ratio: float,
        grad_clip: float,
        selector_weight: float,
        amp: str,
        device: torch.device,
    ) -> None:
        if not (pipe.solver.is_calibrated and bool(pipe.tracker.frozen)):
            raise RuntimeError("stage 2 needs a calibrated pipeline (prepare_policy)")
        self.pipe, self.data, self.device, self.amp = pipe, data, device, amp
        self.grad_clip, self.selector_weight = grad_clip, selector_weight
        named = (list(pipe.selector.named_parameters(prefix="selector"))
                 + list(pipe.solver.named_parameters(prefix="solver")))
        self.selector_params = list(pipe.selector.parameters())
        self.solver_params = list(pipe.solver.parameters())
        self.optimizer = make_optimizer(named, lr, weight_decay, betas, device)
        self.scheduler = make_scheduler(self.optimizer, warmup_steps, total_steps, min_lr_ratio)
        self.step: int = 0

    @torch.no_grad()
    def inputs(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """Frozen-module inputs of one batch.

        Returns:
            ``z_t [B, d]``, ``z_goal [B, d]`` (z_g^(m_t)), ``z_hat [B, d]`` (P_omega(z_t, u_t)),
            ``u [B]``, ``proprio [B, d_s]``, ``actions [B, H, d_a]``, ``c_text [B, d_c]``.
        """
        z_t, z_goal = self.data.lookup(batch["episode"], batch["timestep"], self.device)
        u_t = batch["primitive"].to(self.device, non_blocking=True)
        z_hat = self.pipe.predictor(z_t, u_t).z_hat                            # Eq. 10, frozen P_omega
        return {
            "z_t": z_t,
            "z_goal": z_goal,
            "z_hat": z_hat,
            "u": u_t,
            "proprio": batch["proprio"].to(self.device, non_blocking=True),
            "actions": batch["actions"].to(self.device, non_blocking=True),
            "c_text": self.data.c_text[batch["task"].to(self.device, non_blocking=True)],
        }

    def losses(
        self, x: Dict[str, Tensor], generator: Optional[torch.Generator] = None
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """(L_solver, L_select, selector logits ``[B, N_u]``, e_t ``[B, d_e]``)."""
        e_t = self.pipe.solver.build_conditioning(
            x["u"], x["z_t"], x["z_hat"], x["proprio"], x["c_text"]
        )                                                                       # Eq. 12 (Embed trainable)
        with _autocast(self.device, self.amp):
            loss_solver = self.pipe.solver.flow_matching_loss(x["actions"], e_t, generator=generator)  # Eq. 14
            logits = self.pipe.selector(x["z_t"], x["z_goal"], x["c_text"])   # pi_phi^h scores, Eq. 8
        loss_select = NeuroSymbolicSelector.loss(logits.float(), x["u"])      # cross-entropy, Sec. 4.2
        return loss_solver.float(), loss_select, logits, e_t

    def train_step(self, batch: Dict[str, Tensor]) -> Dict[str, float]:
        self.pipe.selector.train()
        self.pipe.solver.train()
        x = self.inputs(batch)
        loss_solver, loss_select, logits, _ = self.losses(x)
        loss = loss_solver + self.selector_weight * loss_select
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite stage-2 loss at step {self.step}")
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gn_solver = _clip(self.solver_params, self.grad_clip)
        gn_selector = _clip(self.selector_params, self.grad_clip)
        self.optimizer.step()
        self.scheduler.step()
        self.step += 1
        return {
            "loss": float(loss.detach()),
            "flow_matching": float(loss_solver.detach()),
            "selector_ce": float(loss_select.detach()),
            "selector_acc": float((logits.argmax(dim=-1) == x["u"]).float().mean()),
            "grad_norm_solver": gn_solver,
            "grad_norm_selector": gn_selector,
            "lr": self.optimizer.param_groups[0]["lr"],
        }

    @torch.no_grad()
    def evaluate(self, loader: Any, max_batches: int, seed: int = 0) -> Dict[str, float]:
        """Held-out losses, selector accuracy and the MSE of K-step sampled chunks (Sec. 4.4)."""
        self.pipe.selector.eval()
        self.pipe.solver.eval()
        gen = torch.Generator(device=self.device)
        gen.manual_seed(seed)
        sums: Dict[str, float] = defaultdict(float)
        n = 0
        for i, batch in enumerate(loader):
            if i >= max_batches:
                break
            x = self.inputs(batch)
            loss_solver, loss_select, logits, e_t = self.losses(x, generator=gen)
            chunk = self.pipe.solver.generate_chunk(e_t, generator=gen)       # [B, H, d_a], K Euler steps
            sums["flow_matching"] += float(loss_solver)
            sums["selector_ce"] += float(loss_select)
            sums["selector_acc"] += float((logits.argmax(dim=-1) == x["u"]).float().mean())
            sums["chunk_mse"] += float((chunk.float() - x["actions"]).pow(2).mean())
            n += 1
        return {f"val_{k}": v / max(n, 1) for k, v in sums.items()}

    def state_dict(self) -> Dict[str, Any]:
        return {"optimizer": self.optimizer.state_dict(), "scheduler": self.scheduler.state_dict(), "step": self.step}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.optimizer.load_state_dict(state["optimizer"])
        self.scheduler.load_state_dict(state["scheduler"])
        self.step = int(state["step"])


# ---------------------------------------------------------------------------------------------
# Loop
# ---------------------------------------------------------------------------------------------
def _format(metrics: Dict[str, float]) -> str:
    return " ".join(f"{k} {v:.4g}" for k, v in metrics.items())


def fit(
    trainer: Any,
    loader: Any,
    val_loader: Optional[Any],
    *,
    stage: str,
    total_steps: int,
    args: argparse.Namespace,
    batch_size: int,
    logger: logging.Logger,
    metrics: MetricsWriter,
    save: Any,
) -> None:
    """Shared loop: train steps, logging, validation and periodic checkpoints until ``total_steps``."""
    if trainer.step >= total_steps:
        logger.info("[%s] already at step %d of %d", stage, trainer.step, total_steps)
        return
    data = _cycle(loader)
    t_last, n_last = time.perf_counter(), 0
    while trainer.step < total_steps:
        epoch, batch = next(data)
        record = trainer.train_step(batch)
        n_last += batch_size
        step = trainer.step
        if step % args.log_every == 0 or step == total_steps:
            elapsed = time.perf_counter() - t_last
            record["samples_per_s"] = n_last / max(elapsed, 1e-9)
            logger.info("[%s] step %d/%d epoch %d %s", stage, step, total_steps, epoch, _format(record))
            metrics.write({"stage": stage, "step": step, "epoch": epoch, **record})
            t_last, n_last = time.perf_counter(), 0
        if val_loader is not None and args.val_every > 0 and (step % args.val_every == 0 or step == total_steps):
            val = trainer.evaluate(val_loader, args.val_batches)
            logger.info("[%s] step %d validation %s", stage, step, _format(val))
            metrics.write({"stage": stage, "step": step, **val})
        if args.ckpt_every > 0 and step % args.ckpt_every == 0 and step < total_steps:
            save(step)


def _make_loader(ds: LiberoHDF5Dataset, batch_size: int, args: argparse.Namespace, shuffle: bool) -> Any:
    return get_dataloader(ds, batch_size, shuffle=shuffle, num_workers=args.num_workers, seed=args.seed)


def _val_loader(val_ds: LiberoHDF5Dataset, batch_size: int, args: argparse.Namespace,
                logger: logging.Logger) -> Optional[Any]:
    if len(val_ds) < batch_size:
        if len(val_ds) > 0:
            logger.info("validation set has %d samples (< batch size %d); validation disabled", len(val_ds), batch_size)
        return None
    return _make_loader(val_ds, batch_size, args, shuffle=False)


def train_jepa(
    args: argparse.Namespace,
    cfg: VPAConfig,
    pipe: VPAInferencePipeline,
    train_ds: LiberoHDF5Dataset,
    val_ds: LiberoHDF5Dataset,
    device: torch.device,
    logger: logging.Logger,
    metrics: MetricsWriter,
    extra: Dict[str, Any],
    resume: Optional[Dict[str, Any]],
) -> None:
    """Stage 1, then ``finish_jepa`` and ``jepa_final.pt``."""
    momentum = MomentumEncoder(pipe.vision_encoder, momentum=args.ema_momentum).to(device)
    if resume is not None and resume.get("momentum_encoder") is not None:
        momentum.encoder.load_state_dict(resume["momentum_encoder"])
    vicreg = pipe.make_vicreg_loss(args.lambda_v, args.lambda_c, gamma=args.vicreg_gamma,
                                   head_reduction=args.head_reduction)
    trainer = JEPATrainer(
        pipe, vicreg, momentum, lr=args.jepa_lr, weight_decay=args.weight_decay, betas=tuple(args.betas),
        warmup_steps=args.warmup_steps, total_steps=args.jepa_steps, min_lr_ratio=args.min_lr_ratio,
        grad_clip=args.grad_clip, amp=args.amp, device=device, encoder_lr=args.encoder_lr,
    )
    if resume is not None and resume.get("trainer") is not None:
        trainer.load_state_dict(resume["trainer"])
        _restore_rng(resume)
        logger.info("[jepa] resumed at step %d", trainer.step)
    logger.info("[jepa] E_psi %.2fM params, P_omega %.2fM params, EMA momentum %g, lambda_v %g, lambda_c %g",
                count_parameters(pipe.vision_encoder) / 1e6, count_parameters(pipe.predictor) / 1e6,
                args.ema_momentum, args.lambda_v, args.lambda_c)

    def save(step: int, name: Optional[str] = None) -> None:
        path = os.path.join(args.out, name or f"jepa_step{step:07d}.pt")
        save_checkpoint(path, stage="jepa", step=step, cfg=cfg, pipe=pipe, trainer_state=trainer.state_dict(),
                        extra=extra, momentum=momentum)
        if name is None:
            _prune(args.out, "jepa", args.keep_ckpts)
        logger.info("[jepa] saved %s", path)

    fit(trainer, _make_loader(train_ds, args.batch_size, args, shuffle=True),
        _val_loader(val_ds, args.batch_size, args, logger), stage="jepa", total_steps=args.jepa_steps,
        args=args, batch_size=args.batch_size, logger=logger, metrics=metrics, save=save)
    tau_star = finish_jepa(pipe)
    logger.info("[jepa] finished: E_psi, P_omega frozen; gamma* = %.6g, tau* = %.6g",
                pipe.tracker.gamma_bar_star, tau_star)
    metrics.write({"stage": "jepa", "step": trainer.step, "gamma_star": pipe.tracker.gamma_bar_star,
                   "tau_star": tau_star})
    save(trainer.step, "jepa_final.pt")


def train_policy(
    args: argparse.Namespace,
    cfg: VPAConfig,
    pipe: VPAInferencePipeline,
    ds: LiberoHDF5Dataset,
    train_ds: LiberoHDF5Dataset,
    val_ds: LiberoHDF5Dataset,
    device: torch.device,
    logger: logging.Logger,
    metrics: MetricsWriter,
    extra: Dict[str, Any],
    resume: Optional[Dict[str, Any]],
) -> None:
    """Stage 2 (selector + solver warm start), then ``policy_final.pt``."""
    data, info = prepare_policy(args, pipe, ds, train_ds, device, logger)
    logger.info("[policy] calibration %s", _format(info))
    metrics.write({"stage": "policy", "step": 0, **info})
    batch_size = args.policy_batch_size or args.batch_size
    trainer = PolicyTrainer(
        pipe, data, lr=args.policy_lr, weight_decay=args.weight_decay, betas=tuple(args.betas),
        warmup_steps=args.warmup_steps, total_steps=args.policy_steps, min_lr_ratio=args.min_lr_ratio,
        grad_clip=args.grad_clip, selector_weight=args.selector_weight, amp=args.amp, device=device,
    )
    if resume is not None and resume.get("trainer") is not None:
        trainer.load_state_dict(resume["trainer"])
        _restore_rng(resume)
        logger.info("[policy] resumed at step %d", trainer.step)
    logger.info("[policy] pi_phi^h %.2fM params, pi_theta^l %.2fM params, K = %d, H = %d",
                count_parameters(pipe.selector) / 1e6, count_parameters(pipe.solver) / 1e6,
                cfg.integration_steps, cfg.horizon)
    train_lat = train_ds.subset(train_ds.episode_ids)
    train_lat.load_images = False
    val_lat = val_ds.subset(val_ds.episode_ids)
    val_lat.load_images = False

    def save(step: int, name: Optional[str] = None) -> None:
        path = os.path.join(args.out, name or f"policy_step{step:07d}.pt")
        save_checkpoint(path, stage="policy", step=step, cfg=cfg, pipe=pipe, trainer_state=trainer.state_dict(),
                        extra=extra)
        if name is None:
            _prune(args.out, "policy", args.keep_ckpts)
        logger.info("[policy] saved %s", path)

    fit(trainer, _make_loader(train_lat, batch_size, args, shuffle=True),
        _val_loader(val_lat, batch_size, args, logger), stage="policy", total_steps=args.policy_steps,
        args=args, batch_size=batch_size, logger=logger, metrics=metrics, save=save)
    pipe.eval()
    save(trainer.step, "policy_final.pt")


# ---------------------------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train VPA on LIBERO demonstrations (stage 1: JEPA, stage 2: policy).")
    io = p.add_argument_group("data and output")
    io.add_argument("--data", nargs="+", help="LIBERO .hdf5 files and/or directories")
    io.add_argument("--out", default="runs/vpa", help="output directory")
    io.add_argument("--stage", choices=("all", "jepa", "policy"), default="all")
    io.add_argument("--resume", help="continue from a checkpoint (model, optimiser, schedule, step)")
    io.add_argument("--init-from", help="load model weights only (e.g. jepa_final.pt for --stage policy)")
    io.add_argument("--camera-keys", "--camera-key", dest="camera_keys", nargs="+", metavar="KEY",
                    help="camera(s) forming I_t, in view order (default: agentview_rgb, or the checkpoint's); "
                         "e.g. --camera-keys agentview_rgb eye_in_hand_rgb")
    io.add_argument("--proprio-keys", nargs="+", default=list(DEFAULT_PROPRIO_KEYS))
    io.add_argument("--no-rotate", action="store_true", help="do not rotate LIBERO frames by 180 degrees")
    io.add_argument("--primitive-source", choices=("auto", "hdf5", "gripper"), default="auto")
    io.add_argument("--milestone-source", choices=("auto", "hdf5", "final", "gripper"), default="auto")
    io.add_argument("--gripper-window", type=int, default=10)
    io.add_argument("--chunk-padding", choices=("repeat", "drop"), default="repeat")
    io.add_argument("--milestone-phases", choices=("threshold", "index"), default="threshold",
                    help="how m_t is assigned along demonstrations in stage 2 (Eq. 9a with tau*, or frame index)")
    io.add_argument("--val-fraction", type=float, default=0.05)

    model = p.add_argument_group("model (VPAConfig overrides)")
    model.add_argument("--config-json", help="JSON object of VPAConfig fields")
    model.add_argument("--image-size", type=int,
                       help="square frame size (default: the data's native size; 112 with --init-encoder)")
    model.add_argument("--patch-size", type=int, help="ViT patch size (default 16; 14 with --init-encoder)")
    model.add_argument("--init-encoder",
                       help="initialise E_psi from pre-trained DINOv2: 'dinov2-small' (ViT-S/14, matches the default "
                            "width 384 / 6 heads), a Hugging Face DINOv2 id or a local directory; new runs only")
    model.add_argument("--latent-dim", type=int)
    model.add_argument("--horizon", type=int, help="H = H_max")
    model.add_argument("--min-horizon", type=int, help="H_min")
    model.add_argument("--integration-steps", type=int, choices=(1, 2, 3), help="K")
    model.add_argument("--predictor-stride", type=int, help="nu <= H_min")
    model.add_argument("--num-primitives", type=int, help="N_u (default: from the labels)")
    model.add_argument("--clip-model", help="Hugging Face CLIP name (default: VPAConfig.clip_model_name)")
    model.add_argument("--text-encoder", choices=("clip", "toy"), default="clip",
                       help="'toy' is the offline stand-in used by the self-tests")

    opt = p.add_argument_group("optimisation")
    opt.add_argument("--jepa-steps", type=int, default=50_000)
    opt.add_argument("--policy-steps", type=int, default=50_000)
    opt.add_argument("--batch-size", type=int, default=64, help="B >= 2")
    opt.add_argument("--policy-batch-size", type=int, help="default: --batch-size")
    opt.add_argument("--jepa-lr", type=float, default=3e-4)
    opt.add_argument("--encoder-lr", type=float,
                     help="stage-1 learning rate of E_psi except its latent head (default: --jepa-lr); "
                          "e.g. 1e-4 with --init-encoder")
    opt.add_argument("--policy-lr", type=float, default=3e-4)
    opt.add_argument("--weight-decay", type=float, default=0.05)
    opt.add_argument("--betas", type=float, nargs=2, default=(0.9, 0.95))
    opt.add_argument("--warmup-steps", type=int, default=1000)
    opt.add_argument("--min-lr-ratio", type=float, default=0.05)
    opt.add_argument("--grad-clip", type=float, default=1.0, help="<= 0 disables clipping")
    opt.add_argument("--ema-momentum", type=float, default=0.996, help="momentum of E_psi_bar")
    opt.add_argument("--lambda-v", type=float, default=25.0)
    opt.add_argument("--lambda-c", type=float, default=1.0)
    opt.add_argument("--vicreg-gamma", type=float, default=1.0, help="variance margin gamma of Eq. 18")
    opt.add_argument("--head-reduction", choices=("mean", "sum"), default="mean")
    opt.add_argument("--selector-weight", type=float, default=1.0)

    sysg = p.add_argument_group("system")
    sysg.add_argument("--device", default="auto")
    sysg.add_argument("--amp", choices=("none", "bf16"), default="none", help="bf16 autocast (CUDA only)")
    sysg.add_argument("--tf32", action="store_true", help="allow TF32 matmuls on CUDA")
    sysg.add_argument("--num-workers", type=int, default=4)
    sysg.add_argument("--encode-block", type=int, default=256, help="frames per block when caching latents")
    sysg.add_argument("--cache-on-device", action="store_true", help="keep the stage-2 latent cache on the device")
    sysg.add_argument("--log-every", type=int, default=50)
    sysg.add_argument("--val-every", type=int, default=1000)
    sysg.add_argument("--val-batches", type=int, default=20)
    sysg.add_argument("--ckpt-every", type=int, default=5000)
    sysg.add_argument("--keep-ckpts", type=int, default=3, help="periodic checkpoints kept per stage (0 = all)")
    sysg.add_argument("--seed", type=int, default=0)
    sysg.add_argument("--self-test", action="store_true", help="run the offline self-test and exit")
    return p


def run(args: argparse.Namespace) -> VPAInferencePipeline:
    """Run the requested stage(s); returns the trained pipeline."""
    if not args.data:
        raise ValueError("--data is required")
    if args.resume and args.init_from:
        raise ValueError("use either --resume or --init-from, not both")
    if args.batch_size < 2 or (args.policy_batch_size is not None and args.policy_batch_size < 2):
        raise ValueError("batch sizes must be >= 2 (unbiased batch statistics, Eqs. 9b, 18, 19)")
    os.makedirs(args.out, exist_ok=True)
    logger = setup_logging(args.out)
    metrics = MetricsWriter(args.out)
    seed_everything(args.seed)
    device = resolve_device(args.device)
    if args.tf32 and device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    ckpt_path = args.resume or args.init_from
    ckpt = load_checkpoint(ckpt_path) if ckpt_path else None
    if args.stage == "policy" and ckpt is None:
        raise ValueError("--stage policy needs --init-from (a stage-1 checkpoint) or --resume")

    # ---- data and configuration (a checkpoint's own configuration wins over the flags)
    overrides = dict(ckpt["config"]) if ckpt is not None else resolve_overrides(args)
    if args.init_encoder and ckpt is not None:
        logger.warning("--init-encoder is ignored: the encoder weights come from %s", ckpt_path)
    if args.init_encoder and ckpt is None:
        overrides.setdefault("patch_size", 14)          # DINOv2 patch size
        overrides.setdefault("image_size", (112, 112))  # 8 x 8 patches, as many tokens as 128 / 16
    horizon = int(overrides.get("horizon", VPAConfig.horizon))
    stride = int(overrides.get("predictor_stride", VPAConfig.predictor_stride))
    stored_keys = camera_keys_of(ckpt["extra"]["data"]) if ckpt is not None else None
    camera_keys = tuple(args.camera_keys) if args.camera_keys else (stored_keys or (DEFAULT_CAMERA_KEY,))
    if stored_keys is not None and camera_keys != stored_keys:
        raise ValueError(f"--camera-keys {list(camera_keys)} differ from the checkpoint's {list(stored_keys)}")
    ds = LiberoHDF5Dataset(
        args.data, horizon, predictor_stride=stride, camera_keys=camera_keys, proprio_keys=args.proprio_keys,
        rotate_180=not args.no_rotate, primitive_source=args.primitive_source,
        milestone_source=args.milestone_source, gripper_window=args.gripper_window,
        chunk_padding=args.chunk_padding,
    )
    cfg = VPAConfig(**overrides) if ckpt is not None else make_config(overrides, ds)
    ds.image_size = (int(cfg.image_size[0]), int(cfg.image_size[1]))
    check_data_matches(cfg, ds)
    train_ds, val_ds = ds.split_episodes(args.val_fraction, seed=args.seed)
    logger.info("data: %d files, %d tasks, %d episodes (%d train / %d val), %d frames; %d train samples; "
                "labels %s (N_u = %d), milestones %s; d_s = %d, d_a = %d, frames %s, cameras %s",
                len(ds.files), len(ds.instructions), len(ds.episodes), train_ds.episode_ids.size,
                val_ds.episode_ids.size, ds.num_frames, len(train_ds), ds.primitive_source, cfg.num_primitives,
                ds.milestone_source, ds.proprio_dim, ds.action_dim, ds.frame_shape, list(ds.camera_keys))
    if ckpt is not None:
        stored = ckpt["extra"]["data"]["val_episodes"]
        if stored != val_ds.episode_ids.tolist():
            raise ValueError("the train/validation split differs from the checkpoint's (different data, "
                             "--val-fraction or --seed)")

    # ---- model
    text_info = ckpt["extra"]["text_encoder"] if ckpt is not None else {"kind": args.text_encoder, "embed_dim": 12}
    text_kind = text_info["kind"]
    text_encoder = build_text_encoder(text_kind, cfg.clip_model_name, text_info["embed_dim"])
    pipe = VPAInferencePipeline.from_config(cfg, text_encoder=text_encoder)
    encoder_init: Optional[Dict[str, Any]] = None
    if ckpt is not None:
        load_model_state(pipe, ckpt["model"], text_kind)
        encoder_init = ckpt["extra"].get("encoder_init")
        logger.info("loaded %s (stage %s, step %d)", ckpt_path, ckpt["stage"], ckpt["step"])
    elif args.init_encoder:
        encoder_init = init_vision_encoder(pipe.vision_encoder, args.init_encoder)  # before E_psi_bar is copied
        logger.info("E_psi initialised from %s: first %d of %d blocks, patch grid %s", encoder_init["source"],
                    encoder_init["blocks_used"], encoder_init["blocks_available"], encoder_init["patch_grid"])
    pipe.to(device)
    extra = {
        "data": {
            "files": ds.files, "instructions": ds.instructions, "camera_keys": list(ds.camera_keys),
            "proprio_keys": list(ds.proprio_keys), "rotate_180": ds.rotate_180,
            "primitive_source": ds.primitive_source, "milestone_source": ds.milestone_source,
            "gripper_window": ds.gripper_window, "chunk_padding": ds.chunk_padding,
            "val_episodes": val_ds.episode_ids.tolist(),
        },
        "text_encoder": {"kind": text_kind, "name": cfg.clip_model_name, "embed_dim": text_encoder.embed_dim},
        "encoder_init": encoder_init,
        "args": _json_safe(vars(args)),
    }
    with open(os.path.join(args.out, "config.json"), "w", encoding="utf-8") as f:
        json.dump({"config": _json_safe(dataclasses.asdict(cfg)), **_json_safe(extra)}, f, indent=2)
    logger.info("device %s, VPAConfig %s", device, _json_safe(dataclasses.asdict(cfg)))

    # ---- stages
    resume = ckpt if args.resume else None
    stage1_done = bool(pipe.tracker.frozen)
    if args.stage in ("all", "jepa"):
        if stage1_done:
            logger.info("[jepa] the loaded model already finished stage 1 (gamma* = %.6g)",
                        pipe.tracker.gamma_bar_star)
        else:
            train_jepa(args, cfg, pipe, train_ds, val_ds, device, logger, metrics, extra,
                       resume if resume is not None and resume["stage"] == "jepa" else None)
    if args.stage in ("all", "policy"):
        train_policy(args, cfg, pipe, ds, train_ds, val_ds, device, logger, metrics, extra,
                     resume if resume is not None and resume["stage"] == "policy" else None)
    ds.close()
    return pipe


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_arg_parser().parse_args(argv)
    if args.self_test:
        _self_test()
        return
    run(args)


# ---------------------------------------------------------------------------------------------
# Self-test (run: python train.py --self-test)
# ---------------------------------------------------------------------------------------------
_RTOL, _ATOL = 1e-5, 1e-6  # float32 module output vs float64 reference (CLAUDE.md)

_TINY_CONFIG = {
    "image_size": [16, 16], "patch_size": 8, "vit_width": 32, "vit_depth": 2, "vit_heads": 4, "latent_dim": 16,
    "selector_hidden": [32], "kappa": 0.1, "alpha_gamma": 0.5, "ensemble_heads": 3, "predictor_embed_dim": 8,
    "predictor_hidden": 32, "predictor_hidden_layers": 1, "primitive_embed_dim": 8, "horizon": 4,
    "min_horizon": 2, "integration_steps": 2, "beta": 0.5, "alpha_sigma": 0.2, "field_width": 32,
    "field_depth": 2, "field_heads": 4, "time_embed_dim": 16,
}


def _close(actual: float, ref: float, rtol: float = _RTOL, atol: float = _ATOL) -> bool:
    return abs(actual - ref) <= atol + rtol * abs(ref)


def _ref_jepa_loss(heads: Tensor, target: Tensor, z: Tensor, gamma: float, eps: float,
                   lambda_v: float, lambda_c: float) -> float:
    """Eq. 11 with Eqs. 18, 19 by explicit loops in float64 (head mean of the invariance term)."""
    h, tg, zz = heads.double().tolist(), target.double().tolist(), z.double().tolist()
    n_e, b, d = len(h), len(zz), len(zz[0])
    inv = sum(sum(sum((h[i][k][j] - tg[k][j]) ** 2 for j in range(d)) for k in range(b)) / b
              for i in range(n_e)) / n_e
    mean = [sum(zz[k][j] for k in range(b)) / b for j in range(d)]
    cov = [[sum((zz[k][i] - mean[i]) * (zz[k][j] - mean[j]) for k in range(b)) / (b - 1) for j in range(d)]
           for i in range(d)]
    v = sum(max(0.0, gamma - math.sqrt(cov[j][j] + eps)) for j in range(d)) / d
    c = sum(cov[i][j] ** 2 for i in range(d) for j in range(d) if i != j) / d
    return inv + lambda_v * v + lambda_c * c


def _ref_margin(z: Tensor, eps: float) -> float:
    """sqrt((1/d) sum_j Var(Z_.j) + eps), unbiased, float64 loops (Eq. 9b)."""
    zz = z.double().tolist()
    b, d = len(zz), len(zz[0])
    var = []
    for j in range(d):
        m = sum(zz[k][j] for k in range(b)) / b
        var.append(sum((zz[k][j] - m) ** 2 for k in range(b)) / (b - 1))
    return math.sqrt(sum(var) / d + eps)


def _snapshot(module: nn.Module) -> List[Tensor]:
    return [p.detach().clone() for p in module.parameters()]


def _same(a: List[Tensor], b: List[Tensor]) -> bool:
    return all(torch.equal(x, y) for x, y in zip(a, b, strict=True))


def _test_args(data: str, out: str, cfg_path: str, extra: Sequence[str] = ()) -> argparse.Namespace:
    return build_arg_parser().parse_args([
        "--data", data, "--out", out, "--config-json", cfg_path, "--text-encoder", "toy", "--device", "cpu",
        "--num-workers", "0", "--batch-size", "6", "--val-fraction", "0.3", "--milestone-source", "gripper",
        "--gripper-window", "2", "--warmup-steps", "1", "--jepa-lr", "1e-3", "--policy-lr", "1e-3",
        "--ema-momentum", "0.9", "--log-every", "1", "--val-every", "2", "--val-batches", "2",
        "--encode-block", "5", *extra,
    ])


def _self_test() -> None:
    torch.manual_seed(0)
    with tempfile.TemporaryDirectory() as tmp:
        data_dir = os.path.join(tmp, "data")
        os.makedirs(data_dir)
        _write_synthetic_libero(os.path.join(data_dir, "KITCHEN_SCENE1_put_the_bowl_on_the_plate_demo.hdf5"),
                                (14, 11, 16, 12), seed=5)
        _write_synthetic_libero(os.path.join(data_dir, "LIVING_ROOM_SCENE2_stack_the_blocks_demo.hdf5"),
                                (13, 10, 15), seed=6, instruction=None)
        cfg_path = os.path.join(tmp, "tiny.json")
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(_TINY_CONFIG, f)
        args = _test_args(data_dir, os.path.join(tmp, "unit"), cfg_path)
        os.makedirs(args.out)
        logger = setup_logging(args.out)
        device = torch.device("cpu")

        # ---- configuration from data + overrides
        ds = LiberoHDF5Dataset(data_dir, 4, primitive_source="auto", milestone_source="gripper", gripper_window=2)
        cfg = make_config(resolve_overrides(args), ds)
        ds.image_size = tuple(cfg.image_size)
        check_data_matches(cfg, ds)
        assert (cfg.proprio_dim, cfg.action_dim, cfg.num_primitives, cfg.in_channels) == (15, 7, 4, 3)
        assert cfg.image_size == (16, 16) and cfg.selector_hidden == (32,)
        assert max(len(e.milestones) for e in ds.episodes) >= 3                  # multi-milestone episodes
        train_ds, val_ds = ds.split_episodes(args.val_fraction, seed=args.seed)
        pipe = VPAInferencePipeline.from_config(cfg, text_encoder=build_text_encoder("toy", cfg.clip_model_name))
        momentum = MomentumEncoder(pipe.vision_encoder, momentum=args.ema_momentum)
        vicreg = pipe.make_vicreg_loss(args.lambda_v, args.lambda_c, gamma=args.vicreg_gamma)
        trainer = JEPATrainer(pipe, vicreg, momentum, lr=1e-3, weight_decay=0.05, betas=(0.9, 0.95),
                              warmup_steps=1, total_steps=10, min_lr_ratio=0.1, grad_clip=1.0, amp="none",
                              device=device)
        loader = get_dataloader(train_ds, 6, num_workers=0, seed=1)
        frozen_parts = [_snapshot(m) for m in (pipe.selector, pipe.solver, pipe.text_encoder)]

        # ---- stage 1, two steps: Eq. 11 value, EMA after the optimiser step, Eq. 9b on Z^(n)
        batches = iter(loader)
        gamma_prev = None
        for _ in range(2):
            batch = next(batches)
            with torch.no_grad():
                z_ref = pipe.vision_encoder(batch["image"])
                heads_ref = pipe.predictor.forward_heads(z_ref, batch["primitive"])
                target_ref = momentum(batch["next_image"])
            loss_ref = _ref_jepa_loss(heads_ref, target_ref, z_ref, args.vicreg_gamma, cfg.vicreg_eps,
                                      args.lambda_v, args.lambda_c)
            bar_before = [p.detach().double().clone() for p in momentum.encoder.parameters()]
            online_before = _snapshot(pipe.vision_encoder)
            out = trainer.train_step(batch)
            assert _close(out["loss"], loss_ref), (out["loss"], loss_ref)
            assert not _same(online_before, _snapshot(pipe.vision_encoder))
            for p_bar, before, p in zip(momentum.encoder.parameters(), bar_before,
                                        pipe.vision_encoder.parameters(), strict=True):
                ref = 0.9 * before + 0.1 * p.detach().double()
                assert torch.allclose(p_bar.double(), ref, rtol=_RTOL, atol=_ATOL)
            inst = _ref_margin(z_ref, cfg.vicreg_eps)
            gamma_ref = inst if gamma_prev is None else (1 - cfg.alpha_gamma) * gamma_prev + cfg.alpha_gamma * inst
            assert _close(float(pipe.tracker.gamma_bar), gamma_ref), (float(pipe.tracker.gamma_bar), gamma_ref)
            assert _close(out["tau"], cfg.kappa * math.sqrt(2 * cfg.latent_dim) * gamma_ref)
            gamma_prev = float(pipe.tracker.gamma_bar)
        assert trainer.step == 2
        assert all(_same(a, _snapshot(m)) for a, m in zip(frozen_parts, (pipe.selector, pipe.solver,
                                                                          pipe.text_encoder), strict=True))
        assert all(p.grad is None for p in momentum.encoder.parameters())
        val = trainer.evaluate(get_dataloader(val_ds, 2, shuffle=False, num_workers=0), 2)
        assert set(val) == {"val_loss", "val_invariance", "val_variance", "val_covariance"}
        assert _close(float(pipe.tracker.gamma_bar), gamma_prev)               # evaluate() changes nothing

        # ---- end of stage 1: gamma* = gamma_N, tau* (Eq. 9c), E_psi and P_omega frozen
        tau_star = finish_jepa(pipe)
        assert pipe.tracker.gamma_bar_star == gamma_prev
        assert _close(tau_star, cfg.kappa * math.sqrt(2 * cfg.latent_dim) * gamma_prev)
        try:
            JEPATrainer(pipe, vicreg, momentum, lr=1e-3, weight_decay=0.0, betas=(0.9, 0.95), warmup_steps=0,
                        total_steps=1, min_lr_ratio=0.1, grad_clip=0.0, amp="none", device=device)
            raise AssertionError("stage 1 must refuse a frozen gamma")
        except RuntimeError:
            pass

        # ---- stage 2 preparation: latent cache, mu_Z, mu_S, sigma_S, calibration, Eq. 9a phases
        data, info = prepare_policy(args, pipe, ds, train_ds, device, logger)
        assert not any(p.requires_grad for p in pipe.vision_encoder.parameters())
        assert not any(p.requires_grad for p in pipe.predictor.parameters())
        with torch.no_grad():
            for e, t in ((0, 0), (2, 7), (len(ds.episodes) - 1, ds.episodes[-1].length - 1)):
                z_single = pipe.vision_encoder(ds.frame(e, t)[None])[0]
                row = int(ds.frame_offsets[e]) + t
                # batched vs single-frame GEMMs accumulate in a different order
                assert torch.allclose(data.latents[row], z_single, rtol=1e-4, atol=1e-5)
            acc = torch.zeros(cfg.latent_dim, dtype=torch.float64)
            count = 0
            for e in train_ds.episode_ids.tolist():
                for t in range(ds.episodes[e].length):
                    acc += pipe.vision_encoder(ds.frame(e, t)[None])[0].double()
                    count += 1
        assert torch.allclose(pipe.solver.latent_mean.double(), acc / count, rtol=1e-4, atol=1e-5)
        prop = np.concatenate([ds.proprioception(e) for e in train_ds.episode_ids.tolist()]).astype(np.float64)
        assert np.allclose(pipe.solver.proprio_mean.numpy(), prop.mean(0), rtol=_RTOL, atol=_ATOL)
        assert np.allclose(pipe.solver.proprio_std.numpy(), prop.std(0, ddof=1), rtol=_RTOL, atol=_ATOL)
        assert float(pipe.solver.gamma_bar_star) == gamma_prev == float(pipe.predictor.latent_scale)
        assert _close(info["tau_star"], tau_star)
        # Eq. 9a reference (float64 loop, independent of MilestoneTracker)
        tau = pipe.tracker.threshold
        checked = 0
        for e, ep in enumerate(ds.episodes):
            off = int(ds.frame_offsets[e])
            z = data.latents[off:off + ep.length].double()
            m, goals, ambiguous = 1, [], False
            for t in range(ep.length):
                g = ep.milestones[m - 1]
                goals.append(off + g)
                dist = float(torch.linalg.vector_norm(z[t] - z[g]))
                ambiguous |= abs(dist - tau) < 1e-5
                if dist < tau and m < len(ep.milestones):
                    m += 1
            if not ambiguous:
                assert data.goal_index[off:off + ep.length].tolist() == goals, e
                checked += 1
        assert checked >= len(ds.episodes) - 1, checked
        index_goal, _ = milestone_goal_index(pipe, ds, data.latents, "index", device)
        for e, ep in enumerate(ds.episodes):
            off = int(ds.frame_offsets[e])
            ref = [off + ep.milestones[sum(1 for k in ep.milestones[:-1] if k < t)] for t in range(ep.length)]
            assert index_goal[off:off + ep.length].tolist() == ref
        c_ref = pipe.text_encoder(ds.instructions)
        assert torch.equal(data.c_text, c_ref)

        # ---- stage 2 step: e_t (Eq. 12), L_solver (Eqs. 13-14), L_select (Sec. 4.2), frozen E_psi / P_omega
        trainer2 = PolicyTrainer(pipe, data, lr=1e-3, weight_decay=0.05, betas=(0.9, 0.95), warmup_steps=1,
                                 total_steps=10, min_lr_ratio=0.1, grad_clip=1.0, selector_weight=1.0,
                                 amp="none", device=device)
        lat_ds = train_ds.subset(train_ds.episode_ids)
        lat_ds.load_images = False
        batch = next(iter(get_dataloader(lat_ds, 6, num_workers=0, seed=2)))
        x = trainer2.inputs(batch)
        rows = data.offsets[batch["episode"]] + batch["timestep"]
        assert torch.equal(x["z_t"], data.latents[rows])
        assert torch.equal(x["z_goal"], data.latents[data.goal_index[rows]])
        assert torch.equal(x["actions"], batch["actions"]) and torch.equal(x["c_text"], c_ref[batch["task"]])
        with torch.no_grad():
            assert torch.equal(x["z_hat"], pipe.predictor(x["z_t"], batch["primitive"]).z_hat)
        torch.manual_seed(123)
        loss_solver, loss_select, logits, e_t = trainer2.losses(x)
        sol = pipe.solver
        e_ref = torch.cat([
            sol.primitive_embedding.weight.detach().double()[batch["primitive"]],
            (x["z_t"].double() - sol.latent_mean.double()) / gamma_prev,
            (x["z_hat"].double() - sol.latent_mean.double()) / gamma_prev,
            (batch["proprio"].double() - sol.proprio_mean.double()) / sol.proprio_std.double(),
            c_ref[batch["task"]].double(),
        ], dim=1)
        assert torch.allclose(e_t.double(), e_ref, rtol=_RTOL, atol=_ATOL)
        torch.manual_seed(123)
        xi = torch.randn(batch["actions"].shape)
        rho = torch.rand(batch["actions"].shape[0])
        a_rho = rho.view(-1, 1, 1) * batch["actions"] + (1 - rho.view(-1, 1, 1)) * xi
        with torch.no_grad():
            vel = sol.vector_field(a_rho, rho, e_t).double()
        fm_ref = (vel - (batch["actions"].double() - xi.double())).pow(2).sum(dim=(1, 2)).mean()
        assert _close(float(loss_solver), float(fm_ref)), (float(loss_solver), float(fm_ref))
        lg = logits.detach().double()
        ce_ref = sum(float(torch.logsumexp(lg[i], 0) - lg[i, int(batch["primitive"][i])])
                     for i in range(lg.shape[0])) / lg.shape[0]
        assert _close(float(loss_select), ce_ref)
        frozen = [_snapshot(m) for m in (pipe.vision_encoder, pipe.predictor)]
        trainable = [_snapshot(m) for m in (pipe.selector, pipe.solver)]
        for b in get_dataloader(lat_ds, 6, num_workers=0, seed=3):
            trainer2.train_step(b)
            if trainer2.step == 2:
                break
        assert all(_same(a, _snapshot(m)) for a, m in zip(frozen, (pipe.vision_encoder, pipe.predictor), strict=True))
        assert not any(_same(a, _snapshot(m)) for a, m in zip(trainable, (pipe.selector, pipe.solver), strict=True))
        assert pipe.tracker.gamma_bar_star == gamma_prev
        val2 = trainer2.evaluate(get_dataloader(lat_ds, 2, shuffle=False, num_workers=0), 2)
        assert set(val2) == {"val_flow_matching", "val_selector_ce", "val_selector_acc", "val_chunk_mse"}
        ds.close()

        # ---- end to end through the CLI, then the checkpoint drives the inference pipeline (K + 3)
        e2e = _test_args(data_dir, os.path.join(tmp, "e2e"), cfg_path,
                         ["--jepa-steps", "3", "--policy-steps", "3", "--ckpt-every", "2"])
        trained = run(e2e)
        for name in ("jepa_final.pt", "policy_final.pt", "jepa_step0000002.pt", "policy_step0000002.pt",
                     "metrics.jsonl", "config.json", "train.log"):
            assert os.path.exists(os.path.join(e2e.out, name)), name
        with open(os.path.join(e2e.out, "metrics.jsonl"), encoding="utf-8") as f:
            records = [json.loads(line) for line in f]
        assert {r["stage"] for r in records} == {"jepa", "policy"}
        assert any("gamma_star" in r for r in records) and any("val_chunk_mse" in r for r in records)
        loaded = load_pipeline(os.path.join(e2e.out, "policy_final.pt"))
        ref_state, got_state = trained.state_dict(), loaded.state_dict()
        assert ref_state.keys() == got_state.keys()
        assert all(torch.equal(ref_state[k].cpu(), got_state[k]) for k in ref_state)
        assert bool(loaded.tracker.frozen) and loaded.solver.is_calibrated and not loaded.training
        check = LiberoHDF5Dataset(data_dir, 4, milestone_source="gripper", gripper_window=2, image_size=(16, 16))
        loaded.reset(check.instructions[0], check.milestone_frames(0)[None])
        step_out = loaded.step(check.frame(0, 0)[None], torch.from_numpy(check.proprioception(0)[:1].copy()))
        assert step_out.num_sequential_evaluations == cfg.integration_steps + 3
        assert step_out.action_chunk.shape == (1, cfg.horizon, cfg.action_dim)
        check.close()

        # ---- resume stage 1 from a periodic checkpoint
        res_dir = os.path.join(tmp, "resume")
        run(_test_args(data_dir, res_dir, cfg_path, ["--stage", "jepa", "--jepa-steps", "2", "--ckpt-every", "1"]))
        first = load_checkpoint(os.path.join(res_dir, "jepa_step0000001.pt"))
        assert first["step"] == 1 and not bool(first["model"]["tracker.frozen"])
        assert len(first["trainer"]["optimizer"]["state"]) > 0
        run(_test_args(data_dir, res_dir, cfg_path, ["--stage", "jepa", "--jepa-steps", "4",
                                                     "--resume", os.path.join(res_dir, "jepa_step0000001.pt")]))
        final = load_checkpoint(os.path.join(res_dir, "jepa_final.pt"))
        assert final["step"] == 4 and bool(final["model"]["tracker.frozen"])
        assert len(first["trainer"]["optimizer"]["param_groups"]) == 2      # no --encoder-lr: groups as before

        # ---- pre-trained E_psi initialisation and --encoder-lr (offline: a random DINOv2 saved to disk)
        from transformers import Dinov2Config, Dinov2Model

        dino_dir = os.path.join(tmp, "dino")
        Dinov2Model(Dinov2Config(hidden_size=32, num_hidden_layers=3, num_attention_heads=4, mlp_ratio=4,
                                 patch_size=8, image_size=32)).save_pretrained(dino_dir)
        dino_run = os.path.join(tmp, "dino_run")
        run(_test_args(data_dir, dino_run, cfg_path, ["--stage", "jepa", "--jepa-steps", "1",
                                                      "--init-encoder", dino_dir, "--encoder-lr", "1e-4"]))
        dino_ckpt = load_checkpoint(os.path.join(dino_run, "jepa_final.pt"))
        record = dino_ckpt["extra"]["encoder_init"]
        assert record["source"] == dino_dir and record["blocks_used"] == 2 and record["patch_grid"] == [2, 2]
        groups = dino_ckpt["trainer"]["optimizer"]["param_groups"]
        assert sorted({g["initial_lr"] for g in groups}) == [1e-4, 1e-3] and len(groups) == 4
        fresh = VPAInferencePipeline.from_config(VPAConfig(**dino_ckpt["config"]),
                                                 text_encoder=build_text_encoder("toy", "unused")).vision_encoder
        before = fresh.cls_token.detach().clone()
        init_vision_encoder(fresh, dino_dir)
        reference = Dinov2Model.from_pretrained(dino_dir).embeddings.cls_token.detach()
        assert torch.equal(fresh.cls_token.detach(), reference) and not torch.equal(before, reference)
        # ---- two cameras end to end: I_t, I_{t+nu}, the milestones and the latent cache all carry both
        # views; the checkpoint records the camera keys; the pipeline runs on [B, 2, C, H, W] (K + 3)
        keys2 = ["agentview_rgb", "eye_in_hand_rgb"]
        two = _test_args(data_dir, os.path.join(tmp, "two_cams"), cfg_path,
                         ["--jepa-steps", "2", "--policy-steps", "2", "--camera-keys", *keys2])
        run(two)
        ckpt2 = load_checkpoint(os.path.join(two.out, "policy_final.pt"))
        assert ckpt2["config"]["num_views"] == 2 and ckpt2["extra"]["data"]["camera_keys"] == keys2
        assert camera_keys_of(ckpt2["extra"]["data"]) == tuple(keys2)
        assert camera_keys_of({"camera_key": "agentview_rgb"}) == ("agentview_rgb",)      # older checkpoints
        assert load_checkpoint(os.path.join(e2e.out, "policy_final.pt"))["config"]["num_views"] == 1
        pipe2 = load_pipeline(os.path.join(two.out, "policy_final.pt"))
        assert pipe2.vision_encoder.num_views == 2
        assert pipe2.vision_encoder.head.in_features == 2 * _TINY_CONFIG["vit_width"]
        check2 = LiberoHDF5Dataset(data_dir, 4, camera_keys=keys2, milestone_source="gripper", gripper_window=2,
                                   image_size=(16, 16))
        pipe2.reset(check2.instructions[0], check2.milestone_frames(0)[None])         # [1, M, 2, C, H, W]
        out2 = pipe2.step(check2.frame(0, 0)[None], torch.from_numpy(check2.proprioception(0)[:1].copy()))
        assert out2.num_sequential_evaluations == cfg.integration_steps + 3
        assert out2.action_chunk.shape == (1, cfg.horizon, cfg.action_dim)
        check2.close()
        # continuing a two-camera checkpoint with other cameras is refused ...
        try:
            run(_test_args(data_dir, os.path.join(tmp, "two_cams_bad"), cfg_path,
                           ["--stage", "policy", "--policy-steps", "1", "--camera-keys", "agentview_rgb",
                            "--init-from", os.path.join(two.out, "jepa_final.pt")]))
            raise AssertionError("expected ValueError for camera keys that differ from the checkpoint")
        except ValueError:
            pass
        # ... and without --camera-keys the checkpoint's cameras are used
        run(_test_args(data_dir, os.path.join(tmp, "two_cams_policy"), cfg_path,
                       ["--stage", "policy", "--policy-steps", "1", "--init-from", os.path.join(two.out, "jepa_final.pt")]))
        cont = load_checkpoint(os.path.join(tmp, "two_cams_policy", "policy_final.pt"))
        assert cont["extra"]["data"]["camera_keys"] == keys2 and cont["config"]["num_views"] == 2
        _close_logging()
    print("train.py self-test passed")


if __name__ == "__main__":
    main()
