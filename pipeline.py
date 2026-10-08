# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
"""
pipeline.py -- Closed-loop VPA inference (preprint_261008.pdf, Fig. 2, Sec. 4, Prop. 5.1).

``VPAInferencePipeline`` strings the four stages together. One decision step follows the
dependency chain of Prop. 5.1 exactly:

    I_t --E_psi--> z_t --pi_phi^h--> u_t --P_omega--> (z_hat_{t+1}, sigma_{t+1}) --> e_t
        --v_theta--> A^(rho_1) --v_theta--> ... --v_theta--> A^(rho_K) = A_hat_t

i.e. K + 3 sequential network evaluations, independent of the chunk length H. The milestone
latents z_g^(1..M) and the text embedding c_text are computed once per episode in ``reset``
(the assumption of Prop. 5.1), so they are not on the per-step chain. Every ``step`` audits the
chain at runtime with forward hooks on the four networks and raises if the evaluations differ
from [E_psi, pi_phi^h, P_omega, v_theta x K] in count or order.

Per decision step (Fig. 2):
    1. z_t = E_psi(I_t)                                                       (Eq. 7)
    2. u_t = argmax pi_phi^h(u | z_t, z_g^(m_t), c_text)                      (Eq. 8)
    3. (z_hat_{t+1}, sigma_{t+1}) = P_omega(z_t, u_t)                         (Eq. 10)
    4. e_t = Concat(Embed(u_t), z_tilde_t, z_hat_tilde_{t+1}, S_tilde_t, c_text)   (Eq. 12)
    5. A_hat_t by K Euler steps of v_theta                                    (Sec. 4.4)
    6. sigma_bar_{t+1} and H_t                                                (Eq. 15)
    7. m_{t+1} by the threshold rule with tau*                                (Eq. 9a)
The primitive u_t, the pointer m_t and e_t are held fixed while the first H_t actions of A_hat_t
are executed (Sec. 3.2); ``act`` manages this execution prefix at control rate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, List, NamedTuple, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor

try:  # package import (e.g. `from vpa.pipeline import ...`)
    from .perception import TextEncoderWrapper, VisionEncoder
    from .predictor import JEPAPredictor, VICRegLoss
    from .selector import MilestoneTracker, NeuroSymbolicSelector
    from .solver import FlowMatchingSolver, UncertaintyHorizonFilter, executed_prefix_mask
except ImportError:  # flat import (files side by side, `python pipeline.py`)
    from perception import TextEncoderWrapper, VisionEncoder
    from predictor import JEPAPredictor, VICRegLoss
    from selector import MilestoneTracker, NeuroSymbolicSelector
    from solver import FlowMatchingSolver, UncertaintyHorizonFilter, executed_prefix_mask

__all__ = ["VPAConfig", "StepOutput", "ActOutput", "VPAInferencePipeline"]


# ---------------------------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------------------------
@dataclass
class VPAConfig:
    """Hyper-parameters of the VPA pipeline.

    The paper fixes the structure but not these values; the defaults are placeholders
    (d_a = 8 follows the paper's 7-DoF + gripper example). d_c is not set here: it is taken
    from the text encoder (``projection_dim`` of the CLIP model).
    """

    # Stage 1 -- perception
    image_size: Tuple[int, int] = (224, 224)
    patch_size: int = 16
    in_channels: int = 3
    vit_width: int = 384
    vit_depth: int = 6
    vit_heads: int = 6
    latent_dim: int = 256                      # d
    clip_model_name: str = "openai/clip-vit-base-patch32"
    # Stage 2 -- selector and milestone tracker
    num_primitives: int = 8                    # N_u
    selector_hidden: Tuple[int, ...] = (512, 512)
    kappa: float = 0.1                         # kappa in (0, 1)
    alpha_gamma: float = 0.01                  # alpha_gamma in (0, 1]
    vicreg_eps: float = 1e-4                   # eps of Eq. 18 (shared with Eq. 9b)
    # Stage 3 -- predictor
    ensemble_heads: int = 5                    # N_e
    predictor_embed_dim: int = 64
    predictor_hidden: int = 512
    predictor_hidden_layers: int = 2
    predictor_stride: int = 1                  # nu, must satisfy nu <= H_min
    # Stage 4 -- solver
    proprio_dim: int = 14                      # d_s
    action_dim: int = 8                        # d_a
    primitive_embed_dim: int = 64              # d_u
    horizon: int = 16                          # H = H_max
    min_horizon: int = 4                       # H_min
    integration_steps: int = 2                 # K in {1, 2, 3}
    beta: float = 1.0                          # beta > 0
    alpha_sigma: float = 0.1                   # alpha_sigma in (0, 1]
    field_width: int = 256
    field_depth: int = 4
    field_heads: int = 4
    time_embed_dim: int = 128


class StepOutput(NamedTuple):
    """Everything produced by one decision step (leading dimension B on every tensor)."""

    action_chunk: Tensor         # [B, H, d_a]  A_hat_t
    executed_horizon: Tensor     # [B] int64    H_t
    executed_mask: Tensor        # [B, H] bool  True on the executed prefix h < H_t
    primitive: Tensor            # [B] int64    u_t
    milestone_pointer: Tensor    # [B] int64    m_t used at this step (1-based)
    next_milestone_pointer: Tensor  # [B] int64 m_{t+1}
    milestone_distance: Tensor   # [B]          ||z_t - z_g^(m_t)||_2
    task_complete: Tensor        # [B] bool     Eq. 9a test satisfied with m_t = M (sticky)
    z_t: Tensor                  # [B, d]
    z_hat: Tensor                # [B, d]       z_hat_{t+1}
    sigma: Tensor                # [B]          sigma_{t+1}
    sigma_bar: Tensor            # [B]          sigma_bar_{t+1}
    conditioning: Tensor         # [B, d_e]     e_t
    decision_mask: Tensor        # [B] bool     elements whose state was committed at this step
    num_sequential_evaluations: int  # always K + 3


class ActOutput(NamedTuple):
    """Control-rate output of ``VPAInferencePipeline.act``."""

    action: Tensor                      # [B, d_a] action to execute now
    replanned: Tensor                   # [B] bool, elements that started a new decision step
    step_output: Optional[StepOutput]   # set when at least one element replanned


# ---------------------------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------------------------
class VPAInferencePipeline(nn.Module):
    """Goal-conditioned VPA closed-loop inference pipeline (Fig. 2).

    Args:
        vision_encoder: E_psi (frozen at deployment).
        text_encoder: frozen text encoder producing c_text.
        selector: pi_phi^h.
        tracker: milestone pointer and threshold (frozen gamma*).
        predictor: P_omega with ensemble uncertainty.
        solver: pi_theta^l (CFM vector field, Embed, standardisation statistics).
        horizon_filter: sigma filter and executed-horizon rule (Eq. 15).
        predictor_stride: nu, the predictor's temporal stride; must satisfy nu <= H_min.

    Episode state (held across loop iterations, reset by ``reset``):
        ``c_text [B, d_c]`` (text intent), ``milestone_latents [B, M, d]``, the pointer m_t and the
        task-complete flag (inside ``tracker``), sigma_bar (inside ``horizon_filter``), and the
        current execution prefix: chunk ``[B, H, d_a]``, H_t ``[B]`` and a cursor ``[B]``.
    """

    CHAIN_NAMES: Tuple[str, str, str, str] = ("E_psi", "pi_phi_h", "P_omega", "v_theta")

    def __init__(
        self,
        vision_encoder: VisionEncoder,
        text_encoder: TextEncoderWrapper,
        selector: NeuroSymbolicSelector,
        tracker: MilestoneTracker,
        predictor: JEPAPredictor,
        solver: FlowMatchingSolver,
        horizon_filter: UncertaintyHorizonFilter,
        predictor_stride: int = 1,
    ) -> None:
        super().__init__()
        self.vision_encoder = vision_encoder
        self.text_encoder = text_encoder
        self.selector = selector
        self.tracker = tracker
        self.predictor = predictor
        self.solver = solver
        self.horizon_filter = horizon_filter
        self.predictor_stride: int = int(predictor_stride)
        self._validate_dimensions()
        self.expected_chain: List[str] = (
            [self.CHAIN_NAMES[0], self.CHAIN_NAMES[1], self.CHAIN_NAMES[2]]
            + [self.CHAIN_NAMES[3]] * solver.num_integration_steps
        )
        # Runtime audit of the dependency chain.
        self._recording: bool = False
        self._chain_log: List[str] = []
        for name, module in zip(
            self.CHAIN_NAMES, (vision_encoder, selector, predictor, solver.vector_field), strict=True
        ):
            module.register_forward_hook(self._make_hook(name))
        # Episode state.
        self.c_text: Optional[Tensor] = None
        self.milestone_latents: Optional[Tensor] = None
        self._chunk: Optional[Tensor] = None
        self._executed_horizon: Optional[Tensor] = None
        self._cursor: Optional[Tensor] = None

    # ------------------------------------------------------------------ construction
    @classmethod
    def from_config(
        cls, config: VPAConfig, text_encoder: Optional[TextEncoderWrapper] = None
    ) -> "VPAInferencePipeline":
        """Instantiate every module from ``config`` (the text encoder may be injected)."""
        if text_encoder is None:
            text_encoder = TextEncoderWrapper(config.clip_model_name)
        d, d_c = config.latent_dim, text_encoder.embed_dim
        vision = VisionEncoder(
            image_size=config.image_size,
            patch_size=config.patch_size,
            in_channels=config.in_channels,
            width=config.vit_width,
            depth=config.vit_depth,
            num_heads=config.vit_heads,
            latent_dim=d,
        )
        selector = NeuroSymbolicSelector(d, d_c, config.num_primitives, config.selector_hidden)
        tracker = MilestoneTracker(d, config.kappa, config.alpha_gamma, config.vicreg_eps)
        predictor = JEPAPredictor(
            d,
            config.num_primitives,
            num_heads=config.ensemble_heads,
            primitive_embed_dim=config.predictor_embed_dim,
            hidden_dim=config.predictor_hidden,
            num_hidden_layers=config.predictor_hidden_layers,
        )
        solver = FlowMatchingSolver(
            num_primitives=config.num_primitives,
            primitive_embed_dim=config.primitive_embed_dim,
            latent_dim=d,
            proprio_dim=config.proprio_dim,
            text_dim=d_c,
            horizon=config.horizon,
            action_dim=config.action_dim,
            num_integration_steps=config.integration_steps,
            width=config.field_width,
            depth=config.field_depth,
            num_heads=config.field_heads,
            time_embed_dim=config.time_embed_dim,
        )
        horizon_filter = UncertaintyHorizonFilter(
            config.min_horizon, config.horizon, config.beta, config.alpha_sigma
        )
        return cls(vision, text_encoder, selector, tracker, predictor, solver, horizon_filter,
                   predictor_stride=config.predictor_stride)

    def _validate_dimensions(self) -> None:
        d = self.vision_encoder.latent_dim
        d_c = self.text_encoder.embed_dim
        checks = [
            ("selector.latent_dim", self.selector.latent_dim, d),
            ("tracker.latent_dim", self.tracker.latent_dim, d),
            ("predictor.latent_dim", self.predictor.latent_dim, d),
            ("solver.latent_dim", self.solver.latent_dim, d),
            ("selector.text_dim", self.selector.text_dim, d_c),
            ("solver.text_dim", self.solver.text_dim, d_c),
            ("predictor.num_primitives", self.predictor.num_primitives, self.selector.num_primitives),
            ("solver.num_primitives", self.solver.num_primitives, self.selector.num_primitives),
            ("horizon_filter.h_max", self.horizon_filter.h_max, self.solver.horizon),
        ]
        for name, got, want in checks:
            if got != want:
                raise ValueError(f"dimension mismatch: {name} = {got}, expected {want}")
        if not 1 <= self.predictor_stride <= self.horizon_filter.h_min:
            raise ValueError("the predictor stride nu must satisfy 1 <= nu <= H_min (Sec. 3.3)")

    def _make_hook(self, name: str) -> Callable[[nn.Module, Any, Any], None]:
        def hook(module: nn.Module, inputs: Any, output: Any) -> None:
            if self._recording:
                self._chain_log.append(name)

        return hook

    # ------------------------------------------------------------------ deployment calibration
    @torch.no_grad()
    def calibrate(
        self,
        gamma_bar_star: float,
        latent_mean: Tensor,
        proprio_mean: Tensor,
        proprio_std: Tensor,
    ) -> float:
        """Install the frozen statistics shared by Stages 2-4 and freeze E_psi and P_omega.

        gamma* fixes tau* = kappa sqrt(2d) gamma* (Eq. 9c), normalises sigma (Sec. 4.3) and scales
        the latent standardisation (Sec. 4.5), so the three are set together from one value.

        Args:
            gamma_bar_star: frozen latent scale gamma*.
            latent_mean: ``[d]`` mu_Z;  proprio_mean, proprio_std: ``[d_s]``.

        Returns:
            tau*.
        """
        tau_star = self.tracker.freeze(gamma_bar_star)
        self.predictor.set_latent_scale(gamma_bar_star)
        self.solver.set_standardization_stats(latent_mean, gamma_bar_star, proprio_mean, proprio_std)
        self.vision_encoder.requires_grad_(False)
        self.predictor.requires_grad_(False)
        self.eval()
        return tau_star

    def _check_calibrated(self) -> None:
        if not (bool(self.tracker.frozen) and self.predictor.has_latent_scale and self.solver.is_calibrated):
            raise RuntimeError("pipeline is not calibrated; call calibrate(gamma*, mu_Z, mu_S, sigma_S)")
        scales = (
            self.tracker.gamma_bar_star,
            float(self.predictor.latent_scale),
            float(self.solver.gamma_bar_star),
        )
        if not all(math.isclose(s, scales[0], rel_tol=1e-6) for s in scales):
            raise RuntimeError(f"inconsistent gamma* across tracker/predictor/solver: {scales}")

    # ------------------------------------------------------------------ shared eps (Eqs. 9b, 18)
    def make_vicreg_loss(
        self,
        lambda_v: float,
        lambda_c: float,
        gamma: float = 1.0,
        head_reduction: str = "mean",
    ) -> VICRegLoss:
        """Build the Eq. 11 objective with the same eps as the tracker's Eq. 9b.

        Eq. 9b uses "the numerical constant of Eq. 18", so v(Z) and gamma_bar must share one eps.
        This factory takes it from ``self.tracker.eps`` (set from ``VPAConfig.vicreg_eps``).

        Returns:
            ``VICRegLoss`` whose ``eps`` equals ``self.tracker.eps``.
        """
        return VICRegLoss(lambda_v, lambda_c, gamma=gamma, eps=self.tracker.eps,
                          head_reduction=head_reduction)  # type: ignore[arg-type]

    def check_vicreg_loss(self, loss: VICRegLoss) -> None:
        """Raise ``ValueError`` if ``loss.eps`` differs from the tracker's eps (Eqs. 9b, 18)."""
        if loss.eps != self.tracker.eps:
            raise ValueError(
                f"VICRegLoss eps = {loss.eps} but MilestoneTracker eps = {self.tracker.eps}; "
                "Eq. 9b must use the eps of Eq. 18"
            )

    # ------------------------------------------------------------------ episode management
    @torch.no_grad()
    def reset(self, instructions: Union[str, Sequence[str]], milestone_frames: Tensor) -> None:
        """Begin an episode: encode x and I_g^(1..M) once, set m_0 = 1 and sigma_bar = 0.

        Args:
            instructions: one string (shared by the batch) or B strings.
            milestone_frames: ``[B, M, C, H_img, W_img]`` ordered milestone goal frames.
        """
        self._check_calibrated()
        if milestone_frames.dim() != 5:
            raise ValueError("milestone_frames must have shape [B, M, C, H_img, W_img]")
        b, m = milestone_frames.shape[:2]
        device = milestone_frames.device
        c_text = self.text_encoder(instructions).to(device)                    # [B or 1, d_c]
        if c_text.shape[0] == 1 and b > 1:
            c_text = c_text.expand(b, -1).contiguous()
        if c_text.shape[0] != b:
            raise ValueError(f"got {c_text.shape[0]} instructions for a batch of {b}")
        self.c_text = c_text                                                     # [B, d_c]
        self.milestone_latents = self.vision_encoder.encode_milestones(milestone_frames)  # [B, M, d]
        self.tracker.reset(m, b, device)
        self.horizon_filter.reset(b, device)
        self._chunk, self._executed_horizon, self._cursor = None, None, None

    def _require_episode(self) -> Tuple[Tensor, Tensor]:
        if self.c_text is None or self.milestone_latents is None:
            raise RuntimeError("call reset(instructions, milestone_frames) before step/act")
        return self.c_text, self.milestone_latents

    # ------------------------------------------------------------------ one decision step
    @torch.no_grad()
    def step(
        self,
        observation: Tensor,
        proprioception: Tensor,
        decision_mask: Optional[Tensor] = None,
        noise: Optional[Tensor] = None,
    ) -> StepOutput:
        """Run one decision step: Observation -> Encoder -> Selector -> Predictor -> Solver.

        Args:
            observation: ``[B, C, H_img, W_img]`` current frames I_t.
            proprioception: ``[B, d_s]`` raw proprioceptive state S_t.
            decision_mask: optional ``[B]`` bool. The networks run for the whole batch, but the
                filter state, milestone pointer and execution prefix are committed only for
                masked elements (used by ``act`` when elements reach the end of their prefix at
                different control steps). Defaults to all True.
            noise: optional ``[B, H, d_a]`` initial flow sample xi.

        Returns:
            ``StepOutput``.
        """
        self._check_calibrated()
        c_text, milestone_latents = self._require_episode()
        b = observation.shape[0]
        if b != c_text.shape[0]:
            raise ValueError(f"observation batch {b} does not match the episode batch {c_text.shape[0]}")
        if proprioception.shape != (b, self.solver.proprio_dim):
            raise ValueError(f"proprioception must have shape [{b}, {self.solver.proprio_dim}]")
        if decision_mask is None:
            decision_mask = torch.ones(b, dtype=torch.bool, device=observation.device)
        decision_mask = decision_mask.to(device=observation.device, dtype=torch.bool)
        if self._chunk is None and not bool(decision_mask.all()):
            raise ValueError("the first decision step of an episode must include every batch element")

        # ---- the K + 3 dependency chain (Prop. 5.1) ----
        self._chain_log = []
        self._recording = True
        try:
            z_t = self.vision_encoder(observation)                                    # [B, d]
            z_goal = self.tracker.active_goal(milestone_latents)                      # [B, d] (gather)
            u_t = self.selector.select(z_t, z_goal, c_text)                           # [B]
            prediction = self.predictor(z_t, u_t)                                     # [B, d], [B]
            e_t = self.solver.build_conditioning(
                u_t, z_t, prediction.z_hat, proprioception, c_text
            )                                                                         # [B, d_e]
            chunk = self.solver.generate_chunk(e_t, noise=noise)                      # [B, H, d_a]
        finally:
            self._recording = False
        if self._chain_log != self.expected_chain:
            raise RuntimeError(
                f"sequential-depth violation: evaluated {self._chain_log}, expected {self.expected_chain}"
            )

        # ---- non-network updates (no additional depth) ----
        sigma_bar, executed_horizon = self.horizon_filter.update(prediction.sigma, decision_mask)  # Eq. 15
        milestone = self.tracker.step(z_t, milestone_latents, update_mask=decision_mask)          # Eq. 9a
        self._commit_prefix(chunk, executed_horizon, decision_mask)

        return StepOutput(
            action_chunk=chunk,
            executed_horizon=executed_horizon,
            executed_mask=executed_prefix_mask(executed_horizon, self.solver.horizon),
            primitive=u_t,
            milestone_pointer=milestone.m_t,
            next_milestone_pointer=milestone.m_next,
            milestone_distance=milestone.distance,
            task_complete=milestone.task_complete,
            z_t=z_t,
            z_hat=prediction.z_hat,
            sigma=prediction.sigma,
            sigma_bar=sigma_bar,
            conditioning=e_t,
            decision_mask=decision_mask,
            num_sequential_evaluations=len(self._chain_log),
        )

    def _commit_prefix(self, chunk: Tensor, executed_horizon: Tensor, mask: Tensor) -> None:
        """Replace the execution prefix of masked elements: chunk ``[B, H, d_a]``, H_t ``[B]``."""
        zeros = torch.zeros_like(executed_horizon)
        if self._chunk is None or self._executed_horizon is None or self._cursor is None:
            self._chunk, self._executed_horizon, self._cursor = chunk.clone(), executed_horizon.clone(), zeros
            return
        self._chunk = torch.where(mask.view(-1, 1, 1), chunk, self._chunk)
        self._executed_horizon = torch.where(mask, executed_horizon, self._executed_horizon)
        self._cursor = torch.where(mask, zeros, self._cursor)

    # ------------------------------------------------------------------ control-rate interface
    @torch.no_grad()
    def act(self, observation: Tensor, proprioception: Tensor) -> ActOutput:
        """Return the action to execute at the current control step.

        Elements whose executed prefix is exhausted (cursor >= H_t), or that have no chunk yet,
        trigger a decision step; the others continue open-loop with their current chunk.

        Args:
            observation: ``[B, C, H_img, W_img]``;  proprioception: ``[B, d_s]``.

        Returns:
            ``ActOutput(action [B, d_a], replanned [B] bool, step_output or None)``.
        """
        b = observation.shape[0]
        if self._chunk is None or self._executed_horizon is None or self._cursor is None:
            need = torch.ones(b, dtype=torch.bool, device=observation.device)
        else:
            need = self._cursor >= self._executed_horizon
        step_output = self.step(observation, proprioception, decision_mask=need) if bool(need.any()) else None
        assert self._chunk is not None and self._cursor is not None
        rows = torch.arange(b, device=self._chunk.device)
        action = self._chunk[rows, self._cursor]                                     # [B, d_a]
        self._cursor = self._cursor + 1
        return ActOutput(action, need, step_output)

    # ------------------------------------------------------------------ state accessors
    @property
    def milestone_pointer(self) -> Optional[Tensor]:
        """Active milestone m_t ``[B]`` (1-based)."""
        return self.tracker.m

    @property
    def task_complete(self) -> Optional[Tensor]:
        """``[B]`` bool: Eq. 9a test satisfied with m_t = M (episode should terminate)."""
        return self.tracker.task_complete

    @property
    def execution_prefix(self) -> Optional[Tuple[Tensor, Tensor, Tensor]]:
        """Current (chunk ``[B, H, d_a]``, H_t ``[B]``, cursor ``[B]``), or None before the first step."""
        if self._chunk is None or self._executed_horizon is None or self._cursor is None:
            return None
        return self._chunk, self._executed_horizon, self._cursor


# ---------------------------------------------------------------------------------------------
# Self-test (run: python pipeline.py)
# ---------------------------------------------------------------------------------------------
_GAMMA_STAR = 1.25  # exactly representable in float32


def _calibration_stats() -> Tuple[Tensor, Tensor, Tensor]:
    """Fixed non-trivial (mu_Z [16], mu_S [6], sigma_S [6]) used by every tiny pipeline."""
    g = torch.Generator().manual_seed(5)
    return torch.randn(16, generator=g), torch.randn(6, generator=g), torch.rand(6, generator=g) + 0.5


def _tiny_pipeline(
    horizon: int, k_steps: int, kappa: float = 0.01, beta: float = 0.5, calibrate: bool = True
) -> VPAInferencePipeline:
    try:
        from .perception import _ToyTextModel, _toy_tokenizer
    except ImportError:
        from perception import _ToyTextModel, _toy_tokenizer

    cfg = VPAConfig(
        image_size=(32, 32), patch_size=8, vit_width=32, vit_depth=2, vit_heads=4, latent_dim=16,
        num_primitives=4, selector_hidden=(32,), kappa=kappa, alpha_gamma=0.5,
        ensemble_heads=3, predictor_embed_dim=8, predictor_hidden=32, predictor_hidden_layers=1,
        proprio_dim=6, action_dim=3, primitive_embed_dim=8, horizon=horizon, min_horizon=2,
        integration_steps=k_steps, beta=beta, alpha_sigma=0.2,
        field_width=32, field_depth=2, field_heads=4, time_embed_dim=16,
    )
    text = TextEncoderWrapper(text_model=_ToyTextModel(projection_dim=12), tokenizer=_toy_tokenizer)
    pipe = VPAInferencePipeline.from_config(cfg, text_encoder=text)
    if calibrate:
        mu_z, mu_s, sd_s = _calibration_stats()
        pipe.calibrate(_GAMMA_STAR, mu_z, mu_s, sd_s)
    return pipe


def _expect(exc: type, fn: Callable[[], Any]) -> None:
    try:
        fn()
    except exc:
        return
    raise AssertionError(f"expected {exc.__name__}")


def _record_chain(pipe: VPAInferencePipeline) -> List[str]:
    """Attach test-owned forward hooks (independent of the pipeline's audit) and return their log."""
    log: List[str] = []
    for name, module in (("E_psi", pipe.vision_encoder), ("pi_phi_h", pipe.selector),
                         ("P_omega", pipe.predictor), ("v_theta", pipe.solver.vector_field)):
        module.register_forward_hook(lambda *_args, n=name: log.append(n))
    return log


def _ref_horizon(pipe: VPAInferencePipeline, sigma_bar: Tensor) -> List[int]:
    """Eq. 15 in float64, with a guard that float32 rounding cannot move the floor."""
    f = pipe.horizon_filter
    out = []
    for s in sigma_bar.tolist():
        x = f.h_max * math.exp(-f.beta * s)
        assert abs(x - round(x)) > 1e-4, x
        out.append(max(f.h_min, math.floor(x)))
    return out


def _snapshot(pipe: VPAInferencePipeline) -> List[Tensor]:
    prefix = pipe.execution_prefix
    assert prefix is not None and pipe.tracker.m is not None and pipe.horizon_filter.sigma_bar is not None
    return [pipe.tracker.m.clone(), pipe.tracker.task_complete.clone(), pipe.horizon_filter.sigma_bar.clone()] + [
        t.clone() for t in prefix]


def _self_test() -> None:
    torch.manual_seed(0)
    gen = torch.Generator().manual_seed(77)
    b, m, c, h_img, ds, da = 3, 2, 3, 32, 6, 3
    texts = ["move the cup", "throw the cup away", "push the block"]

    def frames(*shape: int) -> Tensor:
        return torch.randn(*shape, c, h_img, h_img, generator=gen)

    def proprio() -> Tensor:
        return torch.randn(b, ds, generator=gen)

    # ---- Prop. 5.1: depth K + 3 and call order, for every K and H.
    for k_steps in (1, 2, 3):
        for horizon in (4, 16, 64):
            pipe = _tiny_pipeline(horizon, k_steps)
            log = _record_chain(pipe)
            pipe.reset(texts, frames(b, m))
            log.clear()  # milestone encoding in reset() is per-episode, not on the per-step chain
            out = pipe.step(frames(b), proprio())
            assert log == ["E_psi", "pi_phi_h", "P_omega"] + ["v_theta"] * k_steps, log
            assert out.num_sequential_evaluations == k_steps + 3
            assert out.action_chunk.shape == (b, horizon, da) and out.executed_mask.shape == (b, horizon)
            assert out.conditioning.shape == (b, pipe.solver.cond_dim)

    # ---- Calibration: one gamma* in tracker, predictor and solver; errors before calibrate/reset.
    raw = _tiny_pipeline(16, 2, calibrate=False)
    _expect(RuntimeError, lambda: raw.reset(texts, frames(b, m)))
    _expect(RuntimeError, lambda: raw.step(frames(b), proprio()))
    mu_z, mu_s, sd_s = _calibration_stats()
    tau_star = raw.calibrate(_GAMMA_STAR, mu_z, mu_s, sd_s)
    assert raw.tracker.gamma_bar_star == _GAMMA_STAR
    assert float(raw.predictor.latent_scale) == _GAMMA_STAR and float(raw.solver.gamma_bar_star) == _GAMMA_STAR
    assert math.isclose(tau_star, raw.tracker.kappa * math.sqrt(2 * 16) * _GAMMA_STAR, rel_tol=1e-12)
    assert not any(p.requires_grad for p in raw.vision_encoder.parameters())
    assert not any(p.requires_grad for p in raw.predictor.parameters()) and not raw.training
    _expect(RuntimeError, lambda: raw.step(frames(b), proprio()))         # calibrated, but no reset()
    raw.reset(texts, frames(b, m))
    _expect(ValueError, lambda: raw.step(frames(b), proprio(), decision_mask=torch.tensor([True, False, True])))
    _expect(ValueError, lambda: raw.step(frames(b + 1), torch.randn(b + 1, ds)))
    _expect(ValueError, lambda: raw.step(frames(b), torch.randn(b, ds + 1)))
    raw.predictor.set_latent_scale(2.0 * _GAMMA_STAR)                    # gamma* no longer shared
    _expect(RuntimeError, lambda: raw.step(frames(b), proprio()))

    # ---- Fig. 2 data flow, Eq. 12, Eq. 15, Eq. 9a: step() == a manual run of the same modules.
    pipe = _tiny_pipeline(16, 2)
    obs = frames(b)
    goals = frames(b, m)
    goals[:, 0] = obs                     # milestone 1 is reached at the first step
    pipe.reset(texts, goals)
    du, d = pipe.solver.primitive_embed_dim, pipe.solver.latent_dim
    with torch.no_grad():
        c_text = pipe.text_encoder(texts)
        z_goals = pipe.vision_encoder.encode_milestones(goals)
    assert torch.equal(pipe.c_text, c_text) and torch.equal(pipe.milestone_latents, z_goals)
    for step_idx in range(2):             # second step runs at m_t = 2
        obs_t = obs if step_idx == 0 else frames(b)
        s_t, xi = proprio(), torch.randn(b, 16, da, generator=gen)
        m_t = pipe.tracker.m.clone()
        out = pipe.step(obs_t, s_t, noise=xi)
        assert torch.equal(out.milestone_pointer, m_t)
        with torch.no_grad():
            z_t = pipe.vision_encoder(obs_t)
            z_g = torch.stack([z_goals[i, int(m_t[i]) - 1] for i in range(b)])   # z_g^(m_t)
            u_t = pipe.selector.select(z_t, z_g, c_text)
            pred = pipe.predictor(z_t, u_t)
            e_t = pipe.solver.build_conditioning(u_t, z_t, pred.z_hat, s_t, c_text)
            chunk = pipe.solver.generate_chunk(e_t, noise=xi)
        assert torch.equal(out.z_t, z_t) and torch.equal(out.primitive, u_t)
        assert torch.equal(out.z_hat, pred.z_hat) and torch.equal(out.sigma, pred.sigma)
        assert torch.equal(out.conditioning, e_t) and torch.equal(out.action_chunk, chunk)
        # Eq. 12 inside the pipeline: standardised z_t slot and raw c_text slot.
        z_ref = (out.z_t.double() - mu_z.double()) / _GAMMA_STAR
        assert torch.allclose(out.conditioning[:, du:du + d].double(), z_ref, rtol=1e-5, atol=1e-6)
        assert torch.equal(out.conditioning[:, -c_text.shape[1]:], c_text)
        # Eq. 15: first step sigma_bar = sigma; H_t vs the float64 floor; executed mask.
        if step_idx == 0:
            assert torch.equal(out.sigma_bar, out.sigma)
        assert out.executed_horizon.tolist() == _ref_horizon(pipe, out.sigma_bar)
        assert out.executed_mask.tolist() == [[hh < int(out.executed_horizon[i]) for hh in range(16)]
                                              for i in range(b)]
        if step_idx == 0:                 # Eq. 9a: identical frame -> distance ~ 0 < tau*, m: 1 -> 2
            assert bool((out.milestone_distance < 1e-5).all())
            assert out.next_milestone_pointer.tolist() == [2] * b and not bool(out.task_complete.any())

    # ---- Eq. 9a: M = 1 completes the task without moving the pointer.
    pipe.reset("pick up the block", obs.unsqueeze(1))
    out = pipe.step(obs, proprio())
    assert out.next_milestone_pointer.tolist() == [1] * b and bool(out.task_complete.all())

    # ---- Eq. 9a long rollout against a float64 reference of the pointer rule.
    num_m = 3
    pipe = _tiny_pipeline(16, 1)
    tau = pipe.tracker.threshold
    goals = frames(b, num_m)
    pipe.reset(texts, goals)
    done_prev = [False] * b
    for _ in range(60):
        m_t = pipe.tracker.m.clone()
        obs_t = frames(b)
        hit = (torch.rand(b, generator=gen) < 0.3).tolist()
        for i in range(b):
            if hit[i]:
                obs_t[i] = goals[i, int(m_t[i]) - 1]
        out = pipe.step(obs_t, proprio())
        for i in range(b):
            g = pipe.milestone_latents[i, int(m_t[i]) - 1].double()
            dist = math.sqrt(float(((out.z_t[i].double() - g) ** 2).sum()))
            assert abs(dist - tau) > 1e-5                     # unambiguous side of the strict test
            reached = dist < tau
            want = int(m_t[i]) + int(reached and int(m_t[i]) < num_m)
            assert int(out.next_milestone_pointer[i]) == want
            assert 1 <= int(m_t[i]) <= want <= num_m       # never decreases, stays in range
            done_i = done_prev[i] or (reached and int(m_t[i]) == num_m)
            assert bool(out.task_complete[i]) == done_i     # sticky completion
            done_prev[i] = done_i
    assert all(done_prev) and pipe.tracker.m.tolist() == [num_m] * b

    # ---- Execution prefix: per-element replanning in act() against an independent model.
    pipe = _tiny_pipeline(16, 2, beta=5.0)
    with torch.no_grad():  # untrained toy latents are nearly identical across frames; spread them so
        pipe.vision_encoder.head.weight.mul_(10.0)  # sigma, and hence H_t, differs between elements
    pipe.reset(texts, frames(b, m))
    chunk_m = horizon_m = cursor_m = sbar_m = None
    mixed = 0
    for call in range(60):
        prev_prefix = pipe.execution_prefix
        res = pipe.act(frames(b), proprio())
        expect = torch.ones(b, dtype=torch.bool) if cursor_m is None else cursor_m >= horizon_m
        assert torch.equal(res.replanned, expect)
        mixed += int(bool(expect.any()) and not bool(expect.all()))
        if call == 0:
            so = res.step_output
            chunk_m, horizon_m = so.action_chunk.clone(), so.executed_horizon.clone()
            sbar_m, cursor_m = so.sigma_bar.clone(), torch.zeros(b, dtype=torch.long)
        elif res.step_output is not None:
            so = res.step_output
            for i in range(b):
                if expect[i]:             # replanned: new chunk, H_t, sigma_bar; cursor restarts
                    chunk_m[i], horizon_m[i], sbar_m[i], cursor_m[i] = (
                        so.action_chunk[i], so.executed_horizon[i], so.sigma_bar[i], 0)
                else:                     # not replanned: everything kept bit-exactly
                    assert torch.equal(pipe.execution_prefix[0][i], prev_prefix[0][i])
                    assert int(pipe.execution_prefix[1][i]) == int(prev_prefix[1][i])
                    assert float(so.sigma_bar[i]) == float(sbar_m[i])
                    assert int(so.next_milestone_pointer[i]) == int(so.milestone_pointer[i])
        else:
            assert res.step_output is None
        assert torch.equal(pipe.execution_prefix[0], chunk_m) and torch.equal(pipe.execution_prefix[1], horizon_m)
        assert torch.equal(res.action, chunk_m[torch.arange(b), cursor_m])
        assert bool(((horizon_m >= 2) & (horizon_m <= 16)).all())
        cursor_m = cursor_m + 1
    assert mixed >= 5, mixed

    # ---- Guard: hidden or missing network evaluations raise, and nothing is committed.
    pipe = _tiny_pipeline(16, 2)
    pipe.reset(texts, frames(b, m))
    pipe.step(frames(b), proprio())
    solver, selector = pipe.solver, pipe.selector
    gen_chunk, select = solver.generate_chunk, selector.select

    def extra_v(cond: Tensor, noise: Optional[Tensor] = None) -> Tensor:
        solver.vector_field(torch.zeros(cond.shape[0], 16, da), 0.0, cond)
        return gen_chunk(cond, noise=noise)

    def missing_v(cond: Tensor, noise: Optional[Tensor] = None) -> Tensor:
        a = torch.zeros(cond.shape[0], 16, da)
        return a + solver.vector_field(a, 0.0, cond)        # K - 1 = 1 evaluation instead of 2

    def extra_e(z_t: Tensor, z_goal: Tensor, c_txt: Tensor) -> Tensor:
        pipe.vision_encoder(torch.zeros(z_t.shape[0], c, h_img, h_img))
        return select(z_t, z_goal, c_txt)

    for target, attr, patch in ((solver, "generate_chunk", extra_v), (solver, "generate_chunk", missing_v),
                                (selector, "select", extra_e)):
        before = _snapshot(pipe)
        setattr(target, attr, patch)
        try:
            _expect(RuntimeError, lambda: pipe.step(frames(b), proprio()))
        finally:
            delattr(target, attr)  # restore the class method
        assert all(torch.equal(x, y) for x, y in zip(before, _snapshot(pipe), strict=True))
    assert pipe.step(frames(b), proprio()).num_sequential_evaluations == 2 + 3  # restored

    # ---- Eqs. 9b / 18 share one eps.
    loss_fn = pipe.make_vicreg_loss(lambda_v=25.0, lambda_c=1.0, head_reduction="sum")
    assert loss_fn.eps == pipe.tracker.eps and loss_fn.head_reduction == "sum"
    pipe.check_vicreg_loss(loss_fn)
    _expect(ValueError, lambda: pipe.check_vicreg_loss(
        VICRegLoss(lambda_v=25.0, lambda_c=1.0, eps=10.0 * pipe.tracker.eps)))
    print("pipeline.py self-test passed")

if __name__ == "__main__":
    _self_test()
