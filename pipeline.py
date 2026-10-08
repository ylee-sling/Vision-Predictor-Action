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
    from .predictor import JEPAPredictor
    from .selector import MilestoneTracker, NeuroSymbolicSelector
    from .solver import FlowMatchingSolver, UncertaintyHorizonFilter, executed_prefix_mask
except ImportError:  # flat import (files side by side, `python pipeline.py`)
    from perception import TextEncoderWrapper, VisionEncoder
    from predictor import JEPAPredictor
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
def _tiny_pipeline(horizon: int, k_steps: int) -> VPAInferencePipeline:
    try:
        from .perception import _ToyTextModel, _toy_tokenizer
    except ImportError:
        from perception import _ToyTextModel, _toy_tokenizer

    cfg = VPAConfig(
        image_size=(32, 32), patch_size=8, vit_width=32, vit_depth=2, vit_heads=4, latent_dim=16,
        num_primitives=4, selector_hidden=(32,), kappa=0.2, alpha_gamma=0.5,
        ensemble_heads=3, predictor_embed_dim=8, predictor_hidden=32, predictor_hidden_layers=1,
        proprio_dim=6, action_dim=3, primitive_embed_dim=8, horizon=horizon, min_horizon=2,
        integration_steps=k_steps, beta=0.5, alpha_sigma=0.2,
        field_width=32, field_depth=2, field_heads=4, time_embed_dim=16,
    )
    text = TextEncoderWrapper(text_model=_ToyTextModel(projection_dim=12), tokenizer=_toy_tokenizer)
    pipe = VPAInferencePipeline.from_config(cfg, text_encoder=text)
    pipe.calibrate(1.0, torch.zeros(16), torch.zeros(6), torch.ones(6))
    return pipe


def _self_test() -> None:
    torch.manual_seed(0)
    b, m, c, h_img = 2, 2, 3, 32

    # Depth is K + 3 for every K and does not change with H.
    for k_steps in (1, 2, 3):
        for horizon in (4, 32):
            pipe = _tiny_pipeline(horizon, k_steps)
            obs = torch.randn(b, c, h_img, h_img)
            pipe.reset(["move the cup", "throw the cup away"], torch.randn(b, m, c, h_img, h_img))
            out = pipe.step(obs, torch.randn(b, 6))
            assert out.num_sequential_evaluations == k_steps + 3
            assert out.action_chunk.shape == (b, horizon, 3)
            assert bool(((out.executed_horizon >= 2) & (out.executed_horizon <= horizon)).all())
            assert out.conditioning.shape == (b, pipe.solver.cond_dim)

    # Milestone pointer: a milestone frame identical to the observation is reached (distance 0).
    pipe = _tiny_pipeline(8, 2)
    obs = torch.randn(b, c, h_img, h_img)
    goals = torch.randn(b, m, c, h_img, h_img)
    goals[:, 0] = obs
    pipe.reset("pick up the block", goals)
    out = pipe.step(obs, torch.randn(b, 6))
    assert torch.equal(out.milestone_pointer, torch.ones(b, dtype=torch.long))
    assert torch.equal(out.next_milestone_pointer, torch.full((b,), 2, dtype=torch.long))
    assert not bool(out.task_complete.any())
    pipe.reset("pick up the block", obs.unsqueeze(1))  # M = 1: reaching it completes the task
    assert bool(pipe.step(obs, torch.randn(b, 6)).task_complete.all())

    # Execution prefix: replanning happens exactly when the cursor reaches H_t.
    pipe = _tiny_pipeline(8, 2)
    pipe.reset("push the block", torch.randn(b, m, c, h_img, h_img))
    cursor, horizon_t, chunk = None, None, None
    for _ in range(40):
        result = pipe.act(torch.randn(b, c, h_img, h_img), torch.randn(b, 6))
        expected = torch.ones(b, dtype=torch.bool) if cursor is None else cursor >= horizon_t
        assert torch.equal(result.replanned, expected)
        if result.step_output is not None:
            new = result.step_output
            chunk = new.action_chunk if chunk is None else torch.where(expected.view(-1, 1, 1), new.action_chunk, chunk)
            horizon_t = new.executed_horizon if horizon_t is None else torch.where(expected, new.executed_horizon, horizon_t)
            cursor = torch.zeros(b, dtype=torch.long) if cursor is None else torch.where(expected, torch.zeros_like(cursor), cursor)
        assert torch.allclose(result.action, chunk[torch.arange(b), cursor])
        cursor = cursor + 1

    # A hidden extra network evaluation on the chain is detected.
    original = pipe.solver.generate_chunk

    def leaky(cond: Tensor, noise: Optional[Tensor] = None) -> Tensor:
        pipe.solver.vector_field(torch.zeros(cond.shape[0], 8, 3), 0.0, cond)
        return original(cond, noise=noise)

    pipe.solver.generate_chunk = leaky  # type: ignore[method-assign]
    try:
        pipe.step(torch.randn(b, c, h_img, h_img), torch.randn(b, 6))
        raise AssertionError("an extra v_theta evaluation must be rejected")
    except RuntimeError:
        pass
    print("pipeline.py self-test passed")


if __name__ == "__main__":
    _self_test()
