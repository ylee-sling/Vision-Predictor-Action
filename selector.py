"""
selector.py -- Stage 2 of the Goal-Conditioned VPA framework (preprint_261008.pdf, Sec. 4.2).

Implements
    * ``NeuroSymbolicSelector`` -- the MLP classifier pi_phi^h over the primitive dictionary
      U = {u^(1), ..., u^(N_u)} (Eq. 8):
          u_t = argmax_{u in U} pi_phi^h(u | z_t, z_g^(m_t), c_text),
      trained with cross-entropy against demonstration primitive labels.
    * ``MilestoneTracker``      -- the symbolic milestone pointer m_t and its variance-scaled
      completion threshold (Eqs. 9a-9c):
          m_{t+1}   = m_t + 1   if ||z_t - z_g^(m_t)||_2 < tau and m_t < M,   else m_t      (9a)
          gamma_n   = (1 - a_g) gamma_{n-1} + a_g * sqrt( (1/d) sum_j Var(Z^(n)_{.,j}) + eps )  (9b)
          tau_n     = kappa * sqrt(2d) * gamma_n                                            (9c)
      Once E_psi is frozen, gamma is fixed to gamma* and deployment uses tau* = kappa sqrt(2d) gamma*.

Indexing convention: the pointer m_t is stored 1-based exactly as in the paper (m_t in {1..M});
it is converted to a 0-based index only when gathering z_g^(m_t) from a ``[B, M, d]`` tensor.

This file is self-contained (it imports nothing from the other VPA modules) and can be tested
on its own with ``python selector.py``.
"""

from __future__ import annotations

import math
from typing import NamedTuple, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

__all__ = ["NeuroSymbolicSelector", "MilestoneTracker", "MilestoneUpdate"]


# ---------------------------------------------------------------------------------------------
# pi_phi^h : neuro-symbolic primitive selector (Eq. 8)
# ---------------------------------------------------------------------------------------------
class NeuroSymbolicSelector(nn.Module):
    """MLP classifier pi_phi^h mapping Concat(z_t, z_g^(m_t), c_text) to logits over U (Eq. 8).

    The per-decision cost is a single MLP evaluation, independent of the number of milestones M,
    because only the *active* milestone latent z_g^(m_t) is fed to the network.

    Args:
        latent_dim: d, dimension of z_t and z_g^(m).
        text_dim: d_c, dimension of c_text.
        num_primitives: N_u = |U|.
        hidden_dims: widths of the hidden layers of the MLP.

    Shapes:
        ``forward`` / ``probabilities``: (``[B, d]``, ``[B, d]``, ``[B, d_c]``) -> ``[B, N_u]``
        ``select``:                      (``[B, d]``, ``[B, d]``, ``[B, d_c]``) -> ``[B]`` (int64)
    """

    def __init__(
        self,
        latent_dim: int,
        text_dim: int,
        num_primitives: int,
        hidden_dims: Sequence[int] = (512, 512),
    ) -> None:
        super().__init__()
        if num_primitives < 2:
            raise ValueError("the primitive dictionary must contain at least two primitives")
        self.latent_dim: int = latent_dim
        self.text_dim: int = text_dim
        self.num_primitives: int = num_primitives
        layers = []
        in_features = 2 * latent_dim + text_dim
        for width in hidden_dims:
            layers += [nn.Linear(in_features, width), nn.GELU()]
            in_features = width
        layers.append(nn.Linear(in_features, num_primitives))
        self.mlp = nn.Sequential(*layers)

    def forward(self, z_t: Tensor, z_goal: Tensor, c_text: Tensor) -> Tensor:
        """Unnormalised scores of pi_phi^h(u | z_t, z_g^(m_t), c_text).

        Args:
            z_t:    ``[B, d]``   current latent.
            z_goal: ``[B, d]``   active milestone latent z_g^(m_t).
            c_text: ``[B, d_c]`` instruction embedding.

        Returns:
            logits: ``[B, N_u]``.
        """
        if z_t.shape != z_goal.shape or z_t.shape[-1] != self.latent_dim:
            raise ValueError(f"expected z_t and z_goal of shape [B, {self.latent_dim}]")
        if c_text.shape != (z_t.shape[0], self.text_dim):
            raise ValueError(f"expected c_text of shape [B, {self.text_dim}], got {tuple(c_text.shape)}")
        return self.mlp(torch.cat([z_t, z_goal, c_text], dim=-1))

    def probabilities(self, z_t: Tensor, z_goal: Tensor, c_text: Tensor) -> Tensor:
        """Categorical distribution pi_phi^h(. | z_t, z_g^(m_t), c_text): ``[B, N_u]``."""
        return F.softmax(self(z_t, z_goal, c_text), dim=-1)

    def select(self, z_t: Tensor, z_goal: Tensor, c_text: Tensor) -> Tensor:
        """Eq. 8: u_t = argmax_u pi_phi^h(u | ...). Returns int64 tokens ``[B]``.

        Softmax is strictly monotone, so the argmax of the logits equals the argmax of pi_phi^h.
        The network is invoked through ``__call__`` so that forward hooks (used by the pipeline to
        audit the sequential depth) observe exactly one evaluation.
        """
        return self(z_t, z_goal, c_text).argmax(dim=-1)

    @staticmethod
    def loss(logits: Tensor, primitive_labels: Tensor) -> Tensor:
        """Cross-entropy between pi_phi^h and demonstration primitive labels (Sec. 4.2).

        Args:
            logits: ``[B, N_u]``.
            primitive_labels: ``[B]`` int64 in {0, ..., N_u - 1}.

        Returns:
            scalar loss (mean over the batch).
        """
        return F.cross_entropy(logits, primitive_labels)


# ---------------------------------------------------------------------------------------------
# Milestone pointer m_t and variance-scaled threshold tau (Eqs. 9a-9c)
# ---------------------------------------------------------------------------------------------
class MilestoneUpdate(NamedTuple):
    """Result of one application of Eq. 9a (all tensors have leading dimension B)."""

    m_t: Tensor             # [B] int64, pointer used at this decision step (1-based)
    m_next: Tensor          # [B] int64, m_{t+1}
    distance: Tensor        # [B] float, ||z_t - z_g^(m_t)||_2
    reached: Tensor         # [B] bool,  distance < tau
    task_complete: Tensor   # [B] bool,  sticky flag: test satisfied with m_t = M


class MilestoneTracker(nn.Module):
    """Symbolic finite-state machine over milestones with a variance-calibrated threshold.

    Training-time calibration (Eqs. 9b, 9c):
        ``update_variance_margin(Z)`` is called once per training iteration n with the batch
        Z^(n) in R^{B x d} of *online*-encoder latents. Var is the unbiased batch estimator
        (1 / (B - 1)), i.e. the same estimator as the covariance C(Z) of Eq. 19.
        ``threshold`` then returns tau_n = kappa * sqrt(2d) * gamma_n.

        The paper does not specify gamma_0. If ``gamma_bar_init`` is None, gamma_0 is set to the
        first instantaneous estimate (so gamma_1 equals that estimate); otherwise gamma_0 =
        ``gamma_bar_init``.

    Deployment (after E_psi is frozen, Sec. 4.2/4.3):
        ``freeze()`` fixes gamma* (and therefore tau* = kappa sqrt(2d) gamma*). ``gamma_bar`` is a
        registered buffer, so gamma* is saved and restored with ``state_dict``.

    Episode state (not part of ``state_dict``):
        ``m`` -- ``[B]`` int64 pointer in {1, ..., M}, monotonically non-decreasing;
        ``task_complete`` -- ``[B]`` bool, set once Eq. 9a's test holds with m_t = M.

    Args:
        latent_dim: d.
        kappa: dimensionless tolerance kappa in (0, 1).
        alpha_gamma: EMA rate alpha_gamma in (0, 1].
        eps: numerical constant eps of Eq. 18 (must be the same value used by the VICReg loss).
        gamma_bar_init: optional gamma_0.
    """

    def __init__(
        self,
        latent_dim: int,
        kappa: float,
        alpha_gamma: float,
        eps: float,
        gamma_bar_init: Optional[float] = None,
    ) -> None:
        super().__init__()
        if not 0.0 < kappa < 1.0:
            raise ValueError("kappa must lie in (0, 1)")
        if not 0.0 < alpha_gamma <= 1.0:
            raise ValueError("alpha_gamma must lie in (0, 1]")
        if eps <= 0.0:
            raise ValueError("eps must be positive")
        self.latent_dim: int = int(latent_dim)
        self.kappa: float = float(kappa)
        self.alpha_gamma: float = float(alpha_gamma)
        self.eps: float = float(eps)
        init = float("nan") if gamma_bar_init is None else float(gamma_bar_init)
        if gamma_bar_init is not None and gamma_bar_init <= 0.0:
            raise ValueError("gamma_bar_init must be positive")
        # float32 on purpose: the Apple MPS backend has no float64, and the buffer must move with .to("mps").
        self.register_buffer("gamma_bar", torch.tensor(init, dtype=torch.float32))
        self.register_buffer("frozen", torch.tensor(False))
        # Episode state (runtime only).
        self.m: Optional[Tensor] = None
        self.task_complete: Optional[Tensor] = None
        self.num_milestones: Optional[int] = None

    # ---------------- threshold calibration (Eqs. 9b, 9c) ----------------
    def instantaneous_margin(self, latents: Tensor) -> Tensor:
        """sqrt( (1/d) * sum_j Var(Z_{.,j}) + eps ) for a batch ``Z: [B, d]`` -> scalar tensor."""
        if latents.dim() != 2 or latents.shape[1] != self.latent_dim:
            raise ValueError(f"expected latents of shape [B, {self.latent_dim}], got {tuple(latents.shape)}")
        if latents.shape[0] < 2:
            raise ValueError("the unbiased variance estimator requires B >= 2")
        var = latents.detach().float().var(dim=0, unbiased=True)  # [d] (float32: MPS-safe)
        return torch.sqrt(var.mean() + self.eps)

    @torch.no_grad()
    def update_variance_margin(self, latents: Tensor) -> float:
        """Apply Eq. 9b with the online-encoder batch ``Z^(n): [B, d]``; returns gamma_n."""
        if bool(self.frozen):
            raise RuntimeError("gamma is frozen (E_psi has been frozen); it can no longer be updated")
        inst = self.instantaneous_margin(latents).to(self.gamma_bar.device)
        if torch.isnan(self.gamma_bar):
            self.gamma_bar.copy_(inst)  # gamma_0 := first estimate
        self.gamma_bar.copy_((1.0 - self.alpha_gamma) * self.gamma_bar + self.alpha_gamma * inst)
        return float(self.gamma_bar)

    @property
    def is_calibrated(self) -> bool:
        return not bool(torch.isnan(self.gamma_bar))

    @property
    def threshold(self) -> float:
        """tau = kappa * sqrt(2d) * gamma  (tau_n during training, tau* once frozen; Eq. 9c)."""
        if not self.is_calibrated:
            raise RuntimeError("gamma has not been estimated yet; call update_variance_margin or freeze")
        return self.kappa * math.sqrt(2.0 * self.latent_dim) * float(self.gamma_bar)

    @torch.no_grad()
    def freeze(self, gamma_bar_star: Optional[float] = None) -> float:
        """Fix gamma* (optionally overriding it) for deployment; returns tau*."""
        if gamma_bar_star is not None:
            if gamma_bar_star <= 0.0:
                raise ValueError("gamma_bar_star must be positive")
            self.gamma_bar.fill_(float(gamma_bar_star))
        if not self.is_calibrated:
            raise RuntimeError("cannot freeze an uncalibrated tracker; provide gamma_bar_star")
        self.frozen.fill_(True)
        return self.threshold

    @property
    def gamma_bar_star(self) -> float:
        """The frozen latent scale gamma* (Eq. 9b after freezing)."""
        if not bool(self.frozen):
            raise RuntimeError("gamma* is only defined after freeze()")
        return float(self.gamma_bar)

    # ---------------- pointer logic (Eq. 9a) ----------------
    def reset(self, num_milestones: int, batch_size: int, device: Optional[torch.device] = None) -> None:
        """Start an episode: m_0 = 1 for every batch element."""
        if num_milestones < 1:
            raise ValueError("M must be at least 1")
        self.num_milestones = int(num_milestones)
        self.m = torch.ones(batch_size, dtype=torch.long, device=device)
        self.task_complete = torch.zeros(batch_size, dtype=torch.bool, device=device)

    def _require_episode(self) -> Tuple[Tensor, Tensor, int]:
        if self.m is None or self.task_complete is None or self.num_milestones is None:
            raise RuntimeError("call reset() before using the milestone pointer")
        return self.m, self.task_complete, self.num_milestones

    def active_goal(self, milestone_latents: Tensor) -> Tensor:
        """Gather z_g^(m_t) from ``[B, M, d]`` -> ``[B, d]``."""
        m, _, num_m = self._require_episode()
        if milestone_latents.dim() != 3 or milestone_latents.shape[1] != num_m:
            raise ValueError(f"expected milestone latents of shape [B, {num_m}, d]")
        index = (m - 1).view(-1, 1, 1).expand(-1, 1, milestone_latents.shape[-1])  # [B, 1, d]
        return milestone_latents.gather(1, index).squeeze(1)  # [B, d]

    def completion_test(self, z_t: Tensor, milestone_latents: Tensor) -> Tuple[Tensor, Tensor]:
        """Threshold predicate ||z_t - z_g^(m_t)||_2 < tau.

        Args:
            z_t: ``[B, d]``;  milestone_latents: ``[B, M, d]``.

        Returns:
            distance ``[B]`` and reached ``[B]`` (bool).
        """
        distance = torch.linalg.vector_norm(z_t - self.active_goal(milestone_latents), ord=2, dim=-1)
        return distance, distance < self.threshold

    def step(
        self,
        z_t: Tensor,
        milestone_latents: Tensor,
        update_mask: Optional[Tensor] = None,
    ) -> MilestoneUpdate:
        """Apply Eq. 9a and commit m_{t+1} as the new pointer.

        Args:
            z_t: ``[B, d]`` current latent.
            milestone_latents: ``[B, M, d]``.
            update_mask: optional ``[B]`` bool; only elements that are at a decision step are
                updated (the others keep m_t and their completion flag).

        Returns:
            ``MilestoneUpdate`` with m_t, m_{t+1}, distance, reached and the task-complete flag.
        """
        m, done, num_m = self._require_episode()
        distance, reached = self.completion_test(z_t, milestone_latents)
        advance = reached & (m < num_m)
        finished = reached & (m == num_m)
        if update_mask is not None:
            update_mask = update_mask.to(device=m.device, dtype=torch.bool)
            advance = advance & update_mask
            finished = finished & update_mask
        m_t = m.clone()
        self.m = m + advance.long()
        self.task_complete = done | finished
        return MilestoneUpdate(m_t, self.m.clone(), distance, reached, self.task_complete.clone())


# ---------------------------------------------------------------------------------------------
# Self-test (run: python selector.py)
# ---------------------------------------------------------------------------------------------
def _self_test() -> None:
    torch.manual_seed(0)
    b, d, dc, nu, num_m = 4, 8, 6, 5, 3

    sel = NeuroSymbolicSelector(d, dc, nu, hidden_dims=(32, 32))
    z_t, z_g, c = torch.randn(b, d), torch.randn(b, d), torch.randn(b, dc)
    logits = sel(z_t, z_g, c)
    assert logits.shape == (b, nu)
    u = sel.select(z_t, z_g, c)
    assert u.shape == (b,) and u.dtype == torch.long
    assert torch.equal(u, sel.probabilities(z_t, z_g, c).argmax(-1))
    loss = sel.loss(logits, torch.randint(0, nu, (b,)))
    loss.backward()

    # Eq. 9b / 9c against a direct computation.
    tracker = MilestoneTracker(d, kappa=0.2, alpha_gamma=0.5, eps=1e-4)
    z1, z2 = torch.randn(64, d) * 2.0, torch.randn(64, d) * 3.0
    inst1 = math.sqrt(z1.double().var(0, unbiased=True).mean().item() + 1e-4)
    inst2 = math.sqrt(z2.double().var(0, unbiased=True).mean().item() + 1e-4)
    # Module state is float32 (MPS has no float64); the float64 reference is compared at float32 precision.
    assert math.isclose(tracker.update_variance_margin(z1), inst1, rel_tol=1e-5)
    expected = 0.5 * inst1 + 0.5 * inst2
    assert math.isclose(tracker.update_variance_margin(z2), expected, rel_tol=1e-5)
    assert math.isclose(tracker.threshold, 0.2 * math.sqrt(2 * d) * expected, rel_tol=1e-5)
    tau_star = tracker.freeze()
    try:
        tracker.update_variance_margin(z1)
        raise AssertionError("frozen tracker must not update")
    except RuntimeError:
        pass

    # Eq. 9a: advance only when reached and m < M; completion only at m = M; monotone pointer.
    goals = torch.randn(b, num_m, d) * 10.0
    tracker.reset(num_m, b)
    near_first = goals[:, 0] + 1e-3          # reaches milestone 1 for every element
    upd = tracker.step(near_first, goals)
    assert torch.equal(upd.m_t, torch.ones(b, dtype=torch.long))
    assert torch.equal(upd.m_next, torch.full((b,), 2, dtype=torch.long))
    far = goals[:, 1] + 1e3                  # far from milestone 2: pointer holds
    assert torch.equal(tracker.step(far, goals).m_next, torch.full((b,), 2, dtype=torch.long))
    mask = torch.tensor([True, False, True, False])
    upd = tracker.step(goals[:, 1], goals, update_mask=mask)  # reached; only masked elements move
    assert torch.equal(upd.m_next, torch.tensor([3, 2, 3, 2]))
    upd = tracker.step(goals[:, 2], goals)   # elements 0,2 finish at m = M; 1,3 still on milestone 2
    assert torch.equal(upd.m_next, torch.tensor([3, 2, 3, 2]))
    assert torch.equal(upd.task_complete, torch.tensor([True, False, True, False]))
    assert upd.distance.shape == (b,) and tau_star > 0
    print("selector.py self-test passed")


if __name__ == "__main__":
    _self_test()
