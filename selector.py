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
_RTOL, _ATOL = 1e-5, 1e-6  # float32 module output vs float64 reference (CLAUDE.md)


def _close(actual: Tensor, ref: Tensor) -> bool:
    return torch.allclose(actual.double(), ref.double(), rtol=_RTOL, atol=_ATOL)


def _close_scalar(actual: float, ref: float) -> bool:
    return abs(actual - ref) <= _ATOL + _RTOL * abs(ref)


def _expect(exc: type, fn) -> None:
    try:
        fn()
    except exc:
        return
    raise AssertionError(f"expected {exc.__name__}")


def _ref_selector_logits(sel: NeuroSymbolicSelector, z_t: Tensor, z_g: Tensor, c: Tensor) -> Tensor:
    """Float64 reference of the Eq. 8 MLP: input slots written by hand, Linear/exact-GELU by hand."""
    b, d, dc = z_t.shape[0], z_t.shape[1], c.shape[1]
    x = torch.zeros(b, 2 * d + dc, dtype=torch.float64)
    x[:, :d] = z_t.double()
    x[:, d:2 * d] = z_g.double()
    x[:, 2 * d:] = c.double()
    for layer in sel.mlp:
        if isinstance(layer, nn.Linear):
            x = x @ layer.weight.detach().double().T + layer.bias.detach().double()
        else:
            assert isinstance(layer, nn.GELU) and layer.approximate == "none"
            x = 0.5 * x * (1.0 + torch.erf(x / math.sqrt(2.0)))
    return x  # [B, N_u]


def _ref_softmax(s: Tensor) -> Tensor:
    e = torch.exp(s - s.max(dim=-1, keepdim=True).values)
    return e / e.sum(dim=-1, keepdim=True)


def _ref_margin(z: Tensor, eps: float) -> float:
    """sqrt((1/d) sum_j Var_unbiased(Z_.j) + eps) with explicit float64 sums (Eq. 9b)."""
    zd = z.double()
    bsz, d = zd.shape
    total = 0.0
    for j in range(d):
        col = [float(zd[i, j]) for i in range(bsz)]
        mean = sum(col) / bsz
        total += sum((v - mean) ** 2 for v in col) / (bsz - 1)
    return math.sqrt(total / d + eps)


def _ref_pointer_step(m, done, z, goals, tau, num_m, mask):
    """Per-element float64 reference of Eq. 9a (lists in, lists out)."""
    m_next, done_next, dist, reached = list(m), list(done), [], []
    for i in range(len(m)):
        g = goals[i, m[i] - 1].double()
        dist_i = math.sqrt(float(((z[i].double() - g) ** 2).sum()))
        hit = dist_i < tau
        dist.append(dist_i)
        reached.append(hit)
        if mask[i]:
            if hit and m[i] < num_m:
                m_next[i] = m[i] + 1
            if hit and m[i] == num_m:
                done_next[i] = True
    return m_next, done_next, dist, reached


def _self_test() -> None:
    torch.manual_seed(0)
    gen = torch.Generator().manual_seed(2024)
    b, d, dc, nu, num_m = 4, 8, 6, 5, 3

    # ---- Eq. 8: u_t = argmax pi_phi^h(u | z_t, z_g^(m_t), c_text).
    sel = NeuroSymbolicSelector(d, dc, nu, hidden_dims=(32, 32))
    z_t, z_g = torch.randn(b, d, generator=gen), torch.randn(b, d, generator=gen)
    c = torch.randn(b, dc, generator=gen)
    captured = []
    pre = sel.mlp.register_forward_pre_hook(lambda _m, args: captured.append(args[0].detach().clone()))
    logits = sel(z_t, z_g, c)
    pre.remove()
    x_in = captured[0]
    assert x_in.shape == (b, 2 * d + dc)
    assert torch.equal(x_in[:, :d], z_t) and torch.equal(x_in[:, d:2 * d], z_g)
    assert torch.equal(x_in[:, 2 * d:], c)
    ref_logits = _ref_selector_logits(sel, z_t, z_g, c)
    ref_probs = _ref_softmax(ref_logits)
    assert logits.shape == (b, nu) and _close(logits, ref_logits)
    probs = sel.probabilities(z_t, z_g, c)
    assert _close(probs, ref_probs) and _close(probs.sum(-1), torch.ones(b))
    top2 = ref_probs.topk(2, dim=-1).values
    assert bool(((top2[:, 0] - top2[:, 1]) > 1e-4).all())  # no near-ties: argmax is well defined
    calls = []
    hook = sel.register_forward_hook(lambda *_: calls.append(1))
    u = sel.select(z_t, z_g, c)
    hook.remove()
    assert len(calls) == 1  # one evaluation through __call__ (depth audit)
    assert u.shape == (b,) and u.dtype == torch.long
    assert torch.equal(u, probs.argmax(-1)) and torch.equal(u, ref_probs.argmax(-1))
    sel.loss(logits, torch.randint(0, nu, (b,), generator=gen)).backward()
    _expect(ValueError, lambda: sel(z_t, z_g[:, :-1], c))
    _expect(ValueError, lambda: sel(z_t[:, :-1], z_g[:, :-1], c))
    _expect(ValueError, lambda: sel(z_t, z_g, c[:, :-1]))
    _expect(ValueError, lambda: sel(z_t, z_g, c[:-1]))
    _expect(ValueError, lambda: NeuroSymbolicSelector(d, dc, 1))

    # ---- Eq. 9b / 9c: EMA of the variance margin and tau = kappa sqrt(2d) gamma.
    kappa, eps = 0.2, 1e-4
    sizes = [2, 3, 5, 16, 64, 2, 7, 32, 4, 128, 9, 2]  # includes the minimum B = 2
    batches = [torch.randn(n, d, generator=gen) * (0.5 + 3.0 * float(torch.rand(1, generator=gen)))
               for n in sizes]
    for alpha, init in ((0.3, None), (0.3, 0.7), (1.0, None)):
        tracker = MilestoneTracker(d, kappa=kappa, alpha_gamma=alpha, eps=eps, gamma_bar_init=init)
        if init is None:
            _expect(RuntimeError, lambda tr=tracker: tr.threshold)      # not calibrated yet
        else:  # gamma_0 = gamma_bar_init already defines tau_0
            assert _close_scalar(tracker.threshold, kappa * math.sqrt(2.0 * d) * init)
        _expect(RuntimeError, lambda tr=tracker: tr.gamma_bar_star)     # not frozen yet
        gamma_ref = init
        for n, z in enumerate(batches):
            inst = _ref_margin(z, eps)
            gamma_ref = inst if gamma_ref is None else gamma_ref  # gamma_0 := first estimate
            gamma_ref = (1.0 - alpha) * gamma_ref + alpha * inst
            if alpha == 1.0:
                assert gamma_ref == inst
            if n == 0 and init is not None:
                assert _close_scalar(gamma_ref, (1.0 - alpha) * init + alpha * inst)
            ret = tracker.update_variance_margin(z)
            assert _close_scalar(ret, gamma_ref) and _close_scalar(float(tracker.gamma_bar), gamma_ref)
            assert _close_scalar(tracker.threshold, kappa * math.sqrt(2.0 * d) * gamma_ref)
        # B < 2 raises and leaves gamma unchanged.
        before = float(tracker.gamma_bar)
        _expect(ValueError, lambda tr=tracker: tr.update_variance_margin(torch.randn(1, d)))
        _expect(ValueError, lambda tr=tracker: tr.instantaneous_margin(torch.randn(1, d)))
        _expect(ValueError, lambda tr=tracker: tr.update_variance_margin(torch.randn(4, d + 1)))
        assert float(tracker.gamma_bar) == before
        # freeze(): gamma* and tau* = kappa sqrt(2d) gamma*; no further updates.
        tau_star = tracker.freeze()
        tau_ref = kappa * math.sqrt(2.0 * d) * gamma_ref
        assert _close_scalar(tau_star, tau_ref) and _close_scalar(tracker.threshold, tau_ref)
        assert _close_scalar(tracker.gamma_bar_star, gamma_ref)
        _expect(RuntimeError, lambda tr=tracker: tr.update_variance_margin(batches[0]))
        assert float(tracker.gamma_bar) == before and tracker.threshold == tau_star
    over = MilestoneTracker(d, kappa=kappa, alpha_gamma=0.5, eps=eps)
    _expect(RuntimeError, over.freeze)  # uncalibrated, no gamma_bar_star given
    _expect(ValueError, lambda: over.freeze(gamma_bar_star=-1.0))
    assert _close_scalar(over.freeze(gamma_bar_star=2.5), kappa * math.sqrt(2.0 * d) * 2.5)

    # ---- Eq. 9a strict boundary: kappa sqrt(2d) = 0.25 * 4 = 1, so tau* = gamma* = 1.5 exactly.
    tau = 1.5
    below = float(torch.nextafter(torch.tensor(tau), torch.tensor(0.0)))  # next float32 below tau
    for mm in (2, 1):  # M = 2: advance test at m < M;  M = 1: completion test at m = M
        tr = MilestoneTracker(d, kappa=0.25, alpha_gamma=0.5, eps=eps)
        assert tr.freeze(gamma_bar_star=tau) == tau
        goals = torch.randn(2, mm, d, generator=gen)
        goals[..., 0] = 0.0  # offsets along axis 0 are then represented exactly
        z = goals[:, 0].clone()
        z[0, 0], z[1, 0] = tau, below
        tr.reset(mm, 2)
        upd = tr.step(z, goals)
        assert upd.distance[0].item() == tau and upd.distance[1].item() == below  # exact distances
        assert upd.reached.tolist() == [False, True]
        if mm == 2:
            assert upd.m_next.tolist() == [1, 2] and upd.task_complete.tolist() == [False, False]
        else:  # single-stage task: completes without moving the pointer
            assert upd.m_next.tolist() == [1, 1] and upd.task_complete.tolist() == [False, True]

    # ---- Eq. 9a randomized run against the per-element reference (tau* = 1.5).
    tr = MilestoneTracker(d, kappa=0.25, alpha_gamma=0.5, eps=eps)
    tr.freeze(gamma_bar_star=tau)
    goals = torch.randn(b, num_m, d, generator=gen) * 5.0
    goals[..., 0] = 0.0
    tr.reset(num_m, b)
    m_ref, done_ref = [1] * b, [False] * b
    seen = {"advance": 0, "hold": 0, "complete": 0, "masked_reach": 0, "boundary": 0}
    for _ in range(40):
        kinds = torch.randint(0, 3, (b,), generator=gen).tolist()  # 0 inside, 1 outside, 2 boundary
        mask = (torch.rand(b, generator=gen) < 0.7).tolist()
        z = torch.empty(b, d)
        for i in range(b):
            g = goals[i, m_ref[i] - 1]
            if kinds[i] == 2:
                z[i] = g
                z[i, 0] = tau
            else:
                direction = torch.randn(d, generator=gen)
                radius = 0.7 if kinds[i] == 0 else 4.0
                z[i] = g + radius * direction / direction.norm()
        assert torch.equal(tr.active_goal(goals), torch.stack([goals[i, m_ref[i] - 1] for i in range(b)]))
        m_next_ref, done_next_ref, dist_ref, reached_ref = _ref_pointer_step(
            m_ref, done_ref, z, goals, tau, num_m, mask)
        upd = tr.step(z, goals, update_mask=torch.tensor(mask))
        assert upd.m_t.tolist() == m_ref and upd.m_next.tolist() == m_next_ref
        assert upd.reached.tolist() == reached_ref and upd.task_complete.tolist() == done_next_ref
        assert _close(upd.distance, torch.tensor(dist_ref, dtype=torch.float64))
        assert torch.equal(tr.m, upd.m_next) and torch.equal(tr.task_complete, upd.task_complete)
        for i in range(b):
            assert m_ref[i] <= m_next_ref[i] <= num_m and m_next_ref[i] >= 1   # monotone, in range
            assert not done_ref[i] or done_next_ref[i]                         # sticky completion
            if not mask[i]:
                assert m_next_ref[i] == m_ref[i] and done_next_ref[i] == done_ref[i]
                seen["masked_reach"] += int(reached_ref[i])
            seen["advance"] += int(m_next_ref[i] > m_ref[i])
            seen["hold"] += int(mask[i] and m_next_ref[i] == m_ref[i])
            seen["complete"] += int(done_next_ref[i] and not done_ref[i])
            seen["boundary"] += int(kinds[i] == 2 and not reached_ref[i])
        m_ref, done_ref = m_next_ref, done_next_ref
    assert all(v > 0 for v in seen.values()), seen
    assert all(done_ref)  # every element completed and stayed complete

    # ---- Errors: pointer before reset(), bad constructor / reset arguments.
    fresh = MilestoneTracker(d, kappa=0.25, alpha_gamma=0.5, eps=eps)
    fresh.freeze(gamma_bar_star=1.0)
    _expect(RuntimeError, lambda: fresh.step(torch.randn(b, d), goals))
    _expect(RuntimeError, lambda: fresh.active_goal(goals))
    _expect(RuntimeError, lambda: fresh.completion_test(torch.randn(b, d), goals))
    _expect(ValueError, lambda: fresh.reset(0, b))
    for kw in (dict(kappa=0.0), dict(kappa=1.0), dict(alpha_gamma=0.0), dict(alpha_gamma=1.5),
               dict(eps=0.0), dict(gamma_bar_init=0.0), dict(gamma_bar_init=-1.0)):
        args = dict(kappa=0.25, alpha_gamma=0.5, eps=eps)
        args.update(kw)
        _expect(ValueError, lambda a=args: MilestoneTracker(d, **a))
    print("selector.py self-test passed")


if __name__ == "__main__":
    _self_test()
