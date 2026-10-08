"""
solver.py -- Stage 4 of the Goal-Conditioned VPA framework (preprint_261008.pdf, Secs. 4.4 and 4.5).

Implements
    * ``standardize_latents`` / ``standardize_proprioception`` -- the fixed standardisation of
      Sec. 4.5: z_tilde = (z - mu_Z) / gamma*  (applied to z_t and z_hat_{t+1}) and
      S_tilde = (S - mu_S) / sigma_S with dataset statistics.
    * ``ConditionalVectorField`` -- v_theta : R^{H x d_a} x [0, 1] x R^{d_e} -> R^{H x d_a}.
      A non-causal transformer over the H chunk positions: every evaluation processes the
      whole chunk in parallel (Sec. 4.4).
    * ``FlowMatchingSolver`` -- pi_theta^l:
          e_t      = Concat(Embed(u_t), z_t, z_hat_{t+1}, S_t, c_text) in R^{d_e},
                     d_e = d_u + 2d + d_s + d_c                                         (Eq. 12)
          A^(rho)  = rho A_t + (1 - rho) xi,   xi ~ N(0, I)                              (Eq. 13)
          L_solver = E ||v_theta(A^(rho), rho | e_t) - (A_t - xi)||_2^2,  rho ~ U[0, 1]  (Eq. 14)
          sampling: A^(0) = xi,  A^(rho_{k+1}) = A^(rho_k) + (1/K) v_theta(A^(rho_k), rho_k | e_t),
                    rho_k = k / K,  k = 0, ..., K - 1   (explicit Euler, K in {1, 2, 3})
    * ``UncertaintyHorizonFilter`` -- the fast-attack / slow-release filter and executed horizon:
          sigma_bar_{t+1} = max{ sigma_{t+1}, (1 - a_s) sigma_bar_t + a_s sigma_{t+1} }
          H_t = max( H_min, floor( H_max * exp(-beta * sigma_bar_{t+1}) ) )               (Eq. 15)

This file is self-contained (it imports nothing from the other VPA modules) and can be tested
on its own with ``python solver.py``.
"""

from __future__ import annotations

import math
from typing import Any, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor

__all__ = [
    "standardize_latents",
    "standardize_proprioception",
    "SinusoidalTimeEmbedding",
    "ConditionalVectorField",
    "FlowMatchingSolver",
    "UncertaintyHorizonFilter",
    "executed_prefix_mask",
]

ALLOWED_INTEGRATION_STEPS = (1, 2, 3)


# ---------------------------------------------------------------------------------------------
# Fixed standardisation (Sec. 4.5)
# ---------------------------------------------------------------------------------------------
def standardize_latents(z: Tensor, latent_mean: Tensor, gamma_bar_star: Union[float, Tensor]) -> Tensor:
    """z_tilde = (z - mu_Z) / gamma*  (Sec. 4.5), applied to both z_t and z_hat_{t+1}.

    Args:
        z: ``[..., d]`` latents.
        latent_mean: ``[d]`` mean latent mu_Z over the training data.
        gamma_bar_star: frozen latent scale gamma* (scalar).

    Returns:
        ``[..., d]`` standardised latents.
    """
    return (z - latent_mean) / gamma_bar_star


def standardize_proprioception(state: Tensor, proprio_mean: Tensor, proprio_std: Tensor) -> Tensor:
    """S_tilde = (S - mu_S) / sigma_S with per-dimension dataset statistics (Sec. 4.5).

    Args:
        state: ``[..., d_s]``;  proprio_mean, proprio_std: ``[d_s]`` (std strictly positive).

    Returns:
        ``[..., d_s]``.
    """
    return (state - proprio_mean) / proprio_std


# ---------------------------------------------------------------------------------------------
# v_theta : conditional vector field
# ---------------------------------------------------------------------------------------------
class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal features of the flow time rho in [0, 1].

    Shapes:
        ``rho: [B]`` -> ``[B, dim]``
    """

    def __init__(self, dim: int, scale: float = 1000.0) -> None:
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("time embedding dimension must be even")
        half = dim // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half, dtype=torch.float32) / half)
        self.register_buffer("freqs", freqs, persistent=False)
        self.scale: float = float(scale)

    def forward(self, rho: Tensor) -> Tensor:
        """``[B]`` -> ``[B, dim]``."""
        args = self.scale * rho.to(self.freqs.dtype).unsqueeze(-1) * self.freqs  # [B, dim/2]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class ConditionalVectorField(nn.Module):
    """v_theta(A^(rho), rho | e_t): a non-causal transformer over the H action positions.

    Token h = W_a a_h + p_h + c, where c = MLP(Concat(e_t, time(rho))) is broadcast to every
    position; a conditioning token carrying c is also prepended so that every layer can attend to
    it. No causal mask is used: all H positions are produced jointly in one evaluation.

    Args:
        horizon: H (= H_max), chunk length.
        action_dim: d_a.
        cond_dim: d_e.
        width, depth, num_heads, mlp_ratio: transformer hyper-parameters.
        time_embed_dim: width of the rho embedding.

    Shapes:
        (``A_rho [B, H, d_a]``, ``rho [B]`` or scalar, ``e_t [B, d_e]``) -> ``[B, H, d_a]``
    """

    def __init__(
        self,
        horizon: int,
        action_dim: int,
        cond_dim: int,
        width: int = 256,
        depth: int = 4,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        time_embed_dim: int = 128,
    ) -> None:
        super().__init__()
        self.horizon: int = horizon
        self.action_dim: int = action_dim
        self.cond_dim: int = cond_dim
        self.action_in = nn.Linear(action_dim, width)
        self.pos_embed = nn.Parameter(torch.zeros(1, horizon, width))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.time_embed = SinusoidalTimeEmbedding(time_embed_dim)
        self.cond_mlp = nn.Sequential(
            nn.Linear(cond_dim + time_embed_dim, width), nn.GELU(), nn.Linear(width, width)
        )
        layer = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=num_heads,
            dim_feedforward=int(round(width * mlp_ratio)),
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(layer, num_layers=depth, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(width)
        self.action_out = nn.Linear(width, action_dim)

    def forward(self, actions: Tensor, rho: Union[float, Tensor], cond: Tensor) -> Tensor:
        """Evaluate the vector field.

        Args:
            actions: ``[B, H, d_a]`` current point A^(rho) on the flow.
            rho: flow time, a Python float, a 0-dim tensor, or ``[B]``.
            cond: ``[B, d_e]`` unified conditioning vector e_t.

        Returns:
            velocity: ``[B, H, d_a]``.
        """
        b = actions.shape[0]
        if actions.shape[1:] != (self.horizon, self.action_dim):
            raise ValueError(f"expected actions [B, {self.horizon}, {self.action_dim}], got {tuple(actions.shape)}")
        if cond.shape != (b, self.cond_dim):
            raise ValueError(f"expected e_t of shape [B, {self.cond_dim}], got {tuple(cond.shape)}")
        rho_t = torch.as_tensor(rho, dtype=actions.dtype, device=actions.device)
        if rho_t.dim() == 0:
            rho_t = rho_t.expand(b)
        c = self.cond_mlp(torch.cat([cond, self.time_embed(rho_t).to(cond.dtype)], dim=-1))  # [B, W]
        tokens = self.action_in(actions) + self.pos_embed + c.unsqueeze(1)                # [B, H, W]
        x = self.blocks(torch.cat([c.unsqueeze(1), tokens], dim=1))                        # [B, 1+H, W]
        return self.action_out(self.norm(x[:, 1:]))                                        # [B, H, d_a]


# ---------------------------------------------------------------------------------------------
# pi_theta^l : flow-matching action solver
# ---------------------------------------------------------------------------------------------
class FlowMatchingSolver(nn.Module):
    """Conditional Flow-Matching action-chunk solver pi_theta^l (Secs. 4.4, 4.5).

    Holds the learned primitive embedding Embed : U -> R^{d_u}, the fixed standardisation
    statistics (mu_Z, gamma*, mu_S, sigma_S; set with ``set_standardization_stats`` after the
    JEPA modules are frozen), and the vector field v_theta.

    Args:
        num_primitives: N_u.
        primitive_embed_dim: d_u.
        latent_dim: d.
        proprio_dim: d_s.
        text_dim: d_c.
        horizon: H = H_max.
        action_dim: d_a.
        num_integration_steps: K in {1, 2, 3}.
        **field_kwargs: forwarded to ``ConditionalVectorField``.

    Shapes:
        ``build_conditioning``: (``u [B]``, ``z_t [B, d]``, ``z_hat [B, d]``, ``S_t [B, d_s]``,
                                ``c_text [B, d_c]``) -> ``e_t [B, d_e]``
        ``flow_matching_loss``: (``A_t [B, H, d_a]``, ``e_t [B, d_e]``) -> scalar
        ``generate_chunk``:     ``e_t [B, d_e]`` -> ``A_hat_t [B, H, d_a]``
    """

    def __init__(
        self,
        num_primitives: int,
        primitive_embed_dim: int,
        latent_dim: int,
        proprio_dim: int,
        text_dim: int,
        horizon: int,
        action_dim: int,
        num_integration_steps: int = 2,
        **field_kwargs: Any,
    ) -> None:
        super().__init__()
        if num_integration_steps not in ALLOWED_INTEGRATION_STEPS:
            raise ValueError(f"K must be one of {ALLOWED_INTEGRATION_STEPS}")
        self.num_primitives: int = num_primitives
        self.primitive_embed_dim: int = primitive_embed_dim
        self.latent_dim: int = latent_dim
        self.proprio_dim: int = proprio_dim
        self.text_dim: int = text_dim
        self.horizon: int = horizon
        self.action_dim: int = action_dim
        self.num_integration_steps: int = num_integration_steps
        self.cond_dim: int = primitive_embed_dim + 2 * latent_dim + proprio_dim + text_dim  # d_e
        self.primitive_embedding = nn.Embedding(num_primitives, primitive_embed_dim)       # Embed
        self.vector_field = ConditionalVectorField(horizon, action_dim, self.cond_dim, **field_kwargs)
        nan = float("nan")
        self.register_buffer("latent_mean", torch.full((latent_dim,), nan))
        self.register_buffer("gamma_bar_star", torch.tensor(nan))
        self.register_buffer("proprio_mean", torch.full((proprio_dim,), nan))
        self.register_buffer("proprio_std", torch.full((proprio_dim,), nan))

    # ---------------- Sec. 4.5: fixed statistics ----------------
    @torch.no_grad()
    def set_standardization_stats(
        self,
        latent_mean: Tensor,
        gamma_bar_star: float,
        proprio_mean: Tensor,
        proprio_std: Tensor,
    ) -> None:
        """Store mu_Z ``[d]``, gamma* (scalar), mu_S ``[d_s]`` and sigma_S ``[d_s]``."""
        latent_mean = torch.as_tensor(latent_mean, dtype=self.latent_mean.dtype)
        proprio_mean = torch.as_tensor(proprio_mean, dtype=self.proprio_mean.dtype)
        proprio_std = torch.as_tensor(proprio_std, dtype=self.proprio_std.dtype)
        if latent_mean.shape != (self.latent_dim,):
            raise ValueError(f"latent_mean must have shape [{self.latent_dim}]")
        if proprio_mean.shape != (self.proprio_dim,) or proprio_std.shape != (self.proprio_dim,):
            raise ValueError(f"proprioception statistics must have shape [{self.proprio_dim}]")
        if not gamma_bar_star > 0.0:
            raise ValueError("gamma_bar_star must be positive")
        if not bool((proprio_std > 0).all()):
            raise ValueError("proprio_std must be strictly positive in every dimension")
        if not (torch.isfinite(latent_mean).all() and torch.isfinite(proprio_mean).all()):
            raise ValueError("statistics must be finite")
        self.latent_mean.copy_(latent_mean)
        self.gamma_bar_star.fill_(float(gamma_bar_star))
        self.proprio_mean.copy_(proprio_mean)
        self.proprio_std.copy_(proprio_std)

    @property
    def is_calibrated(self) -> bool:
        return bool(
            torch.isfinite(self.latent_mean).all()
            and torch.isfinite(self.gamma_bar_star)
            and torch.isfinite(self.proprio_mean).all()
            and torch.isfinite(self.proprio_std).all()
        )

    # ---------------- Eq. 12: unified conditioning ----------------
    def build_conditioning(
        self,
        primitive: Tensor,
        z_t: Tensor,
        z_hat: Tensor,
        proprioception: Tensor,
        c_text: Tensor,
    ) -> Tensor:
        """e_t = Concat(Embed(u_t), z_tilde_t, z_hat_tilde_{t+1}, S_tilde_t, c_text)  (Eq. 12).

        Args:
            primitive:      ``[B]`` int64 tokens u_t.
            z_t:            ``[B, d]`` current latent (raw; standardised here).
            z_hat:          ``[B, d]`` predicted latent z_hat_{t+1} (raw; standardised here).
            proprioception: ``[B, d_s]`` raw proprioceptive state S_t (standardised here).
            c_text:         ``[B, d_c]`` instruction embedding.

        Returns:
            e_t: ``[B, d_e]`` with d_e = d_u + 2d + d_s + d_c.
        """
        if not self.is_calibrated:
            raise RuntimeError("standardisation statistics are not set; call set_standardization_stats")
        b = z_t.shape[0]
        expected = {
            "z_t": (z_t, (b, self.latent_dim)),
            "z_hat": (z_hat, (b, self.latent_dim)),
            "proprioception": (proprioception, (b, self.proprio_dim)),
            "c_text": (c_text, (b, self.text_dim)),
        }
        for name, (tensor, shape) in expected.items():
            if tuple(tensor.shape) != shape:
                raise ValueError(f"{name} must have shape {list(shape)}, got {list(tensor.shape)}")
        if primitive.shape != (b,):
            raise ValueError(f"primitive must have shape [{b}]")
        e_t = torch.cat(
            [
                self.primitive_embedding(primitive.long()),                                   # [B, d_u]
                standardize_latents(z_t, self.latent_mean, self.gamma_bar_star),              # [B, d]
                standardize_latents(z_hat, self.latent_mean, self.gamma_bar_star),            # [B, d]
                standardize_proprioception(proprioception, self.proprio_mean, self.proprio_std),  # [B, d_s]
                c_text,                                                                       # [B, d_c]
            ],
            dim=-1,
        )
        return e_t

    # ---------------- Eqs. 13-14: training objective ----------------
    def flow_matching_loss(
        self, actions: Tensor, cond: Tensor, generator: Optional[torch.Generator] = None
    ) -> Tensor:
        """L_solver = E_{t, xi, rho ~ U[0,1]} ||v_theta(A^(rho), rho | e_t) - (A_t - xi)||_2^2 (Eq. 14).

        ||.||_2^2 is summed over the H x d_a chunk entries; the expectation is the batch mean.

        Args:
            actions: ``[B, H, d_a]`` ground-truth chunks A_t.
            cond: ``[B, d_e]`` conditioning vectors e_t.

        Returns:
            scalar loss.
        """
        b = actions.shape[0]
        xi = torch.randn(actions.shape, generator=generator, device=actions.device, dtype=actions.dtype)
        rho = torch.rand(b, generator=generator, device=actions.device, dtype=actions.dtype)  # U[0, 1)
        rho_b = rho.view(b, 1, 1)
        a_rho = rho_b * actions + (1.0 - rho_b) * xi                                    # Eq. 13
        velocity = self.vector_field(a_rho, rho, cond)
        return (velocity - (actions - xi)).pow(2).sum(dim=(1, 2)).mean()

    # ---------------- Sec. 4.4: K-step Euler sampling ----------------
    def generate_chunk(
        self,
        cond: Tensor,
        noise: Optional[Tensor] = None,
        generator: Optional[torch.Generator] = None,
    ) -> Tensor:
        """Sample A_hat_t by integrating v_theta from rho = 0 to 1 with K explicit Euler steps.

        A^(0) = xi ~ N(0, I);  A^(rho_{k+1}) = A^(rho_k) + (1/K) v_theta(A^(rho_k), rho_k | e_t),
        rho_k = k / K for k = 0, ..., K - 1. Exactly K sequential evaluations of v_theta, each over
        the full ``[B, H, d_a]`` chunk; the loop length does not depend on H.

        Args:
            cond: ``[B, d_e]`` conditioning vectors e_t.
            noise: optional ``[B, H, d_a]`` initial sample xi (drawn from N(0, I) if omitted).
            generator: optional RNG for xi.

        Returns:
            A_hat_t: ``[B, H, d_a]``.
        """
        b = cond.shape[0]
        shape = (b, self.horizon, self.action_dim)
        if noise is None:
            noise = torch.randn(shape, generator=generator, device=cond.device, dtype=cond.dtype)
        elif tuple(noise.shape) != shape:
            raise ValueError(f"noise must have shape {list(shape)}")
        k_steps = self.num_integration_steps
        dt = 1.0 / k_steps
        a = noise
        for k in range(k_steps):
            rho_k = k / k_steps
            a = a + dt * self.vector_field(a, rho_k, cond)
        return a


# ---------------------------------------------------------------------------------------------
# Uncertainty-adaptive executed horizon (Eq. 15)
# ---------------------------------------------------------------------------------------------
class UncertaintyHorizonFilter:
    """Fast-attack / slow-release filter on sigma and the executed horizon H_t (Eq. 15).

        sigma_bar_{t+1} = max{ sigma_{t+1}, (1 - alpha_sigma) sigma_bar_t + alpha_sigma sigma_{t+1} }
        H_t             = max( H_min, floor( H_max * exp(-beta * sigma_bar_{t+1}) ) )

    The filter state sigma_bar starts at 0 for each episode, so the first decision step uses
    sigma_bar = sigma exactly (fast attack).

    Args:
        h_min: H_min >= 1.
        h_max: H_max = H (chunk length), H_min <= H_max.
        beta: sensitivity beta > 0.
        alpha_sigma: release rate alpha_sigma in (0, 1].

    Shapes:
        ``update``: ``sigma [B]`` (+ optional ``mask [B]`` bool) -> (``sigma_bar [B]``, ``H_t [B]`` int64)
    """

    def __init__(self, h_min: int, h_max: int, beta: float, alpha_sigma: float) -> None:
        if not 1 <= h_min <= h_max:
            raise ValueError("require 1 <= H_min <= H_max")
        if beta <= 0.0:
            raise ValueError("beta must be positive")
        if not 0.0 < alpha_sigma <= 1.0:
            raise ValueError("alpha_sigma must lie in (0, 1]")
        self.h_min: int = int(h_min)
        self.h_max: int = int(h_max)
        self.beta: float = float(beta)
        self.alpha_sigma: float = float(alpha_sigma)
        self.sigma_bar: Optional[Tensor] = None

    def reset(self, batch_size: int, device: Optional[torch.device] = None) -> None:
        """Start an episode with sigma_bar = 0."""
        self.sigma_bar = torch.zeros(batch_size, device=device)

    def filter(self, sigma: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """Asymmetric filter; commits and returns sigma_bar_{t+1} ``[B]``.

        Args:
            sigma: ``[B]`` raw scores sigma_{t+1} >= 0.
            mask: optional ``[B]`` bool; elements outside the mask keep their previous sigma_bar.
        """
        prev = self.sigma_bar if self.sigma_bar is not None else torch.zeros_like(sigma)
        prev = prev.to(device=sigma.device, dtype=sigma.dtype)
        new = torch.maximum(sigma, (1.0 - self.alpha_sigma) * prev + self.alpha_sigma * sigma)
        if mask is not None:
            new = torch.where(mask.to(device=sigma.device, dtype=torch.bool), new, prev)
        self.sigma_bar = new
        return new

    def horizon(self, sigma_bar: Tensor) -> Tensor:
        """Eq. 15: H_t = max(H_min, floor(H_max * exp(-beta * sigma_bar))) -> ``[B]`` int64."""
        raw = torch.floor(self.h_max * torch.exp(-self.beta * sigma_bar)).long()
        return torch.clamp(raw, min=self.h_min)

    def update(self, sigma: Tensor, mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        """Filter sigma and compute H_t: returns (``sigma_bar [B]``, ``H_t [B]`` int64)."""
        sigma_bar = self.filter(sigma, mask)
        return sigma_bar, self.horizon(sigma_bar)


def executed_prefix_mask(executed_horizon: Tensor, horizon: int) -> Tensor:
    """Boolean mask of the executed prefix: ``H_t [B]`` -> ``[B, H]`` with True for h < H_t."""
    steps = torch.arange(horizon, device=executed_horizon.device)
    return steps.unsqueeze(0) < executed_horizon.unsqueeze(1)


# ---------------------------------------------------------------------------------------------
# Self-test (run: python solver.py)
# ---------------------------------------------------------------------------------------------
class _CountingField(nn.Module):
    """Wraps v_theta and counts evaluations (used by the self-test only)."""

    def __init__(self, field: nn.Module) -> None:
        super().__init__()
        self.field = field
        self.calls: List[float] = []

    def forward(self, actions: Tensor, rho: Union[float, Tensor], cond: Tensor) -> Tensor:
        # Record rho exactly as passed: torch.as_tensor(float) would round to float32 (1/3 != 1/3).
        self.calls.append(rho if isinstance(rho, float) else float(rho.reshape(-1)[0]))
        return self.field(actions, rho, cond)


_RTOL, _ATOL = 1e-5, 1e-6  # float32 module output vs float64 reference (CLAUDE.md)


def _close(actual: Any, ref: Any) -> bool:
    a = torch.as_tensor(actual).detach().double()
    r = torch.as_tensor(ref).detach().double()
    return torch.allclose(a, r, rtol=_RTOL, atol=_ATOL)


def _expect(exc: type, fn: Any) -> None:
    try:
        fn()
    except exc:
        return
    raise AssertionError(f"expected {exc.__name__}")


class _ZeroField(nn.Module):
    """Test stand-in v_theta == 0 (self-test only)."""

    def forward(self, actions: Tensor, rho: Union[float, Tensor], cond: Tensor) -> Tensor:
        return torch.zeros_like(actions)


class _AffineField(nn.Module):
    """Test stand-in v(A, rho, e) = a A + b rho + reshape(e W), easy to evaluate in float64.

    Shapes: (``[B, H, d_a]``, scalar or ``[B]``, ``[B, d_e]``) -> ``[B, H, d_a]``
    """

    def __init__(self, horizon: int, action_dim: int, cond_dim: int, generator: torch.Generator) -> None:
        super().__init__()
        self.a, self.b = 0.7, 0.3
        self.register_buffer("w", 0.1 * torch.randn(cond_dim, horizon * action_dim, generator=generator))

    def forward(self, actions: Tensor, rho: Union[float, Tensor], cond: Tensor) -> Tensor:
        bsz = actions.shape[0]
        rho_t = torch.as_tensor(rho, dtype=actions.dtype)
        rho_t = rho_t.expand(bsz) if rho_t.dim() == 0 else rho_t
        return self.a * actions + self.b * rho_t.view(bsz, 1, 1) + (cond @ self.w).view(actions.shape)


class _RecordingField(nn.Module):
    """Wraps v_theta and records its last input and output (self-test only)."""

    def __init__(self, field: nn.Module) -> None:
        super().__init__()
        self.field = field
        self.record: Optional[Tuple[Tensor, Tensor, Tensor]] = None

    def forward(self, actions: Tensor, rho: Union[float, Tensor], cond: Tensor) -> Tensor:
        out = self.field(actions, rho, cond)
        self.record = (actions.detach().clone(), torch.as_tensor(rho).detach().clone(), out)
        return out


def _ref_conditioning(solver: "FlowMatchingSolver", u: Tensor, z_t: Tensor, z_hat: Tensor,
                      s_t: Tensor, c: Tensor) -> Tensor:
    """Float64 reference of Eq. 12 with the Sec. 4.5 standardisation, written slot by slot."""
    du, d, ds, dc = solver.primitive_embed_dim, solver.latent_dim, solver.proprio_dim, solver.text_dim
    bsz = z_t.shape[0]
    emb = solver.primitive_embedding.weight.detach().double()
    mu_z, g = solver.latent_mean.double(), float(solver.gamma_bar_star)
    mu_s, sd_s = solver.proprio_mean.double(), solver.proprio_std.double()
    e = torch.zeros(bsz, du + 2 * d + ds + dc, dtype=torch.float64)
    for k in range(bsz):
        for j in range(du):
            e[k, j] = emb[int(u[k]), j]
        for j in range(d):
            e[k, du + j] = (float(z_t[k, j]) - float(mu_z[j])) / g
            e[k, du + d + j] = (float(z_hat[k, j]) - float(mu_z[j])) / g
        for j in range(ds):
            e[k, du + 2 * d + j] = (float(s_t[k, j]) - float(mu_s[j])) / float(sd_s[j])
        for j in range(dc):
            e[k, du + 2 * d + ds + j] = float(c[k, j])
    return e


def _self_test() -> None:
    torch.manual_seed(0)
    gen = torch.Generator().manual_seed(99)
    b, nu, du, d, ds, dc, h, da = 3, 4, 5, 6, 7, 8, 10, 2
    field_kwargs = dict(width=32, depth=2, num_heads=4, time_embed_dim=16)

    def make(k: int = 2, horizon: int = h) -> FlowMatchingSolver:
        return FlowMatchingSolver(nu, du, d, ds, dc, horizon, da, num_integration_steps=k, **field_kwargs)

    lat_mean, gamma_star = torch.randn(d, generator=gen), 2.5
    s_mean, s_std = torch.randn(ds, generator=gen), torch.rand(ds, generator=gen) + 0.5
    u = torch.randint(0, nu, (b,), generator=gen)
    z_t, z_hat = torch.randn(b, d, generator=gen), torch.randn(b, d, generator=gen)
    s_t, c = torch.randn(b, ds, generator=gen), torch.randn(b, dc, generator=gen)

    # ---- Eq. 12 + Sec. 4.5: unified conditioning and standardisation.
    solver = make()
    _expect(RuntimeError, lambda: solver.build_conditioning(u, z_t, z_hat, s_t, c))  # stats not set
    bad_std_zero, bad_std_neg = s_std.clone(), s_std.clone()
    bad_std_zero[2], bad_std_neg[4] = 0.0, -0.5
    _expect(ValueError, lambda: solver.set_standardization_stats(lat_mean, gamma_star, s_mean, bad_std_zero))
    _expect(ValueError, lambda: solver.set_standardization_stats(lat_mean, gamma_star, s_mean, bad_std_neg))
    _expect(ValueError, lambda: solver.set_standardization_stats(lat_mean, 0.0, s_mean, s_std))
    _expect(ValueError, lambda: solver.set_standardization_stats(lat_mean, -1.0, s_mean, s_std))
    _expect(ValueError, lambda: solver.set_standardization_stats(lat_mean[:-1], gamma_star, s_mean, s_std))
    _expect(ValueError, lambda: solver.set_standardization_stats(lat_mean, gamma_star, s_mean[:-1], s_std))
    nan_mean = lat_mean.clone()
    nan_mean[0] = float("nan")
    _expect(ValueError, lambda: solver.set_standardization_stats(nan_mean, gamma_star, s_mean, s_std))
    assert not solver.is_calibrated  # rejected statistics were not installed
    solver.set_standardization_stats(lat_mean, gamma_star, s_mean, s_std)
    e_t = solver.build_conditioning(u, z_t, z_hat, s_t, c)
    assert solver.cond_dim == du + 2 * d + ds + dc and e_t.shape == (b, du + 2 * d + ds + dc)
    ref_e = _ref_conditioning(solver, u, z_t, z_hat, s_t, c)
    assert torch.equal(e_t[:, :du], solver.primitive_embedding.weight[u])   # Embed(u_t)
    assert _close(e_t[:, du:du + d], ref_e[:, du:du + d])                    # (z_t - mu_Z) / gamma*
    assert _close(e_t[:, du + d:du + 2 * d], ref_e[:, du + d:du + 2 * d])    # (z_hat - mu_Z) / gamma*
    assert _close(e_t[:, du + 2 * d:du + 2 * d + ds], ref_e[:, du + 2 * d:du + 2 * d + ds])  # S_tilde
    assert torch.equal(e_t[:, -dc:], c)                                      # c_text, not standardised
    assert _close(e_t, ref_e)
    _expect(ValueError, lambda: solver.build_conditioning(u, z_t[:, :-1], z_hat, s_t, c))
    _expect(ValueError, lambda: solver.build_conditioning(u, z_t, z_hat[:-1], s_t, c))
    _expect(ValueError, lambda: solver.build_conditioning(u, z_t, z_hat, s_t[:, :-1], c))
    _expect(ValueError, lambda: solver.build_conditioning(u, z_t, z_hat, s_t, c[:, :-1]))
    _expect(ValueError, lambda: solver.build_conditioning(u[:-1], z_t, z_hat, s_t, c))
    e_t = e_t.detach()

    # ---- Eqs. 13-14: flow-matching loss with a fixed generator.
    a_true = torch.randn(b, h, da, generator=gen)

    def replay(seed: int) -> Tuple[Tensor, Tensor]:
        g = torch.Generator().manual_seed(seed)
        xi_r = torch.randn(a_true.shape, generator=g)   # same draw order as flow_matching_loss
        return xi_r, torch.rand(b, generator=g)

    real_field = solver.vector_field
    solver.vector_field = _ZeroField()
    loss0 = solver.flow_matching_loss(a_true, e_t, generator=torch.Generator().manual_seed(7))
    xi, rho = replay(7)
    ref0 = sum(float(a_true[k, i, j] - xi[k, i, j]) ** 2 for k in range(b) for i in range(h) for j in range(da)) / b
    assert _close(loss0, ref0)
    rec = _RecordingField(real_field)
    solver.vector_field = rec
    loss = solver.flow_matching_loss(a_true, e_t, generator=torch.Generator().manual_seed(7))
    a_in, rho_in, vel = rec.record
    vel_ref = vel.detach()  # reference reads values only; the loss keeps its graph
    assert torch.equal(rho_in, rho) and bool(((rho >= 0) & (rho < 1)).all())
    a_rho_ref = torch.zeros(b, h, da, dtype=torch.float64)
    for k in range(b):
        r = float(rho[k])
        a_rho_ref[k] = r * a_true[k].double() + (1.0 - r) * xi[k].double()          # Eq. 13
    assert _close(a_in, a_rho_ref)
    ref_loss = sum((float(vel_ref[k, i, j]) - (float(a_true[k, i, j]) - float(xi[k, i, j]))) ** 2
                   for k in range(b) for i in range(h) for j in range(da)) / b      # Eq. 14
    assert _close(loss, ref_loss)
    loss.backward()
    assert all(p.grad is not None for p in real_field.parameters())
    solver.vector_field = real_field

    # ---- Euler (Sec. 4.4) for every K, against hand-written loops.
    for k in ALLOWED_INTEGRATION_STEPS:
        sk = make(k)
        sk.set_standardization_stats(lat_mean, gamma_star, s_mean, s_std)
        ek = sk.build_conditioning(u, z_t, z_hat, s_t, c).detach()
        xi = torch.randn(b, h, da, generator=gen)
        real_k = sk.vector_field
        # (a) affine stand-in field vs a float64 Euler loop.
        affine = _AffineField(h, da, sk.cond_dim, gen)
        counting = _CountingField(affine)
        sk.vector_field = counting
        with torch.no_grad():
            chunk = sk.generate_chunk(ek, noise=xi)
        assert counting.calls == [j / k for j in range(k)]
        a_ref = xi.double()
        bias = (ek.double() @ affine.w.double()).view(b, h, da)
        for j in range(k):
            a_ref = a_ref + (1.0 / k) * (affine.a * a_ref + affine.b * (j / k) + bias)
        assert chunk.shape == (b, h, da) and _close(chunk, a_ref)
        # (b) real transformer vs a hand-written float32 loop from the same xi.
        field = _CountingField(real_k)
        sk.vector_field = field
        with torch.no_grad():
            chunk = sk.generate_chunk(ek, noise=xi)
            a_loop = xi.clone()
            for j in range(k):
                a_loop = a_loop + (1.0 / k) * real_k(a_loop, j / k, ek)
        assert field.calls == [j / k for j in range(k)] and _close(chunk, a_loop)
        # (c) noise drawn from a generator == the same noise passed explicitly.
        with torch.no_grad():
            c1 = sk.generate_chunk(ek, generator=torch.Generator().manual_seed(11))
            c2 = sk.generate_chunk(ek, noise=torch.randn(b, h, da, generator=torch.Generator().manual_seed(11)))
        assert torch.equal(c1, c2)
        _expect(ValueError, lambda s=sk, e=ek: s.generate_chunk(e, noise=torch.randn(b, h + 1, da)))
    for bad_k in (0, 4):
        _expect(ValueError, lambda kk=bad_k: make(kk))

    # ---- Horizon independence: K evaluations of v_theta for every H.
    for hh in (4, 16, 64):
        for k in ALLOWED_INTEGRATION_STEPS:
            sk = make(k, horizon=hh)
            sk.set_standardization_stats(lat_mean, gamma_star, s_mean, s_std)
            ek = sk.build_conditioning(u, z_t, z_hat, s_t, c).detach()
            counting = _CountingField(sk.vector_field)
            sk.vector_field = counting
            with torch.no_grad():
                chunk = sk.generate_chunk(ek, generator=torch.Generator().manual_seed(hh + k))
            assert len(counting.calls) == k and chunk.shape == (b, hh, da)

    # ---- Eq. 15 filter: fast attack, slow release, masked elements frozen.
    alpha = 0.25
    seq = [0.0, 0.2, 1.0, 1.0, 1.0, 1.0, 0.3, 0.3, 0.1, 0.0, 0.0, 0.6]  # rise, plateau, decay, rise
    mask1 = [True, True, False, True, False, True, True, False, True, True, False, True]
    filt = UncertaintyHorizonFilter(h_min=3, h_max=h, beta=1.0, alpha_sigma=alpha)
    filt.reset(3)
    assert torch.equal(filt.sigma_bar, torch.zeros(3))
    ref_prev = [0.0, 0.0, 0.0]
    prev_mod = filt.sigma_bar.clone()
    rises, decays = [], []
    for t, s in enumerate(seq):
        sig = [s, s, 0.5 * s]
        mask = [True, mask1[t], True]
        sb, ht = filt.update(torch.tensor(sig), mask=torch.tensor(mask))
        ref = [max(sig[i], (1 - alpha) * ref_prev[i] + alpha * sig[i]) if mask[i] else ref_prev[i]
               for i in range(3)]
        assert _close(sb, torch.tensor(ref, dtype=torch.float64))
        assert torch.equal(ht, filt.horizon(sb))
        if t == 0:
            assert float(sb[0]) == float(torch.tensor(s))     # sigma_bar_0 = 0 -> sigma_bar = sigma
        if sig[0] > ref_prev[0]:                               # rise: applied immediately, bit-exact
            rises.append(t)
            assert float(sb[0]) == float(torch.tensor(sig[0]))
        if sig[0] < ref_prev[0]:                               # decay: gradual, stays above sigma
            decays.append(t)
            assert float(sb[0]) > sig[0] and float(sb[0]) < float(prev_mod[0])
        if not mask[1]:
            assert float(sb[1]) == float(prev_mod[1])          # masked element keeps its value
        ref_prev, prev_mod = ref, sb.clone()
    assert {1, 2, 11} <= set(rises) and {6, 7, 8, 9, 10} <= set(decays)
    filt.reset(3)
    assert torch.equal(filt.sigma_bar, torch.zeros(3))
    instant = UncertaintyHorizonFilter(h_min=3, h_max=h, beta=1.0, alpha_sigma=1.0)
    instant.reset(1)
    for s in seq:
        assert torch.equal(instant.filter(torch.tensor([s])), torch.tensor([s]))

    # ---- Eq. 15 horizon: H_t = max(H_min, floor(H_max exp(-beta sigma_bar))).
    h_min, beta = 3, 1.0
    hf = UncertaintyHorizonFilter(h_min=h_min, h_max=h, beta=beta, alpha_sigma=alpha)

    def ref_h(sb_val: float) -> int:
        return max(h_min, math.floor(h * math.exp(-beta * sb_val)))

    assert hf.horizon(torch.tensor([0.0])).tolist() == [h]          # sigma_bar = 0 -> H_max
    assert hf.horizon(torch.tensor([50.0])).tolist() == [h_min]     # large sigma_bar -> H_min
    assert math.floor(h * math.exp(-1.05)) == h_min and math.floor(h * math.exp(-1.3)) == h_min - 1
    assert hf.horizon(torch.tensor([1.05, 1.3])).tolist() == [h_min, h_min]  # floor = H_min, clamped
    grid = torch.linspace(0.0, 3.0, 31)[1:]
    for v in grid.tolist():
        x = h * math.exp(-beta * v)
        assert abs(x - round(x)) > 1e-4, v                          # floor is robust to float32
    ht = hf.horizon(grid)
    assert ht.dtype == torch.long and ht.tolist() == [ref_h(v) for v in grid.tolist()]
    assert bool((ht[1:] <= ht[:-1]).all()) and bool(((ht >= h_min) & (ht <= h)).all())
    hf.reset(len(grid))
    sb_u, ht_u = hf.update(grid)
    assert torch.equal(ht_u, hf.horizon(sb_u))
    h_exec = torch.tensor([h_min, 6, h])
    pm = executed_prefix_mask(h_exec, h)
    assert pm.shape == (3, h)
    assert pm.tolist() == [[step < int(h_exec[i]) for step in range(h)] for i in range(3)]
    for kw in (dict(h_min=0), dict(h_min=h + 1), dict(beta=0.0), dict(beta=-1.0),
               dict(alpha_sigma=0.0), dict(alpha_sigma=1.5)):
        args = dict(h_min=3, h_max=h, beta=1.0, alpha_sigma=0.25)
        args.update(kw)
        _expect(ValueError, lambda a=args: UncertaintyHorizonFilter(**a))
    print("solver.py self-test passed")

if __name__ == "__main__":
    _self_test()
