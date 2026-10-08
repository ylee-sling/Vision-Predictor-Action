"""
predictor.py -- Stage 3 of the Goal-Conditioned VPA framework (preprint_261008.pdf, Secs. 4.3 and 5.2).

Implements
    * ``JEPAPredictor`` -- the latent dynamics model P_omega (Eq. 10):
          (z_hat_{t+1}, sigma_{t+1}) = P_omega(z_t, u_t),
      realised as an ensemble of N_e independently initialised MLP heads {P_omega^(i)}.
      z_hat_{t+1} is the ensemble mean and the epistemic uncertainty is the normalised disagreement
          sigma_{t+1} = 1 / (sqrt(d) * gamma*) * ( (1/N_e) sum_i ||z_hat^(i)_{t+1} - z_hat_{t+1}||_2^2 )^(1/2).
      All heads are evaluated in parallel by batched matrix products, so the ensemble is one
      network evaluation on the K+3 dependency chain (Prop. 5.1).
    * ``VICRegLoss`` -- the JEPA training objective of Eq. 11 with separate terms:
          invariance  E_t ||z_hat_{t+1} - sg(E_psi_bar(I_{t+1}))||_2^2   (applied to each head)
          variance    v(Z) = (1/d) sum_j max(0, gamma - sqrt(Var(Z_{.,j}) + eps))       (Eq. 18)
          covariance  c(Z) = (1/d) sum_{i != j} [C(Z)]_{i,j}^2,
                      C(Z) = 1/(B-1) sum_b (Z_b - Z_bar)(Z_b - Z_bar)^T               (Eq. 19)
          L_JEPA = invariance + lambda_v v(Z) + lambda_c c(Z)                          (Eq. 11)
      v and c are evaluated on the batch Z of *online*-encoder latents z_t = E_psi(I_t).

This file is self-contained (it imports nothing from the other VPA modules) and can be tested
on its own with ``python predictor.py``.
"""

from __future__ import annotations

import math
from typing import Dict, Literal, NamedTuple, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

__all__ = ["EnsembleLinear", "JEPAPredictor", "PredictorOutput", "VICRegLoss"]


# ---------------------------------------------------------------------------------------------
# Ensemble building block
# ---------------------------------------------------------------------------------------------
class EnsembleLinear(nn.Module):
    """N_e independent affine maps evaluated in one batched product: y_i = x_i W_i + b_i.

    Each member's weights are drawn independently with the default ``nn.Linear`` scheme
    (U(-1/sqrt(in), 1/sqrt(in)) for weights and biases), giving the independent initialisations
    required for the ensemble disagreement of Sec. 4.3.

    Shapes:
        input ``x``: ``[N_e, B, in_features]`` -> output ``[N_e, B, out_features]``
    """

    def __init__(self, num_members: int, in_features: int, out_features: int) -> None:
        super().__init__()
        self.num_members: int = num_members
        self.in_features: int = in_features
        self.out_features: int = out_features
        self.weight = nn.Parameter(torch.empty(num_members, in_features, out_features))
        self.bias = nn.Parameter(torch.empty(num_members, 1, out_features))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        bound = 1.0 / math.sqrt(self.in_features)
        nn.init.uniform_(self.weight, -bound, bound)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: Tensor) -> Tensor:
        """``[N_e, B, in]`` -> ``[N_e, B, out]``."""
        return torch.baddbmm(self.bias, x, self.weight)


class PredictorOutput(NamedTuple):
    """Output of ``JEPAPredictor.forward``."""

    z_hat: Tensor             # [B, d]       ensemble mean  z_hat_{t+1}
    sigma: Tensor             # [B]          normalised ensemble disagreement sigma_{t+1} >= 0
    head_predictions: Tensor  # [N_e, B, d]  per-head forecasts z_hat^(i)_{t+1}


# ---------------------------------------------------------------------------------------------
# P_omega : JEPA latent predictor with prediction-head ensemble (Eq. 10)
# ---------------------------------------------------------------------------------------------
class JEPAPredictor(nn.Module):
    """Primitive-conditioned latent predictor P_omega with N_e independent MLP heads.

    Head i maps Concat(z_t, Embed_i(u_t)) through an MLP to z_hat^(i)_{t+1} in R^d, where
    Embed_i is the head's own learned primitive embedding. Forecasts are nu control steps ahead
    (the predictor stride of Sec. 3.3; the paper writes z_{t+1} for z_{t+nu}).

    The frozen latent scale gamma* is stored in the buffer ``latent_scale`` (set with
    ``set_latent_scale`` after E_psi and P_omega are frozen); it is required to compute sigma.

    Args:
        latent_dim: d.
        num_primitives: N_u.
        num_heads: N_e >= 2 (disagreement is undefined for a single head).
        primitive_embed_dim: width of each head's primitive embedding.
        hidden_dim: hidden width of each head's MLP.
        num_hidden_layers: number of hidden layers per head (>= 1).

    Shapes:
        ``forward_heads``: (``z_t [B, d]``, ``u_t [B]`` int64) -> ``[N_e, B, d]``
        ``forward``:       (``z_t [B, d]``, ``u_t [B]`` int64) -> ``PredictorOutput``
                           (z_hat ``[B, d]``, sigma ``[B]``, head_predictions ``[N_e, B, d]``)
    """

    def __init__(
        self,
        latent_dim: int,
        num_primitives: int,
        num_heads: int = 5,
        primitive_embed_dim: int = 64,
        hidden_dim: int = 512,
        num_hidden_layers: int = 2,
    ) -> None:
        super().__init__()
        if num_heads < 2:
            raise ValueError("the ensemble needs N_e >= 2 heads to measure disagreement")
        if num_hidden_layers < 1:
            raise ValueError("num_hidden_layers must be >= 1")
        self.latent_dim: int = latent_dim
        self.num_primitives: int = num_primitives
        self.num_heads: int = num_heads
        # Independent primitive embedding per head, initialised like nn.Embedding (N(0, 1)).
        self.primitive_embedding = nn.Parameter(torch.randn(num_heads, num_primitives, primitive_embed_dim))
        layers = [EnsembleLinear(num_heads, latent_dim + primitive_embed_dim, hidden_dim), nn.GELU()]
        for _ in range(num_hidden_layers - 1):
            layers += [EnsembleLinear(num_heads, hidden_dim, hidden_dim), nn.GELU()]
        layers.append(EnsembleLinear(num_heads, hidden_dim, latent_dim))
        self.heads = nn.Sequential(*layers)
        self.register_buffer("latent_scale", torch.tensor(float("nan")))  # gamma*

    @torch.no_grad()
    def set_latent_scale(self, gamma_bar_star: float) -> None:
        """Store the frozen latent scale gamma* used to normalise sigma."""
        if not gamma_bar_star > 0.0:
            raise ValueError("gamma_bar_star must be positive")
        self.latent_scale.fill_(float(gamma_bar_star))

    @property
    def has_latent_scale(self) -> bool:
        return not bool(torch.isnan(self.latent_scale))

    def forward_heads(self, z_t: Tensor, u_t: Tensor) -> Tensor:
        """Per-head forecasts z_hat^(i)_{t+1}.

        Args:
            z_t: ``[B, d]`` current latent.
            u_t: ``[B]`` int64 primitive tokens in {0, ..., N_u - 1}.

        Returns:
            ``[N_e, B, d]``.
        """
        if z_t.dim() != 2 or z_t.shape[1] != self.latent_dim:
            raise ValueError(f"expected z_t of shape [B, {self.latent_dim}], got {tuple(z_t.shape)}")
        if u_t.shape != (z_t.shape[0],):
            raise ValueError(f"expected u_t of shape [B], got {tuple(u_t.shape)}")
        b = z_t.shape[0]
        emb = self.primitive_embedding[:, u_t.long(), :]                     # [N_e, B, d_u']
        z = z_t.unsqueeze(0).expand(self.num_heads, b, self.latent_dim)      # [N_e, B, d]
        return self.heads(torch.cat([z, emb], dim=-1))                       # [N_e, B, d]

    def ensemble_uncertainty(self, head_predictions: Tensor, z_hat: Tensor) -> Tensor:
        """sigma_{t+1} = ( (1/N_e) sum_i ||z_hat^(i) - z_hat||_2^2 )^(1/2) / (sqrt(d) * gamma*).

        Args:
            head_predictions: ``[N_e, B, d]``;  z_hat: ``[B, d]``.

        Returns:
            sigma: ``[B]`` (dimensionless).
        """
        if not self.has_latent_scale:
            raise RuntimeError("sigma requires the frozen latent scale gamma*; call set_latent_scale first")
        mean_sq_dev = (head_predictions - z_hat.unsqueeze(0)).pow(2).sum(dim=-1).mean(dim=0)  # [B]
        return mean_sq_dev.sqrt() / (math.sqrt(self.latent_dim) * self.latent_scale.to(z_hat.dtype))

    def forward(self, z_t: Tensor, u_t: Tensor) -> PredictorOutput:
        """Eq. 10: (z_hat_{t+1}, sigma_{t+1}) = P_omega(z_t, u_t).

        The ensemble mean z_hat = (1/N_e) sum_i z_hat^(i) (Sec. 4.3) is evaluated relative to head 1,
        z_hat = z_hat^(1) + (1/N_e) sum_i (z_hat^(i) - z_hat^(1)), which is the same mean in exact
        arithmetic. In float32 it returns z_hat bit-exactly when all heads agree, so sigma = 0
        exactly for identical heads; a plain float32 mean of N_e equal values can be off by one ulp.

        Args:
            z_t: ``[B, d]``;  u_t: ``[B]`` int64.

        Returns:
            ``PredictorOutput(z_hat [B, d], sigma [B], head_predictions [N_e, B, d])``.
        """
        heads = self.forward_heads(z_t, u_t)                       # [N_e, B, d]
        z_hat = heads[0] + (heads - heads[0:1]).mean(dim=0)        # [B, d]  ensemble mean (Sec. 4.3)
        return PredictorOutput(z_hat, self.ensemble_uncertainty(heads, z_hat), heads)


# ---------------------------------------------------------------------------------------------
# VICReg-regularised JEPA objective (Eqs. 11, 18, 19)
# ---------------------------------------------------------------------------------------------
class VICRegLoss(nn.Module):
    """L_JEPA = E_t ||z_hat_{t+1} - sg(z_{t+1})||_2^2 + lambda_v v(Z) + lambda_c c(Z)   (Eq. 11).

    ||.||_2^2 is the squared Euclidean norm summed over the d latent dimensions; E_t is the mean
    over the batch. With an ensemble the prediction (invariance) term is applied to each head;
    the paper does not state how the per-head terms are combined, so ``head_reduction``
    selects the mean (default; L_JEPA reduces exactly to Eq. 11 for N_e = 1 and lambda_v,
    lambda_c keep their meaning for any N_e) or the sum.

    Args:
        gamma: variance margin gamma > 0 (Eq. 18).
        eps: numerical constant eps with 0 < eps < gamma^2 (Eq. 18).
        lambda_v: weight of v(Z); must be > 0 (Prop. 5.2).
        lambda_c: weight of c(Z); >= 0.
        head_reduction: "mean" or "sum" over ensemble heads for the invariance term.

    Shapes:
        ``variance`` / ``covariance``: ``Z [B, d]`` -> scalar
        ``invariance``: (``z_hat [N_e, B, d]`` or ``[B, d]``, ``z_target [B, d]``) -> scalar
        ``forward``: (``z_online [B, d]``, ``z_hat [N_e, B, d]`` or ``[B, d]``, ``z_target [B, d]``)
                     -> (scalar total, dict of scalar terms)
    """

    def __init__(
        self,
        lambda_v: float,
        lambda_c: float,
        gamma: float = 1.0,
        eps: float = 1e-4,
        head_reduction: Literal["mean", "sum"] = "mean",
    ) -> None:
        super().__init__()
        if gamma <= 0.0:
            raise ValueError("gamma must be positive")
        if not 0.0 < eps < gamma ** 2:
            raise ValueError("eps must satisfy 0 < eps < gamma^2")
        if lambda_v <= 0.0:
            raise ValueError("lambda_v must be positive (Prop. 5.2)")
        if lambda_c < 0.0:
            raise ValueError("lambda_c must be non-negative")
        if head_reduction not in ("mean", "sum"):
            raise ValueError("head_reduction must be 'mean' or 'sum'")
        self.gamma: float = float(gamma)
        self.eps: float = float(eps)
        self.lambda_v: float = float(lambda_v)
        self.lambda_c: float = float(lambda_c)
        self.head_reduction: str = head_reduction

    @staticmethod
    def _check_batch(latents: Tensor) -> Tuple[int, int]:
        if latents.dim() != 2:
            raise ValueError(f"expected a latent batch of shape [B, d], got {tuple(latents.shape)}")
        b, d = latents.shape
        if b < 2:
            raise ValueError("variance/covariance estimators require B >= 2")
        return b, d

    def variance(self, latents: Tensor) -> Tensor:
        """Eq. 18: v(Z) = (1/d) sum_j max(0, gamma - sqrt(Var(Z_{.,j}) + eps)).

        Var uses the unbiased (1/(B-1)) estimator, the same estimator as C(Z) in Eq. 19.
        """
        self._check_batch(latents)
        std = torch.sqrt(latents.var(dim=0, unbiased=True) + self.eps)  # [d]
        return F.relu(self.gamma - std).mean()

    def covariance(self, latents: Tensor) -> Tensor:
        """Eq. 19: c(Z) = (1/d) sum_{i != j} [C(Z)]_{i,j}^2 with C(Z) = Zc^T Zc / (B - 1)."""
        b, d = self._check_batch(latents)
        centred = latents - latents.mean(dim=0, keepdim=True)            # [B, d]
        cov = centred.transpose(0, 1) @ centred / (b - 1)                # [d, d]
        off_diagonal = cov - torch.diag_embed(torch.diagonal(cov))
        return off_diagonal.pow(2).sum() / d

    def invariance(self, z_hat: Tensor, z_target: Tensor) -> Tensor:
        """Prediction term E_t ||z_hat_{t+1} - sg(z_{t+1})||_2^2, applied to each head.

        Args:
            z_hat: ``[N_e, B, d]`` per-head forecasts (or ``[B, d]`` for a single predictor).
            z_target: ``[B, d]`` momentum-encoder targets E_psi_bar(I_{t+1}); detached here (sg).
        """
        heads = z_hat.unsqueeze(0) if z_hat.dim() == 2 else z_hat          # [N_e, B, d]
        if heads.shape[1:] != z_target.shape:
            raise ValueError(f"shape mismatch: z_hat {tuple(z_hat.shape)} vs z_target {tuple(z_target.shape)}")
        target = z_target.detach()                                        # sg(.)
        per_head = (heads - target.unsqueeze(0)).pow(2).sum(dim=-1).mean(dim=-1)  # [N_e]
        return per_head.mean() if self.head_reduction == "mean" else per_head.sum()

    def anti_collapse(self, latents: Tensor) -> Tensor:
        """L_anti-collapse(Z) = lambda_v v(Z) + lambda_c c(Z)."""
        return self.lambda_v * self.variance(latents) + self.lambda_c * self.covariance(latents)

    def forward(
        self, z_online: Tensor, z_hat: Tensor, z_target: Tensor
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """Full objective of Eq. 11.

        Args:
            z_online: ``[B, d]`` online latents z_t = E_psi(I_t) (the batch Z).
            z_hat:    ``[N_e, B, d]`` (or ``[B, d]``) forecasts P_omega(z_t, u_t).
            z_target: ``[B, d]`` targets E_psi_bar(I_{t+1}).

        Returns:
            total loss (scalar) and a dict with the "invariance", "variance" and "covariance" terms.
        """
        inv = self.invariance(z_hat, z_target)
        var = self.variance(z_online)
        cov = self.covariance(z_online)
        total = inv + self.lambda_v * var + self.lambda_c * cov
        return total, {"invariance": inv.detach(), "variance": var.detach(), "covariance": cov.detach()}


# ---------------------------------------------------------------------------------------------
# Self-test (run: python predictor.py)
# ---------------------------------------------------------------------------------------------
_RTOL, _ATOL = 1e-5, 1e-6  # float32 module output vs float64 reference (CLAUDE.md)


def _close(actual, ref) -> bool:
    a = torch.as_tensor(actual).detach().double()
    r = torch.as_tensor(ref).detach().double()
    return torch.allclose(a, r, rtol=_RTOL, atol=_ATOL)


def _expect(exc: type, fn) -> None:
    try:
        fn()
    except exc:
        return
    raise AssertionError(f"expected {exc.__name__}")


def _ref_head_forward(pred: JEPAPredictor, z_t: Tensor, u_t: Tensor) -> Tensor:
    """Float64 per-head, per-sample reference of the ensemble heads (Eq. 10) -> ``[N_e, B, d]``."""
    ne, (b, d) = pred.num_heads, z_t.shape
    emb = pred.primitive_embedding.detach().double()
    out = torch.zeros(ne, b, d, dtype=torch.float64)
    for i in range(ne):
        for k in range(b):
            x = torch.cat([z_t[k].double(), emb[i, int(u_t[k])]])  # head i's own embedding row
            for layer in pred.heads:
                if isinstance(layer, EnsembleLinear):
                    x = x @ layer.weight[i].detach().double() + layer.bias[i, 0].detach().double()
                else:
                    assert isinstance(layer, nn.GELU) and layer.approximate == "none"
                    x = 0.5 * x * (1.0 + torch.erf(x / math.sqrt(2.0)))
            out[i, k] = x
    return out


def _ref_sigma(heads: Tensor, gamma_star: float) -> Tensor:
    """sqrt((1/N_e) sum_i ||z_i - z_bar||^2) / (sqrt(d) gamma*) with explicit loops -> ``[B]``."""
    ne, b, d = heads.shape
    h = heads.detach().double()
    sig = torch.zeros(b, dtype=torch.float64)
    for k in range(b):
        mean = [sum(float(h[i, k, j]) for i in range(ne)) / ne for j in range(d)]
        sq = sum((float(h[i, k, j]) - mean[j]) ** 2 for i in range(ne) for j in range(d))
        sig[k] = math.sqrt(sq / ne) / (math.sqrt(d) * gamma_star)
    return sig


def _ref_column_var(z: Tensor) -> list:
    """Unbiased per-column variance with explicit float64 sums."""
    zd = z.detach().double()
    b, d = zd.shape
    out = []
    for j in range(d):
        col = [float(zd[k, j]) for k in range(b)]
        mean = sum(col) / b
        out.append(sum((v - mean) ** 2 for v in col) / (b - 1))
    return out


def _ref_v(z: Tensor, gamma: float, eps: float) -> float:
    """Eq. 18 with loops."""
    var = _ref_column_var(z)
    return sum(max(0.0, gamma - math.sqrt(v + eps)) for v in var) / len(var)


def _ref_cov_matrix(z: Tensor) -> Tensor:
    """C(Z) = 1/(B-1) sum_b (Z_b - Zbar)(Z_b - Zbar)^T with a triple loop (Eq. 19)."""
    zd = z.detach().double()
    b, d = zd.shape
    mean = [sum(float(zd[k, j]) for k in range(b)) / b for j in range(d)]
    c = torch.zeros(d, d, dtype=torch.float64)
    for i in range(d):
        for j in range(d):
            c[i, j] = sum((float(zd[k, i]) - mean[i]) * (float(zd[k, j]) - mean[j]) for k in range(b)) / (b - 1)
    return c


def _ref_c(z: Tensor) -> float:
    """Eq. 19 with loops."""
    c = _ref_cov_matrix(z)
    d = c.shape[0]
    return sum(float(c[i, j]) ** 2 for i in range(d) for j in range(d) if i != j) / d


def _ref_invariance(heads: Tensor, target: Tensor, reduction: str) -> float:
    """Eq. 11 prediction term per head (squared norm over d, mean over B), then mean/sum over heads."""
    h, t = heads.detach().double(), target.detach().double()
    ne, b, d = h.shape
    per_head = [sum(sum((float(h[i, k, j]) - float(t[k, j])) ** 2 for j in range(d)) for k in range(b)) / b
                for i in range(ne)]
    return sum(per_head) / ne if reduction == "mean" else sum(per_head)


def _self_test() -> None:
    torch.manual_seed(0)
    gen = torch.Generator().manual_seed(4321)
    b, d, nu, ne = 32, 8, 4, 5
    pred = JEPAPredictor(d, nu, num_heads=ne, primitive_embed_dim=6, hidden_dim=32, num_hidden_layers=2)
    z_t = torch.randn(b, d, generator=gen)
    u_t = torch.randint(0, nu, (b,), generator=gen)

    # ---- Eq. 10 ensemble: each head is its own network.
    heads = pred.forward_heads(z_t, u_t)
    assert heads.shape == (ne, b, d)
    ref_heads = _ref_head_forward(pred, z_t, u_t)
    assert _close(heads, ref_heads)
    linears = [layer for layer in pred.heads if isinstance(layer, EnsembleLinear)]
    for i in range(ne):
        for j in range(i + 1, ne):
            assert not torch.equal(linears[0].weight[i], linears[0].weight[j])
            assert float((heads[i] - heads[j]).detach().abs().max()) > 1e-3  # independent initialisations
    _expect(ValueError, lambda: pred.forward_heads(z_t[:, :-1], u_t))
    _expect(ValueError, lambda: pred.forward_heads(z_t.unsqueeze(0), u_t))
    _expect(ValueError, lambda: pred.forward_heads(z_t, u_t[:-1]))

    # ---- sigma (Sec. 4.3): requires gamma*.
    _expect(RuntimeError, lambda: pred(z_t, u_t))
    for bad in (0.0, -1.0, float("nan")):
        _expect(ValueError, lambda g=bad: pred.set_latent_scale(g))
    gamma_star = 1.7
    pred.set_latent_scale(gamma_star)
    out = pred(z_t, u_t)
    assert out.z_hat.shape == (b, d) and out.sigma.shape == (b,)
    assert torch.equal(out.head_predictions, heads)
    ref_mean = torch.zeros(b, d, dtype=torch.float64)
    for i in range(ne):
        ref_mean += ref_heads[i]
    assert _close(out.z_hat, ref_mean / ne)
    assert _close(out.sigma, _ref_sigma(ref_heads, gamma_star)) and bool((out.sigma >= 0).all())
    # Identical heads: zero disagreement.
    same = JEPAPredictor(d, nu, num_heads=ne, primitive_embed_dim=6, hidden_dim=32, num_hidden_layers=2)
    with torch.no_grad():
        same.primitive_embedding.copy_(same.primitive_embedding[0:1].expand_as(same.primitive_embedding))
        for layer in same.heads:
            if isinstance(layer, EnsembleLinear):
                layer.weight.copy_(layer.weight[0:1].expand_as(layer.weight))
                layer.bias.copy_(layer.bias[0:1].expand_as(layer.bias))
    same.set_latent_scale(gamma_star)
    out_same = same(z_t, u_t)
    for i in range(1, ne):
        assert torch.equal(out_same.head_predictions[i], out_same.head_predictions[0])
    assert torch.equal(out_same.sigma, torch.zeros(b)), float(out_same.sigma.abs().max())

    # ---- Eq. 18 / Eq. 19 against loop references.
    gamma, eps, lam_v, lam_c = 1.0, 1e-4, 25.0, 1.0
    crit = VICRegLoss(lambda_v=lam_v, lambda_c=lam_c, gamma=gamma, eps=eps)
    z = torch.randn(b, d, generator=gen) * torch.linspace(0.2, 2.0, d)  # some columns below the margin
    var_cols = _ref_column_var(z)
    assert any(math.sqrt(v + eps) < gamma for v in var_cols) and any(math.sqrt(v + eps) > gamma for v in var_cols)
    assert _close(crit.variance(z), _ref_v(z, gamma, eps))
    assert _close(crit.covariance(z), _ref_c(z))
    mix = torch.randn(d, d, generator=gen)
    z_corr = torch.randn(b, d, generator=gen) @ mix  # correlated columns: c clearly > 0
    assert _ref_c(z_corr) > 0.1 and _close(crit.covariance(z_corr), _ref_c(z_corr))
    assert _close(crit.variance(z_corr), _ref_v(z_corr, gamma, eps))
    # Unbiased estimator: for B = 2, Var = (a - b)^2 / 2 (biased would be / 4).
    z2 = torch.randn(2, d, generator=gen) * 0.3
    v2 = sum(max(0.0, gamma - math.sqrt(float(z2[0, j] - z2[1, j]) ** 2 / 2 + eps)) for j in range(d)) / d
    assert _close(crit.variance(z2), v2)

    # ---- Prop. 5.2 (i): constant encoder -> v = gamma - sqrt(eps), c = 0, L_const = lambda_v (gamma - sqrt(eps)).
    for zeta in (torch.randn(d, generator=gen), torch.full((d,), 3.0)):
        zc = zeta.expand(b, d).clone()
        assert _close(crit.variance(zc), gamma - math.sqrt(eps))
        assert _close(crit.covariance(zc), 0.0)
        assert _close(crit.anti_collapse(zc), lam_v * (gamma - math.sqrt(eps)))

    # ---- Prop. 5.2 (ii): whitened batch with B > d -> v = 0, c = 0, C positive definite.
    raw = torch.randn(b, d, generator=gen, dtype=torch.float64)
    q, _ = torch.linalg.qr(raw - raw.mean(dim=0, keepdim=True))       # orthonormal, zero-mean columns
    s = gamma * (1.5 + 1.5 * torch.rand(d, generator=gen, dtype=torch.float64))
    z_white = (math.sqrt(b - 1) * q * s).float()                      # C = diag(s^2)
    assert _close(crit.variance(z_white), 0.0) and _close(crit.covariance(z_white), 0.0)
    eig = torch.linalg.eigvalsh(_ref_cov_matrix(z_white))
    assert bool((eig > 0).all()) and bool((eig >= gamma ** 2 - eps).all())
    # B > d is necessary: with B = d, rank C <= B - 1 < d.
    eig_bd = torch.linalg.eigvalsh(_ref_cov_matrix(torch.randn(d, d, generator=gen)))
    assert float(eig_bd.abs().min()) < 1e-10

    # ---- Eq. 11: total = invariance + lambda_v v + lambda_c c; stop-gradient on the target.
    for reduction in ("mean", "sum"):
        crit_r = VICRegLoss(lambda_v=lam_v, lambda_c=lam_c, gamma=gamma, eps=eps, head_reduction=reduction)
        pred.zero_grad(set_to_none=True)
        z_on = z_t.clone().requires_grad_(True)
        target = torch.randn(b, d, generator=gen).requires_grad_(True)
        hp = pred(z_on, u_t).head_predictions
        total, terms = crit_r(z_on, hp, target)
        inv_ref = _ref_invariance(hp, target, reduction)
        v_ref, c_ref = _ref_v(z_on, gamma, eps), _ref_c(z_on)
        assert _close(terms["invariance"], inv_ref)
        assert _close(terms["variance"], v_ref) and _close(terms["covariance"], c_ref)
        assert _close(total, inv_ref + lam_v * v_ref + lam_c * c_ref)
        assert all(t.grad_fn is None for t in terms.values())
        total.backward()
        assert target.grad is None                                    # sg(.)
        assert z_on.grad is not None
        assert all(p.grad is not None for p in pred.parameters())
    # Single predictor [B, d] is the N_e = 1 case.
    assert _close(crit.invariance(heads[0], z_t), _ref_invariance(heads[:1], z_t, "mean"))

    # ---- Errors.
    for kw in (dict(eps=1.0), dict(eps=2.0), dict(eps=0.0), dict(gamma=0.0), dict(gamma=-1.0),
               dict(lambda_v=0.0), dict(lambda_v=-1.0), dict(lambda_c=-0.1), dict(head_reduction="max")):
        args = dict(lambda_v=1.0, lambda_c=1.0, gamma=1.0, eps=1e-4)
        args.update(kw)
        _expect(ValueError, lambda a=args: VICRegLoss(**a))
    one = torch.randn(1, d)
    _expect(ValueError, lambda: crit.variance(one))
    _expect(ValueError, lambda: crit.covariance(one))
    _expect(ValueError, lambda: crit(one, one, one))
    _expect(ValueError, lambda: crit.variance(torch.randn(2, b, d)))
    _expect(ValueError, lambda: crit.invariance(heads, z_t[:-1]))
    _expect(ValueError, lambda: JEPAPredictor(d, nu, num_heads=1))
    _expect(ValueError, lambda: JEPAPredictor(d, nu, num_hidden_layers=0))
    print("predictor.py self-test passed")


if __name__ == "__main__":
    _self_test()
