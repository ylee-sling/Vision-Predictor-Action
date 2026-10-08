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

        Args:
            z_t: ``[B, d]``;  u_t: ``[B]`` int64.

        Returns:
            ``PredictorOutput(z_hat [B, d], sigma [B], head_predictions [N_e, B, d])``.
        """
        heads = self.forward_heads(z_t, u_t)  # [N_e, B, d]
        z_hat = heads.mean(dim=0)             # [B, d]
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
def _self_test() -> None:
    torch.manual_seed(0)
    b, d, nu, ne = 32, 8, 4, 5
    pred = JEPAPredictor(d, nu, num_heads=ne, primitive_embed_dim=6, hidden_dim=32, num_hidden_layers=2)
    z_t, u_t = torch.randn(b, d), torch.randint(0, nu, (b,))

    heads = pred.forward_heads(z_t, u_t)
    assert heads.shape == (ne, b, d)
    # Heads are independent networks: head i must equal a loop over its own parameters.
    for i in range(ne):
        x = torch.cat([z_t, pred.primitive_embedding[i, u_t]], dim=-1)
        for layer in pred.heads:
            x = x @ layer.weight[i] + layer.bias[i] if isinstance(layer, EnsembleLinear) else layer(x)
        assert torch.allclose(x, heads[i], atol=1e-5)
    assert not torch.allclose(heads[0], heads[1])  # independent initialisations

    try:
        pred(z_t, u_t)
        raise AssertionError("sigma must require gamma*")
    except RuntimeError:
        pass
    gamma_star = 1.7
    pred.set_latent_scale(gamma_star)
    out = pred(z_t, u_t)
    assert out.z_hat.shape == (b, d) and out.sigma.shape == (b,)
    ref = torch.stack([
        torch.sqrt(sum(((heads[i, k] - heads[:, k].mean(0)) ** 2).sum() for i in range(ne)) / ne)
        / (math.sqrt(d) * gamma_star)
        for k in range(b)
    ])
    assert torch.allclose(out.sigma, ref, atol=1e-5) and bool((out.sigma >= 0).all())

    # VICReg terms against direct formulas.
    crit = VICRegLoss(lambda_v=25.0, lambda_c=1.0, gamma=1.0, eps=1e-4)
    z = torch.randn(b, d) * 0.5
    std = torch.sqrt(z.var(0, unbiased=True) + 1e-4)
    assert torch.allclose(crit.variance(z), torch.clamp(1.0 - std, min=0).mean())
    zc = z - z.mean(0)
    cov = sum(torch.outer(zc[k], zc[k]) for k in range(b)) / (b - 1)
    off = sum(cov[i, j] ** 2 for i in range(d) for j in range(d) if i != j) / d
    assert torch.allclose(crit.covariance(z), off, atol=1e-6)
    # Collapsed batch: v = gamma - sqrt(eps), c = 0  (Prop. 5.2 (i)).
    zeta = torch.ones(b, d) * 3.0
    assert torch.allclose(crit.variance(zeta), torch.tensor(1.0 - math.sqrt(1e-4)))
    assert torch.allclose(crit.covariance(zeta), torch.tensor(0.0))

    target = torch.randn(b, d, requires_grad=True)
    total, terms = crit(z_t, out.head_predictions, target)
    inv = ((out.head_predictions - target.detach()) ** 2).sum(-1).mean(-1).mean()
    assert torch.allclose(terms["invariance"], inv.detach())
    assert torch.allclose(total, inv + 25.0 * crit.variance(z_t) + crit.covariance(z_t))
    total.backward()
    assert target.grad is None  # stop-gradient on the target branch
    print("predictor.py self-test passed")


if __name__ == "__main__":
    _self_test()
