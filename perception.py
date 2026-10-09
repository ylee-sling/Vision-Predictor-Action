# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
"""
perception.py -- Stage 1 of the Goal-Conditioned VPA framework (preprint_261008.pdf, Sec. 4.1).

Implements
    * ``VisionEncoder``      -- the shared (Siamese) vision encoder E_psi that maps observation
                                frames I_t and milestone frames I_g^(m) into the latent space
                                Z subset R^d (Eq. 7):
                                    z_t = E_psi(I_t),   z_g^(m) = E_psi(I_g^(m)),  m = 1..M.
    * ``MomentumEncoder``    -- the momentum (exponential-moving-average) copy E_psi_bar that
                                produces the stop-gradient target z_{t+1} = sg(E_psi_bar(I_{t+1}))
                                of the JEPA objective (Eq. 11, Fig. 3).
    * ``TextEncoderWrapper`` -- the lightweight *frozen* text encoder (e.g. CLIP) that embeds the
                                natural-language instruction x into c_text in R^{d_c}.

Layout convention
    The paper writes I_t in R^{H_img x W_img x C}. PyTorch convolutions are channels-first, so
    every image tensor in this module is ``[Batch, C, H_img, W_img]``; a milestone sequence is
    ``[Batch, M, C, H_img, W_img]``. With V >= 2 camera views (``VisionEncoder(num_views=V)``), a
    view axis follows the batch (and milestone) axes: ``[Batch, V, C, H_img, W_img]`` and
    ``[Batch, M, V, C, H_img, W_img]``. With one camera there is no view axis.

This file is self-contained (it imports nothing from the other VPA modules) and can be tested
on its own with ``python perception.py``.
"""

from __future__ import annotations

import copy
from types import SimpleNamespace
from typing import Any, Callable, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor

__all__ = [
    "PatchEmbedding",
    "TransformerBlock",
    "VisionEncoder",
    "MomentumEncoder",
    "TextEncoderWrapper",
]

ImageSize = Union[int, Tuple[int, int]]


def _pair(x: ImageSize) -> Tuple[int, int]:
    return (x, x) if isinstance(x, int) else (int(x[0]), int(x[1]))


# ---------------------------------------------------------------------------------------------
# Vision Transformer building blocks
# ---------------------------------------------------------------------------------------------
class PatchEmbedding(nn.Module):
    """Non-overlapping patch tokenizer of a Vision Transformer.

    Args:
        image_size: (H_img, W_img) or a single int for square frames.
        patch_size: side length P of the square patches; must divide H_img and W_img.
        in_channels: number of image channels C.
        width: token width D_vit.

    Shapes:
        input  ``images``: ``[B, C, H_img, W_img]``
        output ``tokens``: ``[B, N_patch, D_vit]`` with N_patch = (H_img / P) * (W_img / P)
    """

    def __init__(self, image_size: ImageSize, patch_size: int, in_channels: int, width: int) -> None:
        super().__init__()
        h, w = _pair(image_size)
        if h % patch_size != 0 or w % patch_size != 0:
            raise ValueError(f"image size {(h, w)} is not divisible by patch size {patch_size}")
        self.image_size: Tuple[int, int] = (h, w)
        self.patch_size: int = patch_size
        self.num_patches: int = (h // patch_size) * (w // patch_size)
        self.proj = nn.Conv2d(in_channels, width, kernel_size=patch_size, stride=patch_size)

    def forward(self, images: Tensor) -> Tensor:
        """``[B, C, H_img, W_img]`` -> ``[B, N_patch, D_vit]``."""
        x = self.proj(images)  # [B, D_vit, H_img/P, W_img/P]
        return x.flatten(2).transpose(1, 2)  # [B, N_patch, D_vit]


class TransformerBlock(nn.Module):
    """Pre-norm ViT block: x <- x + MHSA(LN(x));  x <- x + MLP(LN(x)).

    Only LayerNorm is used (no BatchNorm), so every sample in a batch is processed independently.
    This is what makes Siamese batching of observations and milestones exact (see
    ``VisionEncoder.encode_siamese``).

    Shapes:
        input/output ``x``: ``[B, N_tok, D_vit]``
    """

    def __init__(self, width: int, num_heads: int, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        if width % num_heads != 0:
            raise ValueError(f"width {width} must be divisible by num_heads {num_heads}")
        hidden = int(round(width * mlp_ratio))
        self.norm1 = nn.LayerNorm(width)
        self.attn = nn.MultiheadAttention(width, num_heads, dropout=0.0, batch_first=True)
        self.norm2 = nn.LayerNorm(width)
        self.mlp = nn.Sequential(nn.Linear(width, hidden), nn.GELU(), nn.Linear(hidden, width))

    def forward(self, x: Tensor) -> Tensor:
        """``[B, N_tok, D_vit]`` -> ``[B, N_tok, D_vit]``."""
        h = self.norm1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        x = x + self.mlp(self.norm2(x))
        return x


# ---------------------------------------------------------------------------------------------
# E_psi : shared Siamese vision encoder (Eq. 7)
# ---------------------------------------------------------------------------------------------
class VisionEncoder(nn.Module):
    """Shared vision encoder E_psi (Vision Transformer backbone, Sec. 4.1, Eq. 7).

    A single set of weights psi embeds both the current observation and every milestone goal
    frame, i.e. the encoder is used as a Siamese network:

        z_t     = E_psi(I_t)          in R^d
        z_g^(m) = E_psi(I_g^(m))      in R^d,   m = 1, ..., M

    Single camera (``num_views = 1``, the paper's setting): the latent is the linear projection of
    the final [CLS] token after the last LayerNorm.

    Several cameras (``num_views = V >= 2``; a decision where the paper is silent). The paper writes
    I_t as one frame. With V cameras, I_t is the ordered tuple of the V views (I_t^1, ..., I_t^V),
    and every milestone I_g^(m) is the tuple of the same views in the same order. One ViT body
    (shared by all views, as it is shared by observations and milestones) yields the final
    LayerNorm'd [CLS] token f(I^v) of each view, and the head maps their concatenation in view order:

        E_psi(I) = W_head Concat(f(I^1), ..., f(I^V)) + b_head,    W_head in R^{d x V D_vit}

    So E_psi is still a single map from an observation to R^d, and everything downstream of z
    (Eqs. 8-15, 18, 19) is unchanged. With V = 1 this is exactly the single-camera encoder: no extra
    module, the same parameter names and shapes, and the same frame shape ``[C, H_img, W_img]``.

    Args:
        image_size: (H_img, W_img) or int.
        patch_size: ViT patch size P.
        in_channels: image channels C.
        width: ViT token width D_vit.
        depth: number of transformer blocks.
        num_heads: attention heads per block.
        latent_dim: dimension d of the latent space Z.
        mlp_ratio: hidden width multiplier of the block MLPs.
        num_views: number of camera views V per frame (1 = single camera).

    Shapes (``frame_shape`` = ``[C, H_img, W_img]`` if V = 1, ``[V, C, H_img, W_img]`` if V >= 2):
        ``forward``:          ``[B, *frame_shape]`` -> ``[B, d]``
        ``encode_milestones``: ``[B, M, *frame_shape]`` -> ``[B, M, d]``
        ``encode_siamese``:   (``[B, *frame_shape]``, ``[B, M, *frame_shape]``)
                              -> (``[B, d]``, ``[B, M, d]``)
    """

    def __init__(
        self,
        image_size: ImageSize = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        width: int = 384,
        depth: int = 6,
        num_heads: int = 6,
        latent_dim: int = 256,
        mlp_ratio: float = 4.0,
        num_views: int = 1,
    ) -> None:
        super().__init__()
        if int(num_views) != num_views or int(num_views) < 1:
            raise ValueError(f"num_views must be an integer >= 1, got {num_views}")
        self.in_channels: int = in_channels
        self.latent_dim: int = latent_dim
        self.num_views: int = int(num_views)
        self.patch_embed = PatchEmbedding(image_size, patch_size, in_channels, width)
        self.image_size: Tuple[int, int] = self.patch_embed.image_size
        self.cls_token = nn.Parameter(torch.zeros(1, 1, width))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches + 1, width))
        self.blocks = nn.ModuleList(
            [TransformerBlock(width, num_heads, mlp_ratio) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(width)
        self.head = nn.Linear(self.num_views * width, latent_dim)  # V = 1: Linear(D_vit, d) as before
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    @property
    def frame_shape(self) -> Tuple[int, ...]:
        """Shape of one observation: ``(C, H_img, W_img)`` if V = 1, ``(V, C, H_img, W_img)`` if V >= 2."""
        c, (h, w) = self.in_channels, self.image_size
        return (c, h, w) if self.num_views == 1 else (self.num_views, c, h, w)

    def _check_frames(self, images: Tensor) -> None:
        expected = self.frame_shape
        if images.dim() != 1 + len(expected) or tuple(images.shape[1:]) != expected:
            layout = "[B, C, H, W]" if self.num_views == 1 else "[B, V, C, H, W]"
            raise ValueError(
                f"expected frames {layout} = [B, {', '.join(str(s) for s in expected)}], "
                f"got {tuple(images.shape)}"
            )

    def _cls_features(self, images: Tensor) -> Tensor:
        """ViT body on single-view frames: ``[N, C, H_img, W_img]`` -> ``[N, D_vit]``.

        Returns the final LayerNorm'd [CLS] token. A plain method, not a submodule, so a
        multi-view forward is still one evaluation of E_psi for the depth audit (Prop. 5.1).
        """
        x = self.patch_embed(images)  # [N, N_patch, D_vit]
        cls = self.cls_token.expand(x.shape[0], -1, -1)  # [N, 1, D_vit]
        x = torch.cat([cls, x], dim=1) + self.pos_embed  # [N, N_patch + 1, D_vit]
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return x[:, 0]  # [N, D_vit]

    def forward(self, images: Tensor) -> Tensor:
        """Encode a batch of observations (Eq. 7).

        Args:
            images: ``[B, C, H_img, W_img]`` (V = 1) or ``[B, V, C, H_img, W_img]`` (V >= 2) float
                frames (observations and/or milestone goals).

        Returns:
            z: ``[B, d]`` latent embeddings E_psi(I).
        """
        self._check_frames(images)
        if self.num_views == 1:
            return self.head(self._cls_features(images))  # [B, d]
        b = images.shape[0]
        # all B * V views through the shared body in one batch (row b * V + v is view v of sample b)
        cls = self._cls_features(images.flatten(0, 1))  # [B * V, D_vit]
        return self.head(cls.reshape(b, self.num_views * cls.shape[-1]))  # Concat in view order -> [B, d]

    def encode_milestones(self, milestones: Tensor) -> Tensor:
        """Encode an ordered milestone sequence (computed once per episode, Sec. 4.1).

        Args:
            milestones: ``[B, M, *frame_shape]`` -- (I_g^(1), ..., I_g^(M)) per batch element.

        Returns:
            z_g: ``[B, M, d]`` -- (z_g^(1), ..., z_g^(M)).
        """
        if milestones.dim() != 2 + len(self.frame_shape):
            layout = "[B, M, C, H, W]" if self.num_views == 1 else "[B, M, V, C, H, W]"
            raise ValueError(f"expected milestones of shape {layout}, got {tuple(milestones.shape)}")
        b, m = milestones.shape[:2]
        z = self(milestones.reshape(b * m, *milestones.shape[2:]))  # [B*M, d]
        return z.reshape(b, m, self.latent_dim)

    def encode_siamese(self, observation: Tensor, milestones: Tensor) -> Tuple[Tensor, Tensor]:
        """Siamese batching: encode observations and milestones in ONE forward pass of E_psi.

        Both streams are concatenated along the batch axis, passed through the shared weights,
        and split again. Because the network contains no cross-sample operation (LayerNorm only),
        the result is identical to encoding the two streams separately.

        Args:
            observation: ``[B, *frame_shape]`` -- current frames I_t.
            milestones:  ``[B, M, *frame_shape]`` -- milestone frames I_g^(1..M).

        Returns:
            z_t: ``[B, d]``;  z_g: ``[B, M, d]``.
        """
        n = len(self.frame_shape)
        if observation.dim() != 1 + n or milestones.dim() != 2 + n:
            if self.num_views == 1:
                raise ValueError("expected observation [B, C, H, W] and milestones [B, M, C, H, W]")
            raise ValueError("expected observation [B, V, C, H, W] and milestones [B, M, V, C, H, W]")
        b, m = milestones.shape[:2]
        if observation.shape[0] != b:
            raise ValueError(f"batch mismatch: observation {observation.shape[0]} vs milestones {b}")
        frames = torch.cat([observation, milestones.reshape(b * m, *milestones.shape[2:])], dim=0)
        z = self(frames)  # [B + B*M, d]
        z_t = z[:b]  # [B, d]
        z_g = z[b:].reshape(b, m, self.latent_dim)  # [B, M, d]
        return z_t, z_g


# ---------------------------------------------------------------------------------------------
# E_psi_bar : momentum target encoder (Eq. 11, Fig. 3)
# ---------------------------------------------------------------------------------------------
class MomentumEncoder(nn.Module):
    """Momentum (EMA) copy E_psi_bar of the online encoder, used only to produce JEPA targets.

    Update rule after each optimizer step (psi_bar <- momentum * psi_bar + (1 - momentum) * psi).
    The paper does not fix the momentum coefficient; it is a constructor argument here.
    The target branch receives no gradients: ``forward`` runs under ``torch.no_grad`` and returns a
    detached tensor, which realises the stop-gradient sg(.) of Eq. 11.

    Shapes:
        ``forward``: ``[B, *frame_shape]`` -> ``[B, d]`` (detached); ``frame_shape`` is the online
        encoder's (``[C, H_img, W_img]``, or ``[V, C, H_img, W_img]`` with V camera views).
    """

    def __init__(self, online_encoder: VisionEncoder, momentum: float = 0.996) -> None:
        super().__init__()
        if not 0.0 <= momentum < 1.0:
            raise ValueError("momentum must lie in [0, 1)")
        self.momentum: float = float(momentum)
        self.encoder: VisionEncoder = copy.deepcopy(online_encoder)
        self.encoder.requires_grad_(False)

    @torch.no_grad()
    def update(self, online_encoder: VisionEncoder) -> None:
        """psi_bar <- momentum * psi_bar + (1 - momentum) * psi (parameters and buffers)."""
        for p_bar, p in zip(self.encoder.parameters(), online_encoder.parameters(), strict=True):
            p_bar.mul_(self.momentum).add_(p.detach(), alpha=1.0 - self.momentum)
        for b_bar, b in zip(self.encoder.buffers(), online_encoder.buffers(), strict=True):
            b_bar.copy_(b)

    @torch.no_grad()
    def forward(self, images: Tensor) -> Tensor:
        """``[B, *frame_shape]`` -> ``[B, d]`` target latents sg(E_psi_bar(I))."""
        return self.encoder(images).detach()


# ---------------------------------------------------------------------------------------------
# Frozen text encoder (c_text)
# ---------------------------------------------------------------------------------------------
class TextEncoderWrapper(nn.Module):
    """Frozen, lightweight text encoder producing the instruction embedding c_text in R^{d_c}.

    By default it loads a pre-trained CLIP text tower with its projection head
    (``transformers.CLIPTextModelWithProjection`` + ``CLIPTokenizer``) and uses the projected
    pooled output (``text_embeds``) as c_text, so d_c = ``config.projection_dim``
    (512 for ``openai/clip-vit-base-patch32``). All parameters are frozen and the module is kept
    in eval mode permanently; c_text is computed once per episode (Sec. 4.1).

    For offline use or tests, a ``text_model`` and ``tokenizer`` can be injected. The injected
    model must accept ``(input_ids, attention_mask)`` and return an object with a
    ``text_embeds`` attribute of shape ``[B, d_c]``; it must expose ``config.projection_dim``.

    Shapes:
        ``forward``:       list of B strings -> ``[B, d_c]``
        ``encode_tokens``: ``input_ids [B, L]`` (+ ``attention_mask [B, L]``) -> ``[B, d_c]``
    """

    def __init__(
        self,
        model_name_or_path: str = "openai/clip-vit-base-patch32",
        text_model: Optional[nn.Module] = None,
        tokenizer: Optional[Callable[..., Any]] = None,
        max_length: int = 77,
    ) -> None:
        super().__init__()
        if text_model is None or tokenizer is None:
            # Imported lazily so the rest of this module works without `transformers` installed.
            from transformers import CLIPTextModelWithProjection, CLIPTokenizer

            if text_model is None:
                text_model = CLIPTextModelWithProjection.from_pretrained(model_name_or_path)
            if tokenizer is None:
                tokenizer = CLIPTokenizer.from_pretrained(model_name_or_path)
        self.text_model: nn.Module = text_model
        self.tokenizer: Callable[..., Any] = tokenizer
        self.max_length: int = int(max_length)
        self.embed_dim: int = int(text_model.config.projection_dim)  # d_c
        self.text_model.requires_grad_(False)
        self.text_model.eval()

    def train(self, mode: bool = True) -> "TextEncoderWrapper":
        """The text encoder is frozen: it stays in eval mode regardless of ``mode``."""
        super().train(mode)
        self.text_model.eval()
        return self

    def _device(self) -> torch.device:
        try:
            return next(self.text_model.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    @torch.no_grad()
    def encode_tokens(self, input_ids: Tensor, attention_mask: Optional[Tensor] = None) -> Tensor:
        """Embed pre-tokenised instructions.

        Args:
            input_ids:      ``[B, L]`` int64 token ids.
            attention_mask: ``[B, L]`` (optional) 1 for real tokens, 0 for padding.

        Returns:
            c_text: ``[B, d_c]``.
        """
        out = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
        c_text = out.text_embeds if hasattr(out, "text_embeds") else out
        if c_text.dim() != 2 or c_text.shape[-1] != self.embed_dim:
            raise RuntimeError(f"text model returned {tuple(c_text.shape)}, expected [B, {self.embed_dim}]")
        return c_text

    @torch.no_grad()
    def forward(self, instructions: Union[str, Sequence[str]]) -> Tensor:
        """Embed natural-language instructions x.

        Args:
            instructions: one string or a sequence of B strings.

        Returns:
            c_text: ``[B, d_c]`` (B = 1 for a single string).
        """
        texts = [instructions] if isinstance(instructions, str) else list(instructions)
        if len(texts) == 0:
            raise ValueError("at least one instruction is required")
        tokens = self.tokenizer(
            texts, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt"
        )
        device = self._device()
        attention_mask = tokens.get("attention_mask")
        return self.encode_tokens(
            tokens["input_ids"].to(device),
            attention_mask.to(device) if attention_mask is not None else None,
        )


# ---------------------------------------------------------------------------------------------
# Self-test (run: python perception.py)
# ---------------------------------------------------------------------------------------------
class _ToyTextModel(nn.Module):
    """Tiny stand-in with the CLIPTextModelWithProjection interface (offline testing only)."""

    def __init__(self, vocab: int = 100, width: int = 32, projection_dim: int = 24) -> None:
        super().__init__()
        self.config = SimpleNamespace(projection_dim=projection_dim)
        self.embed = nn.Embedding(vocab, width)
        self.proj = nn.Linear(width, projection_dim, bias=False)

    def forward(self, input_ids: Tensor, attention_mask: Optional[Tensor] = None) -> SimpleNamespace:
        h = self.embed(input_ids)
        if attention_mask is not None:
            h = h * attention_mask.unsqueeze(-1)
        return SimpleNamespace(text_embeds=self.proj(h.mean(1)))


def _toy_tokenizer(texts: Sequence[str], max_length: int = 16, **_: Any) -> dict:
    ids = torch.zeros(len(texts), max_length, dtype=torch.long)
    mask = torch.zeros(len(texts), max_length, dtype=torch.long)
    for i, text in enumerate(texts):
        codes = [ord(ch) % 100 for ch in text][:max_length]
        ids[i, : len(codes)] = torch.tensor(codes, dtype=torch.long)
        mask[i, : len(codes)] = 1
    return {"input_ids": ids, "attention_mask": mask}


_RTOL, _ATOL = 1e-5, 1e-6  # float32 module output vs float64 reference (CLAUDE.md)


def _close(actual: Tensor, ref: Tensor) -> bool:
    return torch.allclose(actual.double(), ref.double(), rtol=_RTOL, atol=_ATOL)


def _expect_value_error(fn: Callable[[], Any]) -> None:
    try:
        fn()
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def _ref_layer_norm(x: Tensor, weight: Tensor, bias: Tensor, eps: float) -> Tensor:
    """LayerNorm over the last axis, written out: (x - mean) / sqrt(biased var + eps) * w + b."""
    mean = x.sum(-1, keepdim=True) / x.shape[-1]
    var = ((x - mean) ** 2).sum(-1, keepdim=True) / x.shape[-1]
    return (x - mean) / torch.sqrt(var + eps) * weight + bias


def _ref_vit_cls(enc: VisionEncoder, images: Tensor) -> Tensor:
    """Independent float64 reference of the ViT body on single-view frames ``[B, C, H, W]``.

    Re-derives the ViT forward from its definition with float64 copies of the module weights:
    explicit patch loops, per-head softmax attention, exact erf-GELU, hand-written LayerNorm.
    Returns the final LayerNorm'd [CLS] token, ``[B, D_vit]`` float64.
    """
    f64 = {k: v.detach().double() for k, v in enc.state_dict().items()}
    x = images.double()
    bsz, _, h, w = x.shape
    p = enc.patch_embed.patch_size
    w_patch = f64["patch_embed.proj.weight"].reshape(f64["patch_embed.proj.weight"].shape[0], -1)
    tokens = []
    for i in range(h // p):  # row-major patch order
        for j in range(w // p):
            patch = x[:, :, i * p:(i + 1) * p, j * p:(j + 1) * p].reshape(bsz, -1)  # [B, C*P*P]
            tokens.append(patch @ w_patch.T + f64["patch_embed.proj.bias"])
    t = torch.stack(tokens, dim=1)  # [B, N_patch, D]
    t = torch.cat([f64["cls_token"].expand(bsz, -1, -1), t], dim=1) + f64["pos_embed"]
    width = t.shape[-1]
    for n, block in enumerate(enc.blocks):
        pre = f"blocks.{n}."
        heads = block.attn.num_heads
        dh = width // heads
        hn = _ref_layer_norm(t, f64[pre + "norm1.weight"], f64[pre + "norm1.bias"], block.norm1.eps)
        w_in, b_in = f64[pre + "attn.in_proj_weight"], f64[pre + "attn.in_proj_bias"]
        q = hn @ w_in[:width].T + b_in[:width]
        k = hn @ w_in[width:2 * width].T + b_in[width:2 * width]
        v = hn @ w_in[2 * width:].T + b_in[2 * width:]
        head_out = []
        for a in range(heads):
            sl = slice(a * dh, (a + 1) * dh)
            scores = q[..., sl] @ k[..., sl].transpose(1, 2) / dh ** 0.5  # [B, N, N]
            e = torch.exp(scores - scores.max(dim=-1, keepdim=True).values)
            head_out.append((e / e.sum(dim=-1, keepdim=True)) @ v[..., sl])
        attn = torch.cat(head_out, dim=-1) @ f64[pre + "attn.out_proj.weight"].T + f64[pre + "attn.out_proj.bias"]
        t = t + attn
        hn = _ref_layer_norm(t, f64[pre + "norm2.weight"], f64[pre + "norm2.bias"], block.norm2.eps)
        hid = hn @ f64[pre + "mlp.0.weight"].T + f64[pre + "mlp.0.bias"]
        hid = 0.5 * hid * (1.0 + torch.erf(hid / 2.0 ** 0.5))  # exact GELU
        t = t + hid @ f64[pre + "mlp.2.weight"].T + f64[pre + "mlp.2.bias"]
    t = _ref_layer_norm(t, f64["norm.weight"], f64["norm.bias"], enc.norm.eps)
    return t[:, 0]  # [B, D_vit]


def _ref_vit_forward(enc: VisionEncoder, images: Tensor) -> Tensor:
    """Independent float64 reference of z = E_psi(I) (Eq. 7).

    ``[B, C, H, W]`` (one camera): the head applied to the [CLS] token.
    ``[B, V, C, H, W]`` (V cameras): the [CLS] tokens of the V views, each from its own body pass,
    concatenated in view order, then the head. Returns ``[B, d]`` float64.
    """
    w_head = enc.head.weight.detach().double()
    b_head = enc.head.bias.detach().double()
    if images.dim() == 4:
        feats = _ref_vit_cls(enc, images)  # [B, D_vit]
    else:
        feats = torch.cat([_ref_vit_cls(enc, images[:, v]) for v in range(images.shape[1])], dim=-1)  # [B, V*D_vit]
    return feats @ w_head.T + b_head  # [B, d]


def _self_test() -> None:
    torch.manual_seed(0)
    gen = torch.Generator().manual_seed(1234)
    b, m, c, h, w, d = 2, 3, 3, 32, 32, 16  # B != M so a B/M transposition cannot pass
    enc = VisionEncoder(image_size=(h, w), patch_size=8, in_channels=c, width=32, depth=2,
                        num_heads=4, latent_dim=d).eval()
    obs = torch.randn(b, c, h, w, generator=gen)
    goals = torch.randn(b, m, c, h, w, generator=gen)

    # ---- Eq. 7: z_t = E_psi(I_t), z_g^(m) = E_psi(I_g^(m)) against the float64 reference.
    with torch.no_grad():
        z_t = enc(obs)
        z_g = enc.encode_milestones(goals)
        zs_t, zs_g = enc.encode_siamese(obs, goals)
    assert z_t.shape == (b, d) and z_g.shape == (b, m, d)
    assert zs_t.shape == (b, d) and zs_g.shape == (b, m, d)
    ref_t = _ref_vit_forward(enc, obs)
    assert _close(z_t, ref_t)
    for bi in range(b):
        for mi in range(m):
            ref_g = _ref_vit_forward(enc, goals[bi, mi].unsqueeze(0))[0]
            assert _close(z_g[bi, mi], ref_g)
            assert _close(zs_g[bi, mi], ref_g)
    assert _close(zs_t, ref_t)
    # Siamese batching (one pass, shared psi) == encoding the two streams separately.
    assert torch.allclose(zs_t, z_t, rtol=_RTOL, atol=_ATOL)
    assert torch.allclose(zs_g, z_g, rtol=_RTOL, atol=_ATOL)
    # B = 1: no cross-sample statistics.
    with torch.no_grad():
        z1_t, z1_g = enc.encode_siamese(obs[:1], goals[:1])
    assert z1_t.shape == (1, d) and z1_g.shape == (1, m, d)
    assert _close(z1_t, ref_t[:1]) and torch.allclose(z1_g, z_g[:1], rtol=_RTOL, atol=_ATOL)

    # ---- Malformed frames raise ValueError.
    _expect_value_error(lambda: enc(torch.randn(b, c + 1, h, w)))          # wrong C
    _expect_value_error(lambda: enc(torch.randn(b, c, h + 8, w)))          # wrong H
    _expect_value_error(lambda: enc(torch.randn(b, c, h, w + 8)))          # wrong W
    _expect_value_error(lambda: enc(torch.randn(c, h, w)))                 # 3-D
    _expect_value_error(lambda: enc(torch.randn(b, m, c, h, w)))           # 5-D into forward
    _expect_value_error(lambda: enc.encode_milestones(torch.randn(b, c, h, w)))          # 4-D
    _expect_value_error(lambda: enc.encode_milestones(torch.randn(b, m, c, h, w + 8)))   # frame size
    _expect_value_error(lambda: enc.encode_siamese(torch.randn(b + 1, c, h, w), goals))  # batch mismatch
    _expect_value_error(lambda: enc.encode_siamese(torch.randn(c, h, w), goals))         # 3-D obs
    _expect_value_error(lambda: enc.encode_siamese(obs, torch.randn(b, c, h, w)))        # 4-D milestones

    # ---- Momentum target E_psi_bar (Eq. 11): EMA psi_bar <- mu psi_bar + (1 - mu) psi.
    mu = 0.9
    target = MomentumEncoder(enc, momentum=mu)
    assert not any(p.requires_grad for p in target.parameters())
    for p_bar, p in zip(target.encoder.parameters(), enc.parameters(), strict=True):
        assert torch.equal(p_bar, p) and p_bar.data_ptr() != p.data_ptr()  # copy, not alias
    psi_bar_ref = [p.detach().double().clone() for p in enc.parameters()]
    for _ in range(5):
        with torch.no_grad():  # simulated optimizer step on the online encoder
            for p in enc.parameters():
                p.add_(0.1 * torch.randn(p.shape, generator=gen))
        psi_before = [p.detach().clone() for p in enc.parameters()]
        target.update(enc)
        for i, p in enumerate(enc.parameters()):
            psi_bar_ref[i] = mu * psi_bar_ref[i] + (1.0 - mu) * p.detach().double()
        for p_bar, ref, p, p_old in zip(
            target.encoder.parameters(), psi_bar_ref, enc.parameters(), psi_before, strict=True
        ):
            assert _close(p_bar, ref)
            assert torch.equal(p, p_old)  # update() never writes to the online encoder
    # E_psi_bar is the Eq. 7 map evaluated with the EMA weights.
    assert _close(target(obs), _ref_vit_forward(target.encoder, obs))
    # Stop-gradient sg(.): no graph from the target branch, gradients only reach psi.
    obs_g = obs.clone().requires_grad_(True)
    z_tgt = target(obs_g)
    assert not z_tgt.requires_grad and z_tgt.grad_fn is None
    enc.zero_grad(set_to_none=True)
    loss = ((enc(obs_g) - z_tgt) ** 2).sum()
    loss.backward()
    assert all(p.grad is not None for p in enc.parameters())
    assert all(p.grad is None for p in target.parameters())
    _expect_value_error(lambda: MomentumEncoder(enc, momentum=1.0))
    _expect_value_error(lambda: MomentumEncoder(enc, momentum=-0.1))

    # ---- Frozen text encoder c_text in R^{d_c} (offline stand-in with the CLIP interface).
    text = TextEncoderWrapper(text_model=_ToyTextModel(), tokenizer=_toy_tokenizer)
    for mode in (None, True):
        text.train() if mode is None else text.train(mode)
        assert not text.text_model.training
        assert not any(mod.training for mod in text.text_model.modules())
    assert not any(p.requires_grad for p in text.text_model.parameters())
    texts = ["move the cup", "throw the cup away", "push the block"]
    with torch.enable_grad():
        c_text = text(texts)
        c_one = text("move the cup")
    assert c_text.shape == (len(texts), text.embed_dim) and c_one.shape == (1, text.embed_dim)
    assert not c_text.requires_grad and c_text.grad_fn is None
    tokens = _toy_tokenizer(texts, padding=True, truncation=True, max_length=text.max_length,
                            return_tensors="pt")
    with torch.no_grad():
        direct = text.text_model(input_ids=tokens["input_ids"],
                                 attention_mask=tokens["attention_mask"]).text_embeds
    assert torch.equal(c_text, direct)  # wrapper passes ids and mask through unchanged
    assert torch.allclose(c_one[0], c_text[0], rtol=_RTOL, atol=_ATOL)  # B=1 vs B=3 BLAS rounding
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            assert not torch.allclose(c_text[i], c_text[j])
    _expect_value_error(lambda: text([]))

    # ---- Several cameras (num_views = V = 2): one shared ViT body for every view, the head maps the
    # concatenated [CLS] tokens in view order. Own generator, so the checks above are unaffected.
    gen2 = torch.Generator().manual_seed(4321)
    v = 2
    enc2 = VisionEncoder(image_size=(h, w), patch_size=8, in_channels=c, width=32, depth=2,
                         num_heads=4, latent_dim=d, num_views=v).eval()
    assert enc.num_views == 1 and enc.frame_shape == (c, h, w)
    assert enc2.num_views == v and enc2.frame_shape == (v, c, h, w)
    assert enc.head.in_features == 32 and enc2.head.in_features == v * 32
    # the same parameters as the single-camera encoder, except the head's input width
    shapes1 = {k: tuple(p.shape) for k, p in enc.named_parameters()}
    shapes2 = {k: tuple(p.shape) for k, p in enc2.named_parameters()}
    assert shapes1.keys() == shapes2.keys()
    assert all(shapes1[k] == shapes2[k] for k in shapes1 if not k.startswith("head."))
    obs2 = torch.randn(b, v, c, h, w, generator=gen2)
    goals2 = torch.randn(b, m, v, c, h, w, generator=gen2)
    with torch.no_grad():
        z2_t = enc2(obs2)
        z2_g = enc2.encode_milestones(goals2)
        zs2_t, zs2_g = enc2.encode_siamese(obs2, goals2)
        z2_swapped = enc2(obs2.flip(1))                                   # views in the other order
        z2_single = enc2(obs2[1:2])                                       # B = 1
    assert z2_t.shape == (b, d) and z2_g.shape == (b, m, d) and zs2_t.shape == (b, d) and zs2_g.shape == (b, m, d)
    ref2_t = _ref_vit_forward(enc2, obs2)
    assert _close(z2_t, ref2_t) and _close(zs2_t, ref2_t) and _close(z2_single, ref2_t[1:2])
    for bi in range(b):
        for mi in range(m):
            ref_g = _ref_vit_forward(enc2, goals2[bi, mi].unsqueeze(0))[0]
            assert _close(z2_g[bi, mi], ref_g) and _close(zs2_g[bi, mi], ref_g)
    # the head sees the views in camera order: swapping them changes z
    assert not torch.allclose(z2_swapped, z2_t, rtol=1e-3, atol=1e-4)
    assert _close(z2_swapped, _ref_vit_forward(enc2, obs2.flip(1)))
    # a view axis is required with V >= 2 and rejected with V = 1; the view count must match
    _expect_value_error(lambda: enc2(obs))                                          # no view axis
    _expect_value_error(lambda: enc2(torch.randn(b, v + 1, c, h, w)))               # wrong V
    _expect_value_error(lambda: enc2(torch.randn(b, v, c, h + 8, w)))               # wrong frame size
    _expect_value_error(lambda: enc2.encode_milestones(goals))                      # [B, M, C, H, W]
    _expect_value_error(lambda: enc2.encode_siamese(obs2, goals))                   # single-view milestones
    _expect_value_error(lambda: enc2.encode_siamese(obs, goals2))                   # single-view observation
    _expect_value_error(lambda: enc(obs2))                                          # V = 1 rejects a view axis
    _expect_value_error(lambda: enc.encode_milestones(goals2))
    _expect_value_error(lambda: VisionEncoder(image_size=(h, w), patch_size=8, num_views=0))
    # E_psi_bar of a multi-view encoder: the same map with the EMA weights; gradients reach every view
    target2 = MomentumEncoder(enc2, momentum=mu)
    assert _close(target2(obs2), _ref_vit_forward(target2.encoder, obs2))
    obs2_g = obs2.clone().requires_grad_(True)
    enc2.zero_grad(set_to_none=True)
    (enc2(obs2_g) - target2(obs2_g.flip(1))).pow(2).sum().backward()               # non-zero residual
    assert obs2_g.grad is not None and all(bool(obs2_g.grad[:, vi].abs().sum() > 0) for vi in range(v))
    assert all(p.grad is not None for p in enc2.parameters())
    assert all(p.grad is None for p in target2.parameters())
    print("perception.py self-test passed")


if __name__ == "__main__":
    _self_test()
