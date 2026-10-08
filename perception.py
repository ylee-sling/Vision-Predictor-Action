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
    ``[Batch, M, C, H_img, W_img]``.

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

    The latent is the linear projection of the final [CLS] token after the last LayerNorm.

    Args:
        image_size: (H_img, W_img) or int.
        patch_size: ViT patch size P.
        in_channels: image channels C.
        width: ViT token width D_vit.
        depth: number of transformer blocks.
        num_heads: attention heads per block.
        latent_dim: dimension d of the latent space Z.
        mlp_ratio: hidden width multiplier of the block MLPs.

    Shapes:
        ``forward``:          ``[B, C, H_img, W_img]`` -> ``[B, d]``
        ``encode_milestones``: ``[B, M, C, H_img, W_img]`` -> ``[B, M, d]``
        ``encode_siamese``:   (``[B, C, H_img, W_img]``, ``[B, M, C, H_img, W_img]``)
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
    ) -> None:
        super().__init__()
        self.in_channels: int = in_channels
        self.latent_dim: int = latent_dim
        self.patch_embed = PatchEmbedding(image_size, patch_size, in_channels, width)
        self.image_size: Tuple[int, int] = self.patch_embed.image_size
        self.cls_token = nn.Parameter(torch.zeros(1, 1, width))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches + 1, width))
        self.blocks = nn.ModuleList(
            [TransformerBlock(width, num_heads, mlp_ratio) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, latent_dim)
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

    def _check_frames(self, images: Tensor) -> None:
        if images.dim() != 4:
            raise ValueError(f"expected images of shape [B, C, H, W], got {tuple(images.shape)}")
        _, c, h, w = images.shape
        if c != self.in_channels or (h, w) != self.image_size:
            raise ValueError(
                f"expected frames [B, {self.in_channels}, {self.image_size[0]}, {self.image_size[1]}], "
                f"got {tuple(images.shape)}"
            )

    def forward(self, images: Tensor) -> Tensor:
        """Encode a batch of frames.

        Args:
            images: ``[B, C, H_img, W_img]`` float frames (observations and/or milestone goals).

        Returns:
            z: ``[B, d]`` latent embeddings E_psi(I).
        """
        self._check_frames(images)
        x = self.patch_embed(images)  # [B, N_patch, D_vit]
        cls = self.cls_token.expand(x.shape[0], -1, -1)  # [B, 1, D_vit]
        x = torch.cat([cls, x], dim=1) + self.pos_embed  # [B, N_patch + 1, D_vit]
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return self.head(x[:, 0])  # [B, d]

    def encode_milestones(self, milestones: Tensor) -> Tensor:
        """Encode an ordered milestone sequence (computed once per episode, Sec. 4.1).

        Args:
            milestones: ``[B, M, C, H_img, W_img]`` -- (I_g^(1), ..., I_g^(M)) per batch element.

        Returns:
            z_g: ``[B, M, d]`` -- (z_g^(1), ..., z_g^(M)).
        """
        if milestones.dim() != 5:
            raise ValueError(
                f"expected milestones of shape [B, M, C, H, W], got {tuple(milestones.shape)}"
            )
        b, m = milestones.shape[:2]
        z = self(milestones.reshape(b * m, *milestones.shape[2:]))  # [B*M, d]
        return z.reshape(b, m, self.latent_dim)

    def encode_siamese(self, observation: Tensor, milestones: Tensor) -> Tuple[Tensor, Tensor]:
        """Siamese batching: encode observations and milestones in ONE forward pass of E_psi.

        Both streams are concatenated along the batch axis, passed through the shared weights,
        and split again. Because the network contains no cross-sample operation (LayerNorm only),
        the result is identical to encoding the two streams separately.

        Args:
            observation: ``[B, C, H_img, W_img]`` -- current frames I_t.
            milestones:  ``[B, M, C, H_img, W_img]`` -- milestone frames I_g^(1..M).

        Returns:
            z_t: ``[B, d]``;  z_g: ``[B, M, d]``.
        """
        if observation.dim() != 4 or milestones.dim() != 5:
            raise ValueError("expected observation [B, C, H, W] and milestones [B, M, C, H, W]")
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
        ``forward``: ``[B, C, H_img, W_img]`` -> ``[B, d]`` (detached)
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
        """``[B, C, H_img, W_img]`` -> ``[B, d]`` target latents sg(E_psi_bar(I))."""
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


def _self_test() -> None:
    torch.manual_seed(0)
    b, m, c, h, w, d = 2, 3, 3, 32, 32, 16
    enc = VisionEncoder(image_size=(h, w), patch_size=8, in_channels=c, width=32, depth=2,
                        num_heads=4, latent_dim=d).eval()
    obs = torch.randn(b, c, h, w)
    goals = torch.randn(b, m, c, h, w)

    z_t = enc(obs)
    assert z_t.shape == (b, d)
    z_g = enc.encode_milestones(goals)
    assert z_g.shape == (b, m, d)

    # Siamese batching must equal separate encoding (shared weights, per-sample independence).
    zs_t, zs_g = enc.encode_siamese(obs, goals)
    assert torch.allclose(zs_t, z_t, atol=1e-5) and torch.allclose(zs_g, z_g, atol=1e-5)

    # Momentum target: no grad, and EMA update follows psi_bar <- mu*psi_bar + (1-mu)*psi.
    target = MomentumEncoder(enc, momentum=0.9)
    assert not any(p.requires_grad for p in target.parameters())
    with torch.no_grad():
        for p in enc.parameters():
            p.add_(1.0)
    before = [p.clone() for p in target.encoder.parameters()]
    target.update(enc)
    for p_bar_new, p_bar_old, p in zip(target.encoder.parameters(), before, enc.parameters(), strict=True):
        assert torch.allclose(p_bar_new, 0.9 * p_bar_old + 0.1 * p, atol=1e-6)
    assert not target(obs).requires_grad

    # Frozen text encoder (offline stand-in with the CLIP interface).
    text = TextEncoderWrapper(text_model=_ToyTextModel(), tokenizer=_toy_tokenizer)
    text.train()
    assert not text.text_model.training
    c_text = text(["move the cup", "throw the cup away"])
    assert c_text.shape == (2, text.embed_dim) and not c_text.requires_grad
    assert not torch.allclose(c_text[0], c_text[1])
    print("perception.py self-test passed")


if __name__ == "__main__":
    _self_test()
