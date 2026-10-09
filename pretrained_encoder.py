# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
"""
pretrained_encoder.py -- initialise E_psi (``perception.VisionEncoder``) from pre-trained DINOv2 weights.

The paper specifies E_psi as a ViT trained with the JEPA objective (Sec. 4.1, 4.3, Eq. 11) and does not
fix its initialisation. This module only changes the *starting point* of E_psi: the architecture, the
forward pass and every equation stay as they are, and stage 1 of ``train.py`` then trains the encoder
with Eq. 11 exactly as before.

``VisionEncoder`` and a DINOv2 ViT share the same block structure (pre-norm, multi-head self-attention,
GELU MLP, [CLS] token, learned position embeddings), so the first ``vit_depth`` DINOv2 blocks are copied
into the encoder with three exact conversions:

    * Input normalisation. DINOv2 expects x_n = (x - mean) / std (ImageNet statistics); E_psi receives
      x in [0, 1]. The patch embedding is linear without padding, so
          conv(W, b)(x_n) = conv(W / std, b - sum W * mean / std)(x)
      and the normalisation is folded into the patch-embedding weights.
    * LayerScale. A DINOv2 block computes x + ls1 * Attn(LN(x)) and x + ls2 * MLP(LN(x)); the per-channel
      factors ls are folded into the output projections: W_o <- diag(ls1) W_o, b_o <- ls1 * b_o (same for fc2).
    * Position embeddings. DINOv2's own ``interpolate_pos_encoding`` resamples them to E_psi's patch grid.

E_psi's final LayerNorm takes DINOv2's final LayerNorm; the latent head (width -> d) keeps its fresh
initialisation. Two details differ from DINOv2 and are left as they are: E_psi's LayerNorms use the
PyTorch default eps (1e-5 instead of 1e-6), and only the first ``vit_depth`` of DINOv2's blocks are used.

Requirements: ``transformers`` (already a dependency for CLIP); the first use downloads the weights from
Hugging Face (``facebook/dinov2-small``, about 90 MB). ``python pretrained_encoder.py`` runs an offline
self-test against a randomly initialised DINOv2 built from a config (no download).
"""

from __future__ import annotations

import copy
from typing import Any, Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from torch import Tensor

try:  # package import
    from .perception import VisionEncoder
except ImportError:  # flat import (files side by side)
    from perception import VisionEncoder

__all__ = ["DINOV2_ALIASES", "IMAGENET_MEAN", "IMAGENET_STD", "load_dinov2", "copy_dinov2_into", "init_vision_encoder"]

DINOV2_ALIASES: Dict[str, str] = {
    "dinov2-small": "facebook/dinov2-small",   # ViT-S/14: width 384, 6 heads, 12 blocks -- matches VPAConfig defaults
}
IMAGENET_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: Tuple[float, float, float] = (0.229, 0.224, 0.225)

# Parameter names inside one DINOv2 layer; transformers renamed the attention projections over time.
_ATTENTION_KEYS = {
    "q": ("attention.q_proj", "attention.attention.query"),
    "k": ("attention.k_proj", "attention.attention.key"),
    "v": ("attention.v_proj", "attention.attention.value"),
    "o": ("attention.o_proj", "attention.output.dense"),
}


def load_dinov2(name_or_path: str) -> nn.Module:
    """``transformers.Dinov2Model`` from a Hugging Face id, a local directory, or an alias in ``DINOV2_ALIASES``."""
    from transformers import Dinov2Model  # lazy: only needed when the option is used

    model = Dinov2Model.from_pretrained(DINOV2_ALIASES.get(name_or_path, name_or_path))
    model.eval()
    return model


def _layer_tensor(state: Dict[str, Tensor], layer: int, part: str, suffix: str) -> Optional[Tensor]:
    for name in _ATTENTION_KEYS[part]:
        key = f"encoder.layer.{layer}.{name}.{suffix}"
        if key in state:
            return state[key]
    return None


@torch.no_grad()
def copy_dinov2_into(
    encoder: VisionEncoder,
    dinov2: nn.Module,
    mean: Sequence[float] = IMAGENET_MEAN,
    std: Sequence[float] = IMAGENET_STD,
) -> Dict[str, Any]:
    """Copy the first ``len(encoder.blocks)`` DINOv2 blocks into ``encoder`` (see the module docstring).

    Args:
        encoder: E_psi, modified in place. Its width, number of heads, MLP width and patch size must
            match the DINOv2 model, and it must have at most as many blocks.
        dinov2: a ``transformers.Dinov2Model`` (eval mode).
        mean, std: per-channel input normalisation DINOv2 was trained with.

    Returns:
        a JSON-serialisable record of what was copied.
    """
    cfg = dinov2.config
    width = encoder.cls_token.shape[-1]
    depth = len(encoder.blocks)
    heads = encoder.blocks[0].attn.num_heads
    mlp_hidden = encoder.blocks[0].mlp[0].out_features
    patch = encoder.patch_embed.patch_size
    channels = encoder.in_channels
    if getattr(cfg, "use_swiglu_ffn", False):
        raise ValueError("DINOv2 variants with a SwiGLU MLP (giant) do not match VisionEncoder's GELU MLP")
    hf_patch = cfg.patch_size if isinstance(cfg.patch_size, int) else int(cfg.patch_size[0])
    hf_mlp = int(cfg.hidden_size * cfg.mlp_ratio)
    checks = [
        ("width (vit_width)", width, cfg.hidden_size),
        ("attention heads (vit_heads)", heads, cfg.num_attention_heads),
        ("MLP width", mlp_hidden, hf_mlp),
        ("patch size (patch_size)", patch, hf_patch),
        ("input channels", channels, cfg.num_channels),
        ("normalisation channels", len(mean), channels),
    ]
    for name, ours, theirs in checks:
        if ours != theirs:
            raise ValueError(f"E_psi {name} = {ours}, but the DINOv2 model has {theirs}")
    if depth > cfg.num_hidden_layers:
        raise ValueError(f"E_psi has {depth} blocks, the DINOv2 model only {cfg.num_hidden_layers}")

    state = {k: v.detach() for k, v in dinov2.state_dict().items()}
    f64 = torch.float64

    def put(dst: Tensor, src: Tensor) -> None:
        if dst.shape != src.shape:
            raise ValueError(f"shape mismatch {tuple(src.shape)} -> {tuple(dst.shape)}")
        dst.copy_(src.to(device=dst.device, dtype=dst.dtype))

    # patch embedding with the input normalisation folded in (exact: stride = kernel, no padding)
    w = state["embeddings.patch_embeddings.projection.weight"].to(f64)          # [D, C, p, p]
    b = state["embeddings.patch_embeddings.projection.bias"].to(f64)            # [D]
    s = torch.tensor(std, dtype=f64).view(1, -1, 1, 1)
    m = torch.tensor(mean, dtype=f64).view(1, -1, 1, 1)
    w_folded = w / s
    put(encoder.patch_embed.proj.weight, w_folded)
    put(encoder.patch_embed.proj.bias, b - (w_folded * m).sum(dim=(1, 2, 3)))

    # [CLS] token and position embeddings resampled by DINOv2's own interpolation
    gh, gw = encoder.image_size[0] // patch, encoder.image_size[1] // patch
    put(encoder.cls_token, state["embeddings.cls_token"])
    probe = torch.zeros(1, 1 + gh * gw, width, dtype=dinov2.embeddings.position_embeddings.dtype,
                        device=dinov2.embeddings.position_embeddings.device)
    put(encoder.pos_embed, dinov2.embeddings.interpolate_pos_encoding(probe, gh * patch, gw * patch))

    for i, block in enumerate(encoder.blocks):
        pre = f"encoder.layer.{i}"
        put(block.norm1.weight, state[f"{pre}.norm1.weight"])
        put(block.norm1.bias, state[f"{pre}.norm1.bias"])
        put(block.norm2.weight, state[f"{pre}.norm2.weight"])
        put(block.norm2.bias, state[f"{pre}.norm2.bias"])
        qkv_w, qkv_b = [], []
        for part in ("q", "k", "v"):
            weight = _layer_tensor(state, i, part, "weight")
            if weight is None:
                raise KeyError(f"no '{part}' projection for DINOv2 layer {i}; unknown transformers naming")
            bias = _layer_tensor(state, i, part, "bias")
            qkv_w.append(weight)
            qkv_b.append(torch.zeros(weight.shape[0], dtype=weight.dtype) if bias is None else bias)
        put(block.attn.in_proj_weight, torch.cat(qkv_w, dim=0))                # rows: [q; k; v]
        put(block.attn.in_proj_bias, torch.cat(qkv_b, dim=0))
        ls1 = state[f"{pre}.layer_scale1.lambda1"].to(f64)
        ls2 = state[f"{pre}.layer_scale2.lambda1"].to(f64)
        o_w, o_b = _layer_tensor(state, i, "o", "weight"), _layer_tensor(state, i, "o", "bias")
        if o_w is None or o_b is None:
            raise KeyError(f"no attention output projection for DINOv2 layer {i}")
        put(block.attn.out_proj.weight, ls1[:, None] * o_w.to(f64))           # LayerScale folded
        put(block.attn.out_proj.bias, ls1 * o_b.to(f64))
        put(block.mlp[0].weight, state[f"{pre}.mlp.fc1.weight"])
        put(block.mlp[0].bias, state[f"{pre}.mlp.fc1.bias"])
        put(block.mlp[2].weight, ls2[:, None] * state[f"{pre}.mlp.fc2.weight"].to(f64))
        put(block.mlp[2].bias, ls2 * state[f"{pre}.mlp.fc2.bias"].to(f64))

    put(encoder.norm.weight, state["layernorm.weight"])
    put(encoder.norm.bias, state["layernorm.bias"])
    return {
        "type": "dinov2",
        "blocks_used": depth,
        "blocks_available": int(cfg.num_hidden_layers),
        "patch_grid": [gh, gw],
        "pretrained_patch_grid": int(round((state["embeddings.position_embeddings"].shape[1] - 1) ** 0.5)),
        "mean": list(mean),
        "std": list(std),
    }


def init_vision_encoder(encoder: VisionEncoder, name_or_path: str) -> Dict[str, Any]:
    """Load DINOv2 (``load_dinov2``) and copy it into ``encoder``; returns the record of ``copy_dinov2_into``."""
    info = copy_dinov2_into(encoder, load_dinov2(name_or_path))
    info["source"] = DINOV2_ALIASES.get(name_or_path, name_or_path)
    return info


# ---------------------------------------------------------------------------------------------
# Self-test (run: python pretrained_encoder.py) -- offline, random DINOv2 built from a config
# ---------------------------------------------------------------------------------------------
def _layer_out(output: Any) -> Tensor:
    return output[0] if isinstance(output, tuple) else output


@torch.no_grad()
def _dinov2_features(model: nn.Module, x01: Tensor, depth: int, mean: Sequence[float], std: Sequence[float]) -> Tensor:
    """Reference: DINOv2's own modules on normalised input, first ``depth`` layers, final LayerNorm, [CLS]."""
    m = torch.tensor(mean, dtype=x01.dtype).view(1, -1, 1, 1)
    s = torch.tensor(std, dtype=x01.dtype).view(1, -1, 1, 1)
    h = model.embeddings((x01 - m) / s)
    for layer in model.encoder.layer[:depth]:
        h = _layer_out(layer(h))
    return model.layernorm(h)[:, 0]


@torch.no_grad()
def _encoder_features(encoder: VisionEncoder, x01: Tensor) -> Tensor:
    """E_psi's forward pass up to (and including) the final LayerNorm, at the [CLS] token."""
    x = encoder.patch_embed(x01)
    x = torch.cat([encoder.cls_token.expand(x.shape[0], -1, -1), x], dim=1) + encoder.pos_embed
    for block in encoder.blocks:
        x = block(x)
    return encoder.norm(x)[:, 0]


def _self_test() -> None:
    from transformers import Dinov2Config, Dinov2Model

    torch.manual_seed(0)
    cfg = Dinov2Config(hidden_size=32, num_hidden_layers=4, num_attention_heads=4, mlp_ratio=4, patch_size=8,
                       image_size=40, num_channels=3, layerscale_value=0.5, qkv_bias=True)
    model = Dinov2Model(cfg).eval()
    with torch.no_grad():  # random values everywhere, LayerScale and biases included
        for p in model.parameters():
            p.normal_(0.0, 0.3)
    model = model.double()
    eps = cfg.layer_norm_eps

    for image_size in (24, 40):  # 3x3 grid (position embeddings resampled) and 5x5 (the pre-trained grid)
        enc = VisionEncoder(image_size=image_size, patch_size=8, in_channels=3, width=32, depth=2, num_heads=4,
                            latent_dim=8).double().eval()
        head_before = enc.head.weight.clone()
        info = copy_dinov2_into(enc, model)
        assert info["blocks_used"] == 2 and info["patch_grid"] == [image_size // 8] * 2
        assert info["pretrained_patch_grid"] == 5
        assert torch.equal(enc.head.weight, head_before)                       # the latent head is not touched
        x = torch.rand(3, 3, image_size, image_size, dtype=torch.float64)
        ref = _dinov2_features(model, x, 2, IMAGENET_MEAN, IMAGENET_STD)
        # exact conversion: with DINOv2's LayerNorm eps the features agree to float64 round-off
        exact = copy.deepcopy(enc)
        for module in exact.modules():
            if isinstance(module, nn.LayerNorm):
                module.eps = eps
        got = _encoder_features(exact, x)
        assert torch.allclose(got, ref, rtol=1e-9, atol=1e-10), (got - ref).abs().max()
        # E_psi as used (LayerNorm eps 1e-5): close, not identical
        assert torch.allclose(_encoder_features(enc, x), ref, rtol=1e-3, atol=1e-3)
        assert torch.allclose(enc(x), enc.head(_encoder_features(enc, x)), rtol=1e-12, atol=1e-12)

    # mismatched architectures are rejected
    for kwargs in ({"width": 48, "num_heads": 4}, {"width": 32, "num_heads": 8}, {"width": 32, "num_heads": 4,
                                                                                     "depth": 5}):
        bad = VisionEncoder(image_size=24, patch_size=8, in_channels=3, latent_dim=8,
                            **{"depth": 2, **kwargs}).double()
        try:
            copy_dinov2_into(bad, model)
            raise AssertionError(f"expected ValueError for {kwargs}")
        except ValueError:
            pass
    bad_patch = VisionEncoder(image_size=24, patch_size=4, in_channels=3, width=32, depth=2, num_heads=4,
                              latent_dim=8).double()
    try:
        copy_dinov2_into(bad_patch, model)
        raise AssertionError("expected ValueError for the patch size")
    except ValueError:
        pass
    print("pretrained_encoder.py self-test passed")


if __name__ == "__main__":
    _self_test()
