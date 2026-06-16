"""Self-contained vision tower for our DeCLIP-trained CLIP-ViT-B/32, bundled
inside the brain-score submission plugin so the CI sandbox doesn't need to clone
our research repo. Trimmed from src/model.py to JUST the VisionTransformer +
direct dependencies (Attention, TransformerBlock, LayerNorm, QuickGELU). The
text encoder and full CLIP wrapper are dropped — brain-score only needs visual
feature extraction.

State dict loading: our Lightning checkpoint has keys like
`model.visual.conv1.weight`, `model.text.*`, `model.logit_scale`. The plugin's
model.py strips the `model.visual.` prefix and filters to visual-only keys
before calling `VisionTransformer.load_state_dict`.
"""
import math
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class QuickGELU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(1.702 * x)


class LayerNorm(nn.LayerNorm):
    """Standard nn.LayerNorm — kept under its original name so state_dict keys match."""
    def forward(self, x):
        orig_type = x.dtype
        out = super().forward(x)
        return out.to(orig_type)


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        if hasattr(F, 'scaled_dot_product_attention'):
            x = F.scaled_dot_product_attention(q, k, v, attn_mask=None,
                                                dropout_p=self.attn_drop.p if self.training else 0.0,
                                                is_causal=False)
            x = x.transpose(1, 2).reshape(B, N, C)
        else:
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=False, drop=0.,
                 attn_drop=0., ls_init_value: Optional[float] = None,
                 act_layer: Callable = nn.GELU, norm_layer: Callable = LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                              attn_drop=attn_drop, proj_drop=drop)
        self.ls_init_value = ls_init_value
        if ls_init_value is not None:
            self.gamma_1 = nn.Parameter(ls_init_value * torch.ones(dim))
            self.gamma_2 = nn.Parameter(ls_init_value * torch.ones(dim))
        else:
            self.gamma_1 = None
            self.gamma_2 = None
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            act_layer(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop),
        )

    def forward(self, x):
        if self.gamma_1 is None:
            x = x + self.attn(self.norm1(x))
            x = x + self.mlp(self.norm2(x))
        else:
            x = x + self.gamma_1 * self.attn(self.norm1(x))
            x = x + self.gamma_2 * self.mlp(self.norm2(x))
        return x


class VisionTransformer(nn.Module):
    """ViT-B/32 visual encoder for CLIP-style models. Architectural defaults are
    hardcoded to match VIT_B_32_CONFIG from our codebase (image_size=224,
    patch_size=32, width=768, layers=12, heads=12, mlp_ratio=4.0,
    output_dim=512, no layer scale)."""
    def __init__(self, image_size=224, patch_size=32, width=768, layers=12,
                 heads=12, mlp_ratio=4.0, output_dim=512,
                 ls_init_value: Optional[float] = None,
                 act_layer: Callable = nn.GELU,
                 norm_layer: Callable = LayerNorm):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.width = width
        self.output_dim = output_dim
        self.grid_size = (image_size // patch_size, image_size // patch_size)
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.conv1 = nn.Conv2d(3, width, kernel_size=patch_size, stride=patch_size, bias=False)
        self.class_embedding = nn.Parameter(torch.randn(width))
        self.positional_embedding = nn.Parameter(torch.randn(self.num_patches + 1, width) * 0.01)
        self.ln_pre = norm_layer(width)
        self.transformer = nn.Sequential(*[
            TransformerBlock(dim=width, num_heads=heads, mlp_ratio=mlp_ratio,
                             qkv_bias=True, ls_init_value=ls_init_value,
                             act_layer=act_layer, norm_layer=norm_layer)
            for _ in range(layers)
        ])
        self.ln_post = norm_layer(width)
        self.proj = nn.Parameter(torch.randn(width, output_dim) * (1 / width ** 0.5))

    def forward(self, x):
        x = self.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
        x = torch.cat([
            self.class_embedding.to(x.dtype) + torch.zeros(
                x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device,
            ),
            x,
        ], dim=1)
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)
        for block in self.transformer:
            x = block(x)
        x = x[:, 0]
        x = self.ln_post(x)
        x = x @ self.proj
        return x
