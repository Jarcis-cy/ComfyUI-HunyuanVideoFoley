#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Copyright 2020 Ross Wightman
# Modified Model definition
"""Video models (vendorized minimal subset)."""

import math
import torch
import torch.nn as nn

# optional deps
try:
    from timm.layers import to_2tuple  # type: ignore
except Exception:
    def to_2tuple(x):
        return (x, x) if not isinstance(x, tuple) else x


class DividedSpaceTimeBlock(nn.Module):
    def __init__(
        self,
        dim=768,
        num_heads=12,
        attn_type="divided",
        mlp_ratio=4.0,
        qkv_bias=False,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        # Minimal transformer block: LN -> MHSA -> add -> LN -> MLP -> add.
        self.norm1 = norm_layer(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.drop_path = nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            act_layer(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop),
        )

    def forward(self, x, seq_len=196, num_frames=8, approx="none", num_landmarks=128, tok_mask: torch.Tensor = None):
        # x: (B, N, D)
        resid = x
        x = self.norm1(x)
        attn_output, _ = self.attn(x, x, x, need_weights=False)
        x = resid + attn_output
        x = x + self.mlp(self.norm2(x))
        return x


class PatchEmbed(nn.Module):
    """Image to Patch Embedding"""

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = img_size if type(img_size) is tuple else to_2tuple(img_size)
        patch_size = img_size if type(patch_size) is tuple else to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class PatchEmbed3D(nn.Module):
    """Video to Patch Embedding"""

    def __init__(
        self,
        img_size=224,
        temporal_resolution=4,
        in_chans=3,
        patch_size=16,
        z_block_size=2,
        embed_dim=768,
        flatten=True,
    ):
        super().__init__()
        self.height = img_size // patch_size
        self.width = img_size // patch_size
        self.z_block_size = z_block_size
        self.proj = nn.Conv3d(
            in_chans,
            embed_dim,
            kernel_size=(z_block_size, patch_size, patch_size),
            stride=(z_block_size, patch_size, patch_size),
        )
        self.flatten = flatten

    def forward(self, x):
        # x: (B, C, T, H, W)
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)
        return x

