#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Copyright 2020 Ross Wightman
# Modified Model definition

from collections import OrderedDict
from functools import partial

import torch
import torch.nn as nn

# timm fallback for trunc_normal_
try:
    from timm.layers import trunc_normal_  # type: ignore
    _HAS_TIMM = True
except Exception:
    _HAS_TIMM = False
    def trunc_normal_(tensor, std=0.02):
        with torch.no_grad():
            return tensor.normal_(mean=0.0, std=std)

from .vit_helper import PatchEmbed, PatchEmbed3D, DividedSpaceTimeBlock


class VisionTransformer(nn.Module):
    """Vision Transformer with support for patch or hybrid CNN input stage.
    Vendorized: keep interfaces; use safe fallbacks for init.
    """

    def __init__(self, cfg):
        super().__init__()
        self.img_size = cfg.DATA.TRAIN_CROP_SIZE
        self.patch_size = cfg.VIT.PATCH_SIZE
        self.in_chans = cfg.VIT.CHANNELS
        if getattr(cfg.TRAIN, "DATASET", "") == "Epickitchens":
            self.num_classes = [97, 300]
        else:
            self.num_classes = cfg.MODEL.NUM_CLASSES
        self.embed_dim = cfg.VIT.EMBED_DIM
        self.depth = cfg.VIT.DEPTH
        self.num_heads = cfg.VIT.NUM_HEADS
        self.mlp_ratio = cfg.VIT.MLP_RATIO
        self.qkv_bias = cfg.VIT.QKV_BIAS
        self.drop_rate = cfg.VIT.DROP
        self.drop_path_rate = cfg.VIT.DROP_PATH
        self.head_dropout = cfg.VIT.HEAD_DROPOUT
        self.video_input = cfg.VIT.VIDEO_INPUT
        self.temporal_resolution = cfg.VIT.TEMPORAL_RESOLUTION
        self.use_mlp = cfg.VIT.USE_MLP
        self.num_features = self.embed_dim
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.attn_drop_rate = cfg.VIT.ATTN_DROPOUT
        self.head_act = cfg.VIT.HEAD_ACT
        self.cfg = cfg

        # Patch Embedding
        self.patch_embed = PatchEmbed(
            img_size=224, patch_size=self.patch_size, in_chans=self.in_chans, embed_dim=self.embed_dim
        )

        # 3D Patch Embedding
        self.patch_embed_3d = PatchEmbed3D(
            img_size=self.img_size,
            temporal_resolution=self.temporal_resolution,
            patch_size=self.patch_size,
            in_chans=self.in_chans,
            embed_dim=self.embed_dim,
            z_block_size=self.cfg.VIT.PATCH_SIZE_TEMP,
        )
        # zero init proj like upstream
        self.patch_embed_3d.proj.weight.data = torch.zeros_like(self.patch_embed_3d.proj.weight.data)

        # Number of patches
        if self.video_input:
            num_patches = self.patch_embed.num_patches * self.temporal_resolution
        else:
            num_patches = self.patch_embed.num_patches
        self.num_patches = num_patches

        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        trunc_normal_(self.cls_token, std=0.02)

        # Positional embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches + 1, self.embed_dim))
        self.pos_drop = nn.Dropout(p=cfg.VIT.POS_DROPOUT)
        trunc_normal_(self.pos_embed, std=0.02)

        if getattr(self.cfg.VIT, "POS_EMBED", "separate") == "joint":
            self.st_embed = nn.Parameter(torch.zeros(1, num_patches + 1, self.embed_dim))
            trunc_normal_(self.st_embed, std=0.02)
        else:
            self.temp_embed = nn.Parameter(torch.zeros(1, self.temporal_resolution, self.embed_dim))

        # Layer Blocks
        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]
        if getattr(self.cfg.VIT, "ATTN_LAYER", "divided") == "divided":
            self.blocks = nn.ModuleList(
                [
                    DividedSpaceTimeBlock(
                        attn_type=self.cfg.VIT.ATTN_LAYER,
                        dim=self.embed_dim,
                        num_heads=self.num_heads,
                        mlp_ratio=self.mlp_ratio,
                        qkv_bias=self.qkv_bias,
                        drop=self.drop_rate,
                        attn_drop=self.attn_drop_rate,
                        drop_path=dpr[i],
                        norm_layer=norm_layer,
                    )
                    for i in range(self.depth)
                ]
            )

        self.norm = norm_layer(self.embed_dim)

        # MLP head
        if self.use_mlp:
            hidden_dim = self.embed_dim
            if self.head_act == "tanh":
                act = nn.Tanh()
            elif self.head_act == "gelu":
                act = nn.GELU()
            else:
                act = nn.ReLU()
            self.pre_logits = nn.Sequential(
                OrderedDict(
                    [
                        ("fc", nn.Linear(self.embed_dim, hidden_dim)),
                        ("act", act),
                    ]
                )
            )
        else:
            self.pre_logits = nn.Identity()

        # Classifier Head
        self.head_drop = nn.Dropout(p=self.head_dropout)
        if isinstance(self.num_classes, (list,)) and len(self.num_classes) > 1:
            for a, i in enumerate(range(len(self.num_classes))):
                setattr(self, "head%d" % a, nn.Linear(self.embed_dim, self.num_classes[i]))
        else:
            self.head = nn.Linear(self.embed_dim, self.num_classes) if self.num_classes > 0 else nn.Identity()

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        if getattr(self.cfg.VIT, "POS_EMBED", "separate") == "joint":
            return {"pos_embed", "cls_token", "st_embed"}
        else:
            return {"pos_embed", "cls_token", "temp_embed"}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=""):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        B = x.shape[0]
        # apply 3D patching
        x = self.patch_embed_3d(x)
        tok_mask = None

        # Append CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        new_pos_embed = self.pos_embed
        npatch = self.patch_embed.num_patches

        # Add positional embeddings to input
        if self.video_input:
            if getattr(self.cfg.VIT, "POS_EMBED", "separate") == "separate":
                cls_embed = self.pos_embed[:, 0, :].unsqueeze(1)
                # number of token frames after 3D patching
                seq_tokens = x.shape[1] - 1  # exclude CLS
                frames = max(1, seq_tokens // npatch)
                # tile spatial pos embedding across actual frames count
                tile_pos_embed = new_pos_embed[:, 1:, :].repeat(1, frames, 1)
                # use the first `frames` temporal embeddings and tile each across spatial patches
                temp = self.temp_embed[:, :frames, :]
                tile_temporal_embed = temp.repeat_interleave(npatch, 1)
                total_pos_embed = tile_pos_embed + tile_temporal_embed
                total_pos_embed = torch.cat([cls_embed, total_pos_embed], dim=1)
                x = x + total_pos_embed
            else:
                x = x + self.st_embed
        else:
            x = x + new_pos_embed

        # Apply positional dropout
        x = self.pos_drop(x)

        # Encoding using transformer layers
        for i, blk in enumerate(self.blocks):
            x = blk(
                x,
                seq_len=npatch,
                num_frames=self.temporal_resolution,
                approx=getattr(self.cfg.VIT, "APPROX_ATTN_TYPE", "none"),
                num_landmarks=getattr(self.cfg.VIT, "APPROX_ATTN_DIM", 64),
                tok_mask=tok_mask,
            )

        return x, tok_mask

