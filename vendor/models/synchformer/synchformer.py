# Copyright (c) Tencent.
# Licensed under the Apache License, Version 2.0.
from __future__ import annotations
import logging
import math
from typing import Any, Mapping

# Optional deps with graceful fallbacks
try:
    import einops  # type: ignore
    _HAS_EINOPS = True
except Exception as _e:
    einops = None
    _HAS_EINOPS = False
    _EINOPS_IMPORT_ERROR = _e

try:
    import torchaudio  # type: ignore
    _HAS_TORCHAUDIO = True
except Exception as _e:
    torchaudio = None
    _HAS_TORCHAUDIO = False
    _TORCHAUDIO_IMPORT_ERROR = _e

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .motionformer import MotionFormer
from .ast_model import AST
from .utils import Config

class Synchformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.vfeat_extractor = MotionFormer(extract_features=True, factorize_space_time=True, agg_space_module="TransformerEncoderLayer", agg_time_module="torch.nn.Identity", add_global_repr=False,)
        self.afeat_extractor = AST(extract_features=True, max_spec_t=66, factorize_freq_time=True, agg_freq_module="TransformerEncoderLayer", agg_time_module="torch.nn.Identity", add_global_repr=False,)
        self.vproj = nn.Linear(in_features=768, out_features=768)
        self.aproj = nn.Linear(in_features=768, out_features=768)
        self.transformer = GlobalTransformer(tok_pdrop=0.0, embd_pdrop=0.1, resid_pdrop=0.1, attn_pdrop=0.1, n_layer=3, n_head=8, n_embd=768)
    def forward(self, vis):
        B, S, Tv, C, H, W = vis.shape
        vis = vis.permute(0, 1, 3, 2, 4, 5)
        vis = self.vfeat_extractor(vis)
        return vis
    def compare_v_a(self, vis: torch.Tensor, aud: torch.Tensor):
        vis = self.vproj(vis); aud = self.aproj(aud)
        B, S, tv, D = vis.shape; B, S, ta, D = aud.shape
        vis = vis.view(B, S * tv, D); aud = aud.view(B, S * ta, D)
        logits = self.transformer(vis, aud)
        return logits
    def extract_vfeats(self, vis):
        B, S, Tv, C, H, W = vis.shape; vis = vis.permute(0, 1, 3, 2, 4, 5)
        vis = self.vfeat_extractor(vis); return vis
    def extract_afeats(self, aud):
        B, S, _, Fa, Ta = aud.shape; aud = aud.view(B, S, Fa, Ta).permute(0, 1, 3, 2)
        aud, _ = self.afeat_extractor(aud); return aud
    def compute_loss(self, logits, targets, loss_fn: str = None):
        loss = None
        if targets is not None:
            if loss_fn is None or loss_fn == "cross_entropy":
                loss = F.cross_entropy(logits, targets)
            else:
                raise NotImplementedError(f"Loss {loss_fn} not implemented")
        return loss
    def load_state_dict(self, sd: Mapping[str, Any], strict: bool = True):
        return super().load_state_dict(sd, strict)

class RandInitPositionalEncoding(nn.Module):
    def __init__(self, block_shape: list, n_embd: int):
        super().__init__()
        self.block_shape = block_shape; self.n_embd = n_embd
        self.pos_emb = nn.Parameter(torch.randn(1, *block_shape, n_embd))
    def forward(self, token_embeddings):
        return token_embeddings + self.pos_emb

class GlobalTransformer(torch.nn.Module):
    def __init__(self, tok_pdrop=0.0, embd_pdrop=0.1, resid_pdrop=0.1, attn_pdrop=0.1, n_layer=3, n_head=8, n_embd=768, pos_emb_block_shape=[198,], n_off_head_out=21,) -> None:
        super().__init__()
        self.config = Config(embd_pdrop=embd_pdrop, resid_pdrop=resid_pdrop, attn_pdrop=attn_pdrop, n_layer=n_layer, n_head=n_head, n_embd=n_embd)
        self.vis_in_lnorm = torch.nn.LayerNorm(n_embd)
        self.aud_in_lnorm = torch.nn.LayerNorm(n_embd)
        self.OFF_tok = torch.nn.Parameter(torch.randn(1, 1, n_embd))
        self.MOD_tok = torch.nn.Parameter(torch.randn(1, 1, n_embd))
        self.tok_pdrop = tok_pdrop
        self.tok_drop_vis = torch.nn.Dropout1d(tok_pdrop)
        self.tok_drop_aud = torch.nn.Dropout1d(tok_pdrop)
        self.pos_emb_cfg = RandInitPositionalEncoding(block_shape=pos_emb_block_shape, n_embd=n_embd)
        self.drop = torch.nn.Dropout(embd_pdrop)
        self.blocks = torch.nn.Sequential(*[Block(self.config) for _ in range(n_layer)])
        self.ln_f = torch.nn.LayerNorm(n_embd)
        self.off_head = torch.nn.Linear(in_features=n_embd, out_features=n_off_head_out)
    def forward(self, v: torch.Tensor, a: torch.Tensor, targets=None, attempt_to_apply_heads=True):
        B, Sv, D = v.shape; B, Sa, D = a.shape
        if not _HAS_EINOPS:
            raise ImportError(f"einops is required for Synchformer.GlobalTransformer but not available: {_EINOPS_IMPORT_ERROR}")
        off_tok = einops.repeat(self.OFF_tok, "1 1 d -> b 1 d", b=B)
        mod_tok = einops.repeat(self.MOD_tok, "1 1 d -> b 1 d", b=B)
        v, a = self.vis_in_lnorm(v), self.aud_in_lnorm(a)
        if self.tok_pdrop > 0: v, a = self.tok_drop_vis(v), self.tok_drop_aud(a)
        x = torch.cat((off_tok, v, mod_tok, a), dim=1)
        if hasattr(self, "pos_emb_cfg"): x = self.pos_emb_cfg(x)
        x = self.drop(x); x = self.blocks(x); x = self.ln_f(x)
        if attempt_to_apply_heads and hasattr(self, "off_head"): x = self.off_head(x[:, 0, :])
        return x

class SelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__(); assert config.n_embd % config.n_head == 0
        self.key = nn.Linear(config.n_embd, config.n_embd); self.query = nn.Linear(config.n_embd, config.n_embd); self.value = nn.Linear(config.n_embd, config.n_embd)
        self.attn_drop = nn.Dropout(config.attn_pdrop); self.resid_drop = nn.Dropout(config.resid_pdrop); self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.n_head = config.n_head
    def forward(self, x):
        B, T, C = x.size(); k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = F.softmax(att, dim=-1); y = self.attn_drop(att) @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C); y = self.resid_drop(self.proj(y)); return y

class Block(nn.Module):
    def __init__(self, config):
        super().__init__(); self.ln1 = nn.LayerNorm(config.n_embd); self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = SelfAttention(config)
        self.mlp = nn.Sequential(nn.Linear(config.n_embd, 4 * config.n_embd), nn.GELU(), nn.Linear(4 * config.n_embd, config.n_embd), nn.Dropout(config.resid_pdrop),)
    def forward(self, x):
        x = x + self.attn(self.ln1(x)); x = x + self.mlp(self.ln2(x)); return x

# helpers for audio encoding (if needed outside)

def pad_or_truncate(audio: torch.Tensor, max_spec_t: int, pad_mode: str = "constant", pad_value: float = 0.0):
    difference = max_spec_t - audio.shape[-1]
    if difference > 0:
        pad_dims = (0, difference); audio = torch.nn.functional.pad(audio, pad_dims, pad_mode, pad_value)
    elif difference < 0:
        audio = audio[..., :max_spec_t]
    return audio

def encode_audio_with_sync(synchformer: Synchformer, x: torch.Tensor, mel: torchaudio.transforms.MelSpectrogram) -> torch.Tensor:
    b, t = x.shape; segment_size = 10240; step_size = 10240 // 2
    num_segments = (t - segment_size) // step_size + 1; segments = []
    for i in range(num_segments): segments.append(x[:, i * step_size : i * step_size + segment_size])
    x = torch.stack(segments, dim=1)
    x = mel(x); x = torch.log(x + 1e-6); x = pad_or_truncate(x, 66)
    mean = -4.2677393; std = 4.5689974; x = (x - mean) / (2 * std)
    x = synchformer.extract_afeats(x.unsqueeze(2)); return x

