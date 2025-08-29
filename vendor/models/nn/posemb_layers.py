# Copyright (c) Tencent.
# Licensed under the Apache License, Version 2.0.
from __future__ import annotations
import torch
from typing import Union, Tuple

def _to_tuple(x, dim=2):
    if isinstance(x, int): return (x,) * dim
    elif len(x) == dim: return x
    else: raise ValueError(f"Expected length {dim} or int, but got {x}")

def get_meshgrid_nd(start, *args, dim=2):
    if len(args) == 0:
        num = _to_tuple(start, dim=dim); start = (0,) * dim; stop = num
    elif len(args) == 1:
        start = _to_tuple(start, dim=dim); stop = _to_tuple(args[0], dim=dim); num = [stop[i] - start[i] for i in range(dim)]
    elif len(args) == 2:
        start = _to_tuple(start, dim=dim); stop = _to_tuple(args[0], dim=dim); num = _to_tuple(args[1], dim=dim)
    else:
        raise ValueError(f"len(args) should be 0, 1 or 2, but got {len(args)}")
    axis_grid = []
    for i in range(dim):
        a, b, n = start[i], stop[i], num[i]
        g = torch.linspace(a, b, n + 1, dtype=torch.float32)[:n]
        axis_grid.append(g)
    grid = torch.meshgrid(*axis_grid, indexing="ij"); grid = torch.stack(grid, dim=0)
    return grid

# Rotary Positional Embedding

def get_nd_rotary_pos_embed(rope_dim_list, start, *args, theta=10000.0, use_real=False, theta_rescale_factor=1.0, freq_scaling=1.0):
    grid = get_meshgrid_nd(start, *args, dim=len(rope_dim_list))
    embs = []
    for i in range(len(rope_dim_list)):
        emb = get_1d_rotary_pos_embed(rope_dim_list[i], grid[i].reshape(-1), theta, use_real=use_real, theta_rescale_factor=theta_rescale_factor, freq_scaling=freq_scaling)
        embs.append(emb)
    if use_real:
        cos = torch.cat([emb[0] for emb in embs], dim=1)
        sin = torch.cat([emb[1] for emb in embs], dim=1)
        return cos, sin
    else:
        emb = torch.cat(embs, dim=1)
        return emb

def get_1d_rotary_pos_embed(dim: int, pos: Union[torch.FloatTensor, int], theta: float = 10000.0, use_real: bool = False, theta_rescale_factor: float = 1.0, freq_scaling: float = 1.0) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    if isinstance(pos, int): pos = torch.arange(pos).float()
    if theta_rescale_factor != 1.0:
        theta *= theta_rescale_factor ** (dim / (dim - 1))
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    freqs *= freq_scaling
    freqs = torch.outer(pos, freqs)
    if use_real:
        freqs_cos = freqs.cos().repeat_interleave(2, dim=1)
        freqs_sin = freqs.sin().repeat_interleave(2, dim=1)
        return freqs_cos, freqs_sin
    else:
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
        return freqs_cis

