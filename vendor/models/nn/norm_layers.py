# Copyright (c) Tencent.
# Licensed under the Apache License, Version 2.0.
from __future__ import annotations
import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, dim: int, elementwise_affine=True, eps: float = 1e-6, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.eps = eps
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim, **factory_kwargs))
    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        if hasattr(self, "weight"):
            output = output * self.weight
        return output

def get_norm_layer(norm_layer):
    if norm_layer == "layer": return nn.LayerNorm
    elif norm_layer == "rms": return RMSNorm
    else: raise NotImplementedError(f"Norm layer {norm_layer} is not implemented")

