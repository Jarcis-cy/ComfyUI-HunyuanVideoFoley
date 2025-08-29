# Vendored minimal base classes from DAC VAE needed at inference time
from __future__ import annotations
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Union
import numpy as np
import torch
from torch import nn

SUPPORTED_VERSIONS = ["1.0.0"]

@dataclass
class DACFile:
    codes: torch.Tensor
    chunk_length: int
    original_length: int
    input_db: float
    channels: int
    sample_rate: int
    padding: bool
    dac_version: str

class CodecMixin:
    def get_output_length(self, input_length):
        L = input_length
        for layer in self.modules():
            if isinstance(layer, (nn.Conv1d, nn.ConvTranspose1d)):
                d = layer.dilation[0]; k = layer.kernel_size[0]; s = layer.stride[0]
                if isinstance(layer, nn.Conv1d): L = ((L - d * (k - 1) - 1) / s) + 1
                elif isinstance(layer, nn.ConvTranspose1d): L = (L - 1) * s + d * (k - 1) + 1
                L = math.floor(L)
        return L
    def get_delay(self):
        l_out = self.get_output_length(0); L = l_out
        layers = [l for l in self.modules() if isinstance(l, (nn.Conv1d, nn.ConvTranspose1d))]
        for layer in reversed(layers):
            d = layer.dilation[0]; k = layer.kernel_size[0]; s = layer.stride[0]
            if isinstance(layer, nn.ConvTranspose1d): L = ((L - d * (k - 1) - 1) / s) + 1
            elif isinstance(layer, nn.Conv1d): L = (L - 1) * s + d * (k - 1) + 1
            L = math.ceil(L)
        l_in = L; return (l_in - l_out) // 2

