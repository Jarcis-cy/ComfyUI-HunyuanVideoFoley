from __future__ import annotations
import os
import torch
from torch import nn

# Resolve and import the real implementation from local HunyuanVideo-Foley repo
import sys
from pathlib import Path
RealDAC = None
try:
    _here = Path(__file__).resolve()
    # Try to locate repo root and HunyuanVideo-Foley folder
    for p in _here.parents:
        cand = p / 'HunyuanVideo-Foley'
        if cand.exists() and cand.is_dir():
            if str(cand) not in sys.path:
                sys.path.insert(0, str(cand))
            break
    from hunyuanvideo_foley.models.dac_vae.model.dac import DAC as RealDAC  # type: ignore
except Exception as _e:
    RealDAC = None
    _REAL_DAC_IMPORT_ERROR = _e


class DAC(nn.Module):
    sample_rate: int = 48000

    def __init__(self, inner: nn.Module | None = None):
        super().__init__()
        if inner is None:
            if RealDAC is None:
                raise ImportError("Real DAC implementation not found. Ensure HunyuanVideo-Foley submodule is present or 'descript-audiotools' installed.")
            inner = RealDAC()
        self.inner = inner
        # propagate sample_rate if available
        if hasattr(self.inner, 'sample_rate'):
            self.sample_rate = int(getattr(self.inner, 'sample_rate'))

    @staticmethod
    def load(path: str):
        # The upstream RealDAC().load(path) typically loads internal weights; if absent, just instantiate
        if RealDAC is not None:
            try:
                # Some versions expose a classmethod load; otherwise we can manually load state_dict
                if hasattr(RealDAC, 'load') and callable(getattr(RealDAC, 'load')):
                    inner = RealDAC.load(path)  # type: ignore
                else:
                    inner = RealDAC()
                    if os.path.isfile(path):
                        sd = torch.load(path, map_location='cpu', weights_only=False)
                        inner.load_state_dict(sd, strict=False)
                return DAC(inner)
            except Exception as e:
                # fallback: best-effort init
                inner = RealDAC()
                return DAC(inner)
        raise ImportError("Real DAC implementation not available.")

    def forward(self, *args, **kwargs):
        return self.inner(*args, **kwargs)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        # Expect shape (B, C, T) -> returns (B, 1, samples) or (B, samples)
        audio = self.inner.decode(latents)
        # Normalize output to (B, samples)
        if isinstance(audio, dict) and 'audio' in audio:
            audio = audio['audio']
        if audio.ndim == 3 and audio.shape[1] == 1:
            audio = audio[:, 0, :]
        return audio

