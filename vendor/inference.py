from __future__ import annotations

import torch
from .feature_utils import feature_process
from .model_utils import denoise_process


def infer(video_path, prompt, model_dict, cfg, guidance_scale=4.5, num_inference_steps=50):
    visual_feats, text_feats, audio_len_in_s = feature_process(
        video_path,
        prompt,
        model_dict,
        cfg
    )
    audio, sample_rate = denoise_process(
        visual_feats,
        text_feats,
        audio_len_in_s,
        model_dict,
        cfg,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps
    )
    return audio[0], sample_rate

