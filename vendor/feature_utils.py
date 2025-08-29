from __future__ import annotations

import os
import numpy as np
import torch
import av
from PIL import Image
from einops import rearrange
from typing import List, Tuple

from .config_utils import AttributeDict
from .constants import FPS_VISUAL, DEFAULT_NEGATIVE_PROMPT

class FeatureExtractionError(Exception):
    pass

def get_frames_av(video_path: str, fps: float, max_length: float | None = None) -> Tuple[np.ndarray, float]:
    end_sec = max_length if max_length is not None else 15
    next_frame_time_for_each_fps = 0.0
    time_delta_for_each_fps = 1 / fps
    output_frames = []
    with av.open(video_path) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        for packet in container.demux(stream):
            for frame in packet.decode():
                frame_time = frame.time
                if frame_time is None or frame_time < 0:
                    continue
                if frame_time > end_sec:
                    break
                frame_np = None
                this_time = frame_time
                while this_time >= next_frame_time_for_each_fps:
                    if frame_np is None:
                        frame_np = frame.to_ndarray(format="rgb24")
                    output_frames.append(frame_np)
                    next_frame_time_for_each_fps += time_delta_for_each_fps
    output_frames = np.stack(output_frames) if len(output_frames) else np.zeros((0, 224, 224, 3), dtype=np.uint8)
    vid_len_in_s = len(output_frames) / fps if fps > 0 else 0
    if max_length is not None and len(output_frames) > int(max_length * fps):
        output_frames = output_frames[: int(max_length * fps)]
        vid_len_in_s = max_length
    return output_frames, vid_len_in_s

@torch.inference_mode()
def encode_video_with_siglip2(x: torch.Tensor, model_dict, batch_size: int = -1):
    b, t, c, h, w = x.shape
    if batch_size < 0:
        batch_size = b * t
    x = rearrange(x, "b t c h w -> (b t) c h w")
    outputs = []
    for i in range(0, b * t, batch_size):
        outputs.append(model_dict.siglip2_model.get_image_features(pixel_values=x[i : i + batch_size]))
    res = torch.cat(outputs, dim=0)
    res = rearrange(res, "(b t) d -> b t d", b=b)
    return res

@torch.inference_mode()
def encode_video_with_sync(x: torch.Tensor, model_dict, batch_size: int = -1):
    b, t, c, h, w = x.shape
    assert c == 3 and h == 224 and w == 224
    segment_size = 16
    step_size = 8
    num_segments = (t - segment_size) // step_size + 1
    segments = []
    for i in range(num_segments):
        segments.append(x[:, i * step_size : i * step_size + segment_size])
    x = torch.stack(segments, dim=1).to(model_dict.device)
    outputs = []
    if batch_size < 0:
        batch_size = b * num_segments
    x = rearrange(x, "b s t c h w -> (b s) 1 t c h w")
    for i in range(0, b * num_segments, batch_size):
        with torch.autocast(device_type=model_dict.device.type, enabled=True, dtype=torch.half):
            outputs.append(model_dict.syncformer_model(x[i : i + batch_size]))
    x = torch.cat(outputs, dim=0)
    x = rearrange(x, "(b s) 1 t d -> b (s t) d", b=b)
    return x

@torch.inference_mode()
def encode_video_features(video_path, model_dict):
    visual_features = {}
    frames, _ = get_frames_av(video_path, FPS_VISUAL["siglip2"])
    images = [Image.fromarray(frame).convert('RGB') for frame in frames]
    images = [model_dict.siglip2_preprocess(image) for image in images]
    clip_frames = torch.stack(images).to(model_dict.device).unsqueeze(0)
    visual_features['siglip2_feat'] = encode_video_with_siglip2(clip_frames, model_dict).to(model_dict.device)

    frames, _ = get_frames_av(video_path, FPS_VISUAL["synchformer"])
    images = torch.from_numpy(frames).permute(0, 3, 1, 2)
    sync_frames = model_dict.syncformer_preprocess(images).unsqueeze(0)
    visual_features['syncformer_feat'] = encode_video_with_sync(sync_frames, model_dict)

    vid_len_in_s = sync_frames.shape[1] / FPS_VISUAL["synchformer"]
    visual_features = AttributeDict(visual_features)
    return visual_features, vid_len_in_s

@torch.inference_mode()
def encode_text_feat(text: List[str], model_dict):
    inputs = model_dict.clap_tokenizer(text, padding=True, return_tensors="pt").to(model_dict.device)
    outputs = model_dict.clap_model(**inputs, output_hidden_states=True, return_dict=True)
    return outputs.last_hidden_state, outputs.attentions


def feature_process(video_path, prompt, model_dict, cfg):
    visual_feats, audio_len_in_s = encode_video_features(video_path, model_dict)
    neg_prompt = DEFAULT_NEGATIVE_PROMPT
    prompts = [neg_prompt, prompt]
    text_feat_res, text_feat_mask = encode_text_feat(prompts, model_dict)

    text_feat = text_feat_res[1:]
    uncond_text_feat = text_feat_res[:1]

    if cfg.model_config.model_kwargs.text_length < text_feat.shape[1]:
        text_seq_length = cfg.model_config.model_kwargs.text_length
        text_feat = text_feat[:, :text_seq_length]
        uncond_text_feat = uncond_text_feat[:, :text_seq_length]

    text_feats = AttributeDict({
        'text_feat': text_feat,
        'uncond_text_feat': uncond_text_feat,
    })

    return visual_feats, text_feats, audio_len_in_s

