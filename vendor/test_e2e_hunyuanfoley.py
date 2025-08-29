import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# Optional imports with graceful messages
try:
    import av  # PyAV for decoding
except Exception as e:
    av = None
    _AV_IMPORT_ERROR = e

try:
    import torchaudio
    from torchaudio.transforms import Resample, MelSpectrogram
except Exception as e:
    torchaudio = None
    Resample = None
    MelSpectrogram = None
    _TA_IMPORT_ERROR = e


def decode_video_frames(video_path: str, num_frames: int = 8, out_size: int = 224):
    assert av is not None, f"PyAV not available: {_AV_IMPORT_ERROR}"
    container = av.open(video_path)
    # pick the first video stream
    vstream = next(s for s in container.streams if s.type == "video")
    frames = []
    for frame in container.decode(vstream):
        img = frame.to_ndarray(format="rgb24")  # (H,W,3), uint8
        frames.append(img)
        if len(frames) >= 64:  # limit reading for speed
            break
    container.close()
    if len(frames) == 0:
        raise RuntimeError("No video frames decoded.")
    # Uniformly sample num_frames
    idx = np.linspace(0, len(frames) - 1, num=num_frames).astype(int).tolist()
    sampled = [frames[i] for i in idx]
    x = torch.from_numpy(np.stack(sampled, axis=0)).float() / 255.0  # (T,H,W,3)
    x = x.permute(0, 3, 1, 2)  # (T,3,H,W)
    # resize to 224
    x = F.interpolate(x, size=(out_size, out_size), mode="bilinear", align_corners=False, antialias=True)
    # pack to (B,S,T,C,H,W) -> (1,1,T,3,224,224)
    x = x.unsqueeze(0).unsqueeze(0)  # (1,1,T,3,224,224)
    return x.contiguous()


def decode_audio_mono(video_path: str, target_sr: int = 16000):
    assert av is not None, f"PyAV not available: {_AV_IMPORT_ERROR}"
    assert torchaudio is not None, f"torchaudio not available: {_TA_IMPORT_ERROR}"
    container = av.open(video_path)
    astream = None
    for s in container.streams:
        if s.type == "audio":
            astream = s
            break
    if astream is None:
        container.close()
        raise RuntimeError("No audio stream in input video.")
    samples = []
    sr = astream.rate
    for frame in container.decode(astream):
        arr = frame.to_ndarray()  # shape: (channels, samples)
        if arr.ndim == 1:
            arr = arr[np.newaxis, :]
        # average channels -> mono
        mono = arr.mean(axis=0)
        samples.append(mono)
        if sum(len(s) for s in samples) >= target_sr * 30:  # cap ~30s
            break
    container.close()
    if not samples:
        raise RuntimeError("No audio frames decoded.")
    wav = np.concatenate(samples, axis=0).astype(np.float32)
    # normalize int16-like ranges if needed
    if wav.max() > 1.5 or wav.min() < -1.5:
        wav = wav / 32768.0
    x = torch.from_numpy(wav).unsqueeze(0)  # (1, t)
    if sr != target_sr:
        x = Resample(sr, target_sr)(x)
        sr = target_sr
    return x.squeeze(0), sr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default=r"C:\\Users\\jarcis-cy\\Videos\\WanVideo2_2_I2V_00038.mp4")
    parser.add_argument("--prompt", default="Magma churns and boils, The sizzle of frying meat.")
    parser.add_argument("--out", default=r"C:\\Users\\jarcis-cy\\Videos\\")
    parser.add_argument("--models-root", default=os.path.join("models", "HunyuanVideoFoley"))
    args = parser.parse_args()

    warnings.filterwarnings("default")

    # ensure repo root on sys.path (add both d:/AI/ComfyUI-aki-v1.3 and current cwd)
    repo_root = Path(__file__).resolve().parents[2]
    for p in [str(repo_root), os.getcwd()]:
        if p not in sys.path:
            sys.path.insert(0, p)

    # import vendorized Synchformer
    mod = __import__(
        "custom_nodes.ComfyUI-HunyuanVideoFoley.vendor.models.synchformer.synchformer",
        fromlist=["Synchformer", "encode_audio_with_sync"],
    )
    Synchformer = getattr(mod, "Synchformer")
    encode_audio_with_sync = getattr(mod, "encode_audio_with_sync")

    print("[info] models root:", args.models_root)
    print("[info] video:", args.video)
    print("[info] prompt:", args.prompt)
    print("[info] out dir:", args.out)

    # setup dirs, no downloads performed
    Path(args.models_root).mkdir(parents=True, exist_ok=True)
    Path(args.out).mkdir(parents=True, exist_ok=True)

    # build model
    model = Synchformer()
    model.eval()

    # decode inputs
    if not Path(args.video).exists():
        print("[error] input video not found:", args.video)
        sys.exit(2)

    vis = decode_video_frames(args.video, num_frames=8, out_size=224)
    print("[info] vis shape:", tuple(vis.shape))

    try:
        wav, sr = decode_audio_mono(args.video, target_sr=16000)
        print("[info] audio shape:", tuple(wav.shape), "sr:", sr)
    except Exception as e:
        print("[warning] audio decode failed (will generate synthetic audio):", e)
        sr = 16000
        t = int(5 * sr)
        wav = torch.zeros(t)

    # visual features
    with torch.no_grad():
        vfeats = model.extract_vfeats(vis)
    print("[info] vfeats:", tuple(vfeats.shape), "finite:", torch.isfinite(vfeats).all().item())

    # audio features using helper (build mel)
    if torchaudio is not None:
        mel = MelSpectrogram(sample_rate=16000, win_length=400, hop_length=160, n_fft=1024, n_mels=128)
        with torch.no_grad():
            afeats = encode_audio_with_sync(model, wav.unsqueeze(0), mel)  # (B,S,t,D) or (B,S,1,D)
        print("[info] afeats:", tuple(afeats.shape), "finite:", torch.isfinite(afeats).all().item())
    else:
        afeats = torch.zeros((1, 1, 1, 768))
        print("[warning] torchaudio missing; afeats is zeros.")

    # optional compare_v_a
    logits = None
    try:
        with torch.no_grad():
            logits = model.compare_v_a(vfeats, afeats)
        print("[info] compare_v_a logits shape:", tuple(logits.shape))
    except Exception as e:
        print("[warning] compare_v_a not executed:", e)

    # save out audio (passthrough or silence) as proof of pipeline
    out_wav = os.path.join(args.out, f"synchformer_e2e_{int(time.time())}.wav")
    if torchaudio is not None:
        y = wav.unsqueeze(0)  # (1, t)
        try:
            torchaudio.save(out_wav, y, sample_rate=sr)
            print("[info] saved audio:", out_wav)
        except Exception as e:
            print("[error] failed to save audio:", e)
            sys.exit(3)
    else:
        print("[warning] torchaudio not available; cannot save wav.")

    # write a small report json next to output
    report = {
        "video": args.video,
        "out": out_wav,
        "prompt": args.prompt,
        "vis_shape": tuple(vis.shape),
        "vfeats_shape": tuple(vfeats.shape),
        "afeats_shape": tuple(afeats.shape),
        "logits_shape": tuple(logits.shape) if isinstance(logits, torch.Tensor) else None,
    }
    rpt_path = os.path.join(args.out, f"synchformer_e2e_{int(time.time())}.json")
    try:
        with open(rpt_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print("[info] report:", rpt_path)
    except Exception as e:
        print("[warning] failed to write report:", e)

    print("[done] E2E test completed.")


if __name__ == "__main__":
    main()

