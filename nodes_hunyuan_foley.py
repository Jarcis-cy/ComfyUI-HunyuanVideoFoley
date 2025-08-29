from __future__ import annotations

import os
import tempfile
import traceback
from typing import Any, Dict, Optional, Tuple, List

import torch
try:
    from huggingface_hub import hf_hub_download
except Exception:
    hf_hub_download = None  # optional until needed

import folder_paths

try:
    from server import PromptServer  # type: ignore
except Exception:
    PromptServer = None  # type: ignore

CATEGORY = "HunyuanVideoFoley"
REPO_ID = "tencent/HunyuanVideo-Foley"
REQUIRED_FILES = [
    "hunyuanvideo_foley.pth",
    "vae_128d_48k.pth",
    "synchformer_state_dict.pth",
    "config.yaml",
]


def _error_msg(msg: str) -> str:
    return f"[ComfyUI-HunyuanVideoFoley] {msg}"


def _comfy_models_root() -> str:
    # Use ComfyUI's built-in models directory
    try:
        return folder_paths.models_dir
    except Exception:
        base = os.path.dirname(folder_paths.get_output_directory())
        return os.path.join(base, "models")


def _ensure_hunyuan_folder_registered() -> str:
    models_dir = _comfy_models_root()
    target = os.path.join(models_dir, "hunyuan_foley")
    os.makedirs(target, exist_ok=True)
    try:
        folder_paths.add_model_folder_path("hunyuan_foley", target, is_default=True)
    except Exception:
        pass
    return target


def _ensure_module_available() -> Optional[str]:
    """Ensure vendor-based inference modules can be imported. 不依赖本地项目目录。"""
    try:
        # Lazy check: only ensure vendor package exists and required pip libs likely installed
        import PIL  # pillow
        import av    # pyav
        import einops
        import torchvision
        import transformers
        import diffusers
        import torchaudio
        import numpy
    except Exception as e:
        return _error_msg(f"依赖缺失: {e}")
    return None


def _load_model(model_dir: str, config_path: str, device: torch.device) -> Tuple[Any, Any]:
    err = _ensure_module_available()
    if err:
        raise RuntimeError(err)
    from .vendor.model_utils import load_model  # type: ignore
    return load_model(model_dir, config_path, device)


def _infer(video_path: str, prompt: str, model_dict: Any, cfg: Any, guidance_scale: float, num_inference_steps: int):
    from .vendor.inference import infer as _infer_impl  # type: ignore
    return _infer_impl(video_path, prompt, model_dict, cfg, guidance_scale=guidance_scale, num_inference_steps=num_inference_steps)


class HunyuanVideoFoleyGenerateAudio:
    @classmethod
    def INPUT_TYPES(cls):
        # Compose model choices from models/hunyuan_foley/* directories that contain required files
        base = _ensure_hunyuan_folder_registered()
        options: List[str] = []
        try:
            for name in os.listdir(base):
                cand = os.path.join(base, name)
                if os.path.isdir(cand):
                    files = set(os.listdir(cand))
                    if all(f in files for f in REQUIRED_FILES):
                        options.append(name)
        except Exception:
            pass
        if not options:
            options = ["default"]
        return {
            "required": {
                "video": ("VIDEO", {"tooltip": "输入视频（VIDEO 类型）"}),
                "prompt": ("STRING", {"multiline": True, "default": "", "tooltip": "音频提示词"}),
                "model": (options, {"default": options[0], "tooltip": "选择 models/hunyuan_foley 下的模型配置目录"}),
                "guidance_scale": ("FLOAT", {"default": 4.5, "min": 0.0, "max": 20.0, "step": 0.1}),
                "num_inference_steps": ("INT", {"default": 50, "min": 1, "max": 1000, "step": 1}),
                "device": ( ["auto", "cpu", "cuda", "mps"], {"default": "auto", "tooltip": "推理设备"} ),
                "gpu_id": ("INT", {"default": 0, "min": 0, "max": 8, "step": 1, "tooltip": "CUDA 设备 ID"}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate"
    CATEGORY = CATEGORY
    DESCRIPTION = "使用 HunyuanVideo-Foley 从视频与文本提示生成音频。"

    _cached: Dict[str, Tuple[Any, Any, torch.device]] = {}

    @staticmethod
    def _get_video_temp_path(video) -> str:
        """Save incoming VIDEO to a temporary file path for the external infer() which expects a file path."""
        # Comfy VIDEO input is an abstraction (VideoInput). Use save_to to a temp file.
        from comfy_api.latest._util import VideoContainer, VideoCodec  # type: ignore
        if not hasattr(video, "save_to"):
            raise RuntimeError(_error_msg("传入的 VIDEO 对象不支持保存，无法导出到本地文件。"))
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tmp.close()
        try:
            video.save_to(tmp.name, format=VideoContainer.MP4, codec=VideoCodec.H264)
        except Exception as e:
            # Cleanup and rethrow
            try:
                os.unlink(tmp.name)
            except Exception:
                pass
            raise RuntimeError(_error_msg(f"视频保存失败: {e}"))
        return tmp.name

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # Let Comfy decide reruns based on inputs; no special caching hash
        return True

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        err = _ensure_module_available()
        if err:
            return err
        return True

    def _get_device(self, device_str: str, gpu_id: int) -> torch.device:
        if device_str == "auto":
            if torch.cuda.is_available():
                return torch.device(f"cuda:{gpu_id}")
            try:
                if torch.backends.mps.is_available():
                    return torch.device("mps")
            except Exception:
                pass
            return torch.device("cpu")
        if device_str == "cuda":
            return torch.device(f"cuda:{gpu_id}")
        return torch.device(device_str)

    def _load_or_get_cached(self, model: str, config_path: str, device: torch.device) -> Tuple[Any, Any]:
        cache_key = f"{model}|{config_path}|{str(device)}"
        if cache_key in self._cached:
            md, cfg, dev = self._cached[cache_key]
            if dev == device:
                return md, cfg
        # fresh load
        md, cfg = _load_model(model, config_path, device)
        self._cached[cache_key] = (md, cfg, device)
        return md, cfg

    def generate(
        self,
        video,
        prompt: str,
        model: str,
        guidance_scale: float,
        num_inference_steps: int,
        device: str,
        gpu_id: int,
        unique_id: Optional[str] = None,
    ):
        # Validate availability
        err = _ensure_module_available()
        if err:
            raise RuntimeError(err)

        device_obj = self._get_device(device, gpu_id)

        if PromptServer is not None and unique_id:
            try:
                PromptServer.instance.send_progress_text("加载 HunyuanVideo-Foley 模型...", unique_id)
            except Exception:
                pass

        # Save VIDEO to a temp path
        try:
            video_path = self._get_video_temp_path(video)
        except Exception as e:
            raise RuntimeError(_error_msg(f"无法保存输入视频到临时文件: {e}"))

        # Resolve model directory and ensure files present (download if missing)
        model_dir = self._resolve_model_dir_and_maybe_download(model, unique_id)
        try:
            cfg_path = os.path.join(model_dir, "config.yaml")
            model_dict, cfg = self._load_or_get_cached(model_dir, cfg_path, device_obj)
        except Exception as e:
            # Clean tmp
            try:
                os.unlink(video_path)
            except Exception:
                pass
            tb = traceback.format_exc()
            raise RuntimeError(_error_msg(f"加载模型失败: {e}\n{tb}"))

    def _resolve_model_dir_and_maybe_download(self, model_choice: str, unique_id: Optional[str]) -> str:
        base = _ensure_hunyuan_folder_registered()
        model_dir = os.path.join(base, model_choice)
        os.makedirs(model_dir, exist_ok=True)
        missing: List[str] = []
        for f in REQUIRED_FILES:
            if not os.path.isfile(os.path.join(model_dir, f)):
                missing.append(f)
        if not missing:
            return model_dir
        if hf_hub_download is None:
            raise RuntimeError(_error_msg("缺少模型文件且未安装 huggingface_hub，请先安装后重试。"))
        # Download missing files
        def progress(text: str):
            if PromptServer is not None and unique_id:
                try:
                    PromptServer.instance.send_progress_text(text, unique_id)
                except Exception:
                    pass
        progress(f"检测到缺失模型文件: {missing}，开始从 {REPO_ID} 下载...")
        # Map repo filenames (they are at repo root with these names)
        for fname in missing:
            try:
                progress(f"下载 {fname} ...")
                local_path = hf_hub_download(repo_id=REPO_ID, filename=fname, local_dir=model_dir, local_dir_use_symlinks=False)
                # Basic integrity: file exists and non-empty
                if (not os.path.isfile(local_path)) or os.path.getsize(local_path) == 0:
                    raise RuntimeError(f"{fname} 下载后为空或不存在")
            except Exception as e:
                raise RuntimeError(_error_msg(f"下载 {fname} 失败: {e}"))
        progress("模型文件下载完成。")
        return model_dir


        if PromptServer is not None and unique_id:
            try:
                PromptServer.instance.send_progress_text("开始生成音频...", unique_id)
            except Exception:
                pass

        # Run inference
        try:
            audio, sr = _infer(video_path, prompt, model_dict, cfg, guidance_scale, num_inference_steps)
        except Exception as e:
            tb = traceback.format_exc()
            raise RuntimeError(_error_msg(f"推理失败: {e}\n{tb}"))
        finally:
            try:
                os.unlink(video_path)
            except Exception:
                pass

        # Convert to Comfy AUDIO type: dict {"waveform": [B,C,T], "sample_rate": int}
        # Hunyuan returns audio Tensor [1, T] or [C, T]? infer returns torchaudio.save(audio, sr) where audio is (channels, time)
        # We standardize to mono [1, T] and then to Comfy [B=1, C=1, T]
        if isinstance(audio, torch.Tensor):
            wav = audio
        else:
            wav = torch.tensor(audio)

        # Expect shape (channels, time)
        if wav.ndim == 1:
            wav = wav.unsqueeze(0)
        if wav.shape[0] > 1:
            # mixdown to mono
            wav = wav.mean(dim=0, keepdim=True)
        # Add batch dimension
        wav = wav.unsqueeze(0)

        audio_out = {"waveform": wav.contiguous(), "sample_rate": int(sr)}

        if PromptServer is not None and unique_id:
            try:
                PromptServer.instance.send_progress_text("音频生成完成。", unique_id)
            except Exception:
                pass

        return (audio_out,)

