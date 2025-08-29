from __future__ import annotations

import os
import tempfile
import traceback
from typing import Any, Dict, Optional, Tuple, List
from fractions import Fraction

import torch
try:
    from huggingface_hub import hf_hub_download, snapshot_download
except Exception:
    hf_hub_download = None  # optional until needed
    snapshot_download = None

import folder_paths

try:
    from server import PromptServer  # type: ignore
except Exception:
    PromptServer = None  # type: ignore

# 延迟导入 ffmpeg-python，避免启动时的重依赖
try:
    import ffmpeg
except ImportError:
    ffmpeg = None

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


def _ensure_huggingface_models(model_dir: str, progress_callback=None) -> None:
    """确保 HuggingFace 模型（SigLIP2 和 CLAP）已下载到正确位置"""
    if snapshot_download is None:
        raise RuntimeError(_error_msg("缺少 HuggingFace 模型且未安装 huggingface_hub，请先安装后重试。"))

    # 需要的 HuggingFace 模型
    hf_models = [
        ("google/siglip2-base-patch16-512", os.path.join(model_dir, "google", "siglip2-base-patch16-512")),
        ("laion/larger_clap_general", os.path.join(model_dir, "laion", "larger_clap_general")),
    ]

    for repo_id, local_dir in hf_models:
        if not os.path.isdir(local_dir):
            if progress_callback:
                progress_callback(f"下载 HuggingFace 模型: {repo_id} ...")
            try:
                snapshot_download(
                    repo_id=repo_id,
                    local_dir=local_dir,
                    local_dir_use_symlinks=False,
                    resume_download=True
                )
                if progress_callback:
                    progress_callback(f"完成下载: {repo_id}")
            except Exception as e:
                raise RuntimeError(_error_msg(f"下载 HuggingFace 模型 {repo_id} 失败: {e}"))


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

    def _resolve_model_dir_and_maybe_download(self, model_choice: str, unique_id: Optional[str]) -> str:
        base = _ensure_hunyuan_folder_registered()
        model_dir = os.path.join(base, model_choice)
        os.makedirs(model_dir, exist_ok=True)
        missing: List[str] = []
        for f in REQUIRED_FILES:
            if not os.path.isfile(os.path.join(model_dir, f)):
                missing.append(f)

        # 检查 HuggingFace 模型是否存在
        hf_missing = []
        siglip2_dir = os.path.join(model_dir, "google", "siglip2-base-patch16-512")
        clap_dir = os.path.join(model_dir, "laion", "larger_clap_general")
        if not os.path.isdir(siglip2_dir):
            hf_missing.append("google/siglip2-base-patch16-512")
        if not os.path.isdir(clap_dir):
            hf_missing.append("laion/larger_clap_general")

        if not missing and not hf_missing:
            return model_dir
        if hf_hub_download is None or snapshot_download is None:
            raise RuntimeError(_error_msg("缺少模型文件且未安装 huggingface_hub，请先安装后重试。"))

        # Download missing files
        def progress(text: str):
            if PromptServer is not None and unique_id:
                try:
                    PromptServer.instance.send_progress_text(text, unique_id)
                except Exception:
                    pass

        if missing:
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
            progress("主要模型文件下载完成。")

        if hf_missing:
            progress(f"检测到缺失 HuggingFace 模型: {hf_missing}")
            try:
                _ensure_huggingface_models(model_dir, progress)
            except Exception as e:
                raise RuntimeError(_error_msg(f"下载 HuggingFace 模型失败: {e}"))

        return model_dir


class VideoAudioMerger:
    """音视频合并节点，将音频和视频合并为包含音轨的视频文件"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": ("VIDEO", {"tooltip": "输入视频（VIDEO 类型）"}),
                "audio": ("AUDIO", {"tooltip": "输入音频（AUDIO 类型）"}),
                "audio_sync_mode": (
                    ["stretch", "loop", "truncate", "pad_silence"],
                    {
                        "default": "stretch",
                        "tooltip": "音频同步模式：stretch=拉伸匹配视频长度，loop=循环播放，truncate=截断，pad_silence=静音填充"
                    }
                ),
                "video_codec": (
                    ["copy", "libx264", "libx265"],
                    {"default": "libx264", "tooltip": "视频编码器"}
                ),
                "audio_codec": (
                    ["aac", "mp3", "copy"],
                    {"default": "aac", "tooltip": "音频编码器"}
                ),
                "quality": (
                    ["high", "medium", "low"],
                    {"default": "medium", "tooltip": "输出质量"}
                ),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("VIDEO",)
    RETURN_NAMES = ("merged_video",)
    FUNCTION = "merge_audio_video"
    CATEGORY = CATEGORY + "/Audio"
    DESCRIPTION = "将音频和视频合并为包含音轨的视频文件。支持多种音频同步模式和编码选项。"

    @staticmethod
    def _ensure_ffmpeg_available() -> Optional[str]:
        """检查 ffmpeg-python 是否可用"""
        if ffmpeg is None:
            return _error_msg("缺少 ffmpeg-python 依赖，请安装：pip install ffmpeg-python")
        return None

    @staticmethod
    def _get_video_temp_path(video) -> str:
        """将 VIDEO 对象保存为临时文件"""
        from comfy_api.latest._util import VideoContainer, VideoCodec  # type: ignore
        if not hasattr(video, "save_to"):
            raise RuntimeError(_error_msg("传入的 VIDEO 对象不支持保存，无法导出到本地文件。"))

        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tmp.close()
        try:
            video.save_to(tmp.name, format=VideoContainer.MP4, codec=VideoCodec.H264)
        except Exception as e:
            try:
                os.unlink(tmp.name)
            except Exception:
                pass
            raise RuntimeError(_error_msg(f"视频保存失败: {e}"))
        return tmp.name

    @staticmethod
    def _save_audio_temp(audio_dict: Dict[str, Any]) -> str:
        """将 AUDIO 字典保存为临时音频文件"""
        import torchaudio

        waveform = audio_dict["waveform"]  # [B, C, T]
        sample_rate = audio_dict["sample_rate"]

        # 确保音频格式正确
        if waveform.dim() == 3 and waveform.shape[0] == 1:
            waveform = waveform.squeeze(0)  # [C, T]

        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        tmp.close()
        try:
            torchaudio.save(tmp.name, waveform, sample_rate)
        except Exception as e:
            try:
                os.unlink(tmp.name)
            except Exception:
                pass
            raise RuntimeError(_error_msg(f"音频保存失败: {e}"))
        return tmp.name

    @staticmethod
    def _get_video_duration(video_path: str) -> float:
        """获取视频时长（秒）"""
        try:
            probe = ffmpeg.probe(video_path)
            video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
            if video_stream:
                return float(video_stream['duration'])
            return 0.0
        except Exception as e:
            raise RuntimeError(_error_msg(f"获取视频时长失败: {e}"))

    @staticmethod
    def _get_audio_duration(audio_dict: Dict[str, Any]) -> float:
        """获取音频时长（秒）"""
        waveform = audio_dict["waveform"]  # [B, C, T]
        sample_rate = audio_dict["sample_rate"]

        if waveform.dim() == 3:
            samples = waveform.shape[2]
        elif waveform.dim() == 2:
            samples = waveform.shape[1]
        else:
            samples = waveform.shape[0]

        return samples / sample_rate

    @staticmethod
    def _get_quality_settings(quality: str) -> Dict[str, str]:
        """根据质量等级获取编码设置"""
        settings = {
            "high": {
                "video_bitrate": "5000k",
                "audio_bitrate": "320k",
                "crf": "18"
            },
            "medium": {
                "video_bitrate": "2500k",
                "audio_bitrate": "192k",
                "crf": "23"
            },
            "low": {
                "video_bitrate": "1000k",
                "audio_bitrate": "128k",
                "crf": "28"
            }
        }
        return settings.get(quality, settings["medium"])

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return True

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        err = cls._ensure_ffmpeg_available()
        if err:
            return err
        return True

    def _process_audio_sync(self, audio_path: str, video_duration: float, audio_duration: float,
                           sync_mode: str, output_path: str) -> str:
        """根据同步模式处理音频"""
        try:
            audio_input = ffmpeg.input(audio_path)

            if sync_mode == "stretch":
                # 拉伸音频匹配视频长度
                tempo = audio_duration / video_duration
                if abs(tempo - 1.0) > 0.01:  # 只有差异超过1%才处理
                    audio_processed = audio_input.filter('atempo', tempo)
                else:
                    audio_processed = audio_input

            elif sync_mode == "loop":
                # 循环播放音频直到匹配视频长度
                if audio_duration < video_duration:
                    loop_count = int(video_duration / audio_duration) + 1
                    audio_processed = audio_input.filter('aloop', loop=loop_count-1, size=int(audio_duration * 48000))
                    audio_processed = audio_processed.filter('atrim', duration=video_duration)
                else:
                    audio_processed = audio_input.filter('atrim', duration=video_duration)

            elif sync_mode == "truncate":
                # 截断音频匹配视频长度
                audio_processed = audio_input.filter('atrim', duration=video_duration)

            elif sync_mode == "pad_silence":
                # 用静音填充音频匹配视频长度
                if audio_duration < video_duration:
                    silence_duration = video_duration - audio_duration
                    silence = ffmpeg.input('anullsrc=channel_layout=stereo:sample_rate=48000',
                                         f='-f lavfi -t {silence_duration}')
                    audio_processed = ffmpeg.concat(audio_input, silence, v=0, a=1)
                else:
                    audio_processed = audio_input.filter('atrim', duration=video_duration)
            else:
                audio_processed = audio_input

            # 保存处理后的音频到临时文件
            processed_audio_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
            ffmpeg.output(audio_processed, processed_audio_path).overwrite_output().run(quiet=True)
            return processed_audio_path

        except Exception as e:
            raise RuntimeError(_error_msg(f"音频同步处理失败: {e}"))

    def merge_audio_video(
        self,
        video,
        audio: Dict[str, Any],
        audio_sync_mode: str,
        video_codec: str,
        audio_codec: str,
        quality: str,
        unique_id: Optional[str] = None,
    ):
        """合并音频和视频"""
        # 验证依赖
        err = self._ensure_ffmpeg_available()
        if err:
            raise RuntimeError(err)

        if PromptServer is not None and unique_id:
            try:
                PromptServer.instance.send_progress_text("开始音视频合并...", unique_id)
            except Exception:
                pass

        video_path = None
        audio_path = None
        processed_audio_path = None
        output_path = None

        try:
            # 保存输入文件到临时路径
            if PromptServer is not None and unique_id:
                try:
                    PromptServer.instance.send_progress_text("保存输入文件...", unique_id)
                except Exception:
                    pass

            video_path = self._get_video_temp_path(video)
            audio_path = self._save_audio_temp(audio)

            # 获取视频和音频时长
            video_duration = self._get_video_duration(video_path)
            audio_duration = self._get_audio_duration(audio)

            if PromptServer is not None and unique_id:
                try:
                    PromptServer.instance.send_progress_text(
                        f"视频时长: {video_duration:.2f}s, 音频时长: {audio_duration:.2f}s", unique_id
                    )
                except Exception:
                    pass

            # 处理音频同步
            if PromptServer is not None and unique_id:
                try:
                    PromptServer.instance.send_progress_text(f"处理音频同步 ({audio_sync_mode})...", unique_id)
                except Exception:
                    pass

            processed_audio_path = self._process_audio_sync(
                audio_path, video_duration, audio_duration, audio_sync_mode, None
            )

            # 合并音视频
            if PromptServer is not None and unique_id:
                try:
                    PromptServer.instance.send_progress_text("合并音视频...", unique_id)
                except Exception:
                    pass

            output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
            quality_settings = self._get_quality_settings(quality)

            # 构建 ffmpeg 命令
            video_input = ffmpeg.input(video_path)
            audio_input = ffmpeg.input(processed_audio_path)

            # 设置编码参数
            video_args = {}
            audio_args = {}

            if video_codec == "copy":
                video_args['vcodec'] = 'copy'
            elif video_codec == "libx264":
                video_args.update({
                    'vcodec': 'libx264',
                    'crf': quality_settings['crf'],
                    'preset': 'medium'
                })
            elif video_codec == "libx265":
                video_args.update({
                    'vcodec': 'libx265',
                    'crf': quality_settings['crf'],
                    'preset': 'medium'
                })

            if audio_codec == "copy":
                audio_args['acodec'] = 'copy'
            elif audio_codec == "aac":
                audio_args.update({
                    'acodec': 'aac',
                    'audio_bitrate': quality_settings['audio_bitrate']
                })
            elif audio_codec == "mp3":
                audio_args.update({
                    'acodec': 'mp3',
                    'audio_bitrate': quality_settings['audio_bitrate']
                })

            # 执行合并
            ffmpeg.output(
                video_input, audio_input, output_path,
                **video_args, **audio_args
            ).overwrite_output().run(quiet=True)

            if PromptServer is not None and unique_id:
                try:
                    PromptServer.instance.send_progress_text("创建输出视频对象...", unique_id)
                except Exception:
                    pass

            # 创建输出 VIDEO 对象
            from comfy_api.latest._input_impl.video_types import VideoFromFile  # type: ignore
            output_video = VideoFromFile(output_path)

            if PromptServer is not None and unique_id:
                try:
                    PromptServer.instance.send_progress_text("音视频合并完成！", unique_id)
                except Exception:
                    pass

            return (output_video,)

        except Exception as e:
            tb = traceback.format_exc()
            raise RuntimeError(_error_msg(f"音视频合并失败: {e}\n{tb}"))
        finally:
            # 清理临时文件
            for temp_path in [video_path, audio_path, processed_audio_path]:
                if temp_path:
                    try:
                        os.unlink(temp_path)
                    except Exception:
                        pass

