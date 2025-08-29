# ComfyUI-HunyuanVideoFoley

[English](README.md) | 简体中文

一个将 [HunyuanVideo-Foley](https://github.com/Tencent-Hunyuan/HunyuanVideo-Foley) 推理功能集成到 ComfyUI 的插件，提供“基于视频+文本提示的音频生成”和“音视频合并”能力。

## 功能特性

- 🎵 首次使用自动从 HuggingFace 下载所需模型文件
- 🎬 基于视频内容与文本提示生成匹配的音频
- 🔧 合并音频与视频，输出包含音轨的视频文件

## 节点说明

### 1. HunyuanVideo-Foley Generate Audio
从视频和文本提示生成音频的核心节点。

输入参数：
- `video`: VIDEO 类型（支持 LoadVideo 或 CreateVideo 输出）
- `prompt`: 文本提示词，描述期望的音频效果
- `model`: 模型配置选择（自动检测 `models/hunyuan_foley` 下的可用模型）
- `guidance_scale`: 引导强度（默认 4.5）
- `num_inference_steps`: 推理步数（默认 50）
- `device`: 推理设备（auto/cpu/cuda/mps）
- `gpu_id`: CUDA 设备 ID

输出：
- `audio`: `{ "waveform": [B=1, C=1, T], "sample_rate": int }`

### 2. Video Audio Merger
将音频和视频合并为包含音轨的视频文件。

输入参数：
- `video`: VIDEO 类型输入视频
- `audio`: AUDIO 类型输入音频
- `audio_sync_mode`: 音频同步模式：`stretch` / `loop` / `truncate` / `pad_silence`
- `video_codec`: 视频编码器（`copy` / `libx264` / `libx265`）
- `audio_codec`: 音频编码器（`aac` / `mp3` / `copy`）
- `quality`: 输出质量（`high` / `medium` / `low`）

输出：
- `merged_video`: 合并后的视频（VIDEO）

## 使用方法

```
LoadVideo → HunyuanVideoFoleyGenerateAudio → VideoAudioMerger → SaveVideo
                    ↑                              ↑
                 文本提示词                       音频输入
```

1. 载入一个视频（LoadVideo）
2. 使用 “HunyuanVideo-Foley Generate Audio” 生成音频
3. 使用 “Video Audio Merger” 合并音视频
4. 保存或预览最终视频

## 安装依赖

```bash
# 使用 ComfyUI 的 Python 环境
D:\AI\ComfyUI-aki-v1.3\.ext\python.exe -m pip install ffmpeg-python
```

## 注意事项

- 首次使用会自动下载模型文件，请确保网络连接正常
- 需要安装 FFmpeg 用于音视频处理
- 推荐使用 CUDA 设备以获得更好的性能

