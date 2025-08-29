# ComfyUI-HunyuanVideoFoley

一个将 [HunyuanVideo-Foley](https://github.com/Tencent-Hunyuan/HunyuanVideo-Foley) 推理功能集成到 ComfyUI 的插件节点：从视频 + 文本提示生成音频（AUDIO）。

## 节点
- HunyuanVideo-Foley Generate Audio
  - 输入：
    - video: VIDEO 类型（可接 ComfyUI 的 LoadVideo 或 CreateVideo 输出）
    - prompt: 文字提示
    - config_path: 配置文件路径（默认指向 HunyuanVideo-Foley/HunyuanVideo-Foley/config.yaml）
    - model_path: 模型目录（默认 HunyuanVideo-Foley/HunyuanVideo-Foley）
    - guidance_scale, num_inference_steps, device, gpu_id
  - 输出：
    - audio: AUDIO 类型，结构为 {"waveform": [B=1, C=1, T], "sample_rate": int}
