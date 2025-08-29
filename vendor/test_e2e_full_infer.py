import argparse
import os
import sys
from pathlib import Path

import torch
import gc


REQUIRED_FILES = [
    "hunyuanvideo_foley.pth",
    "vae_128d_48k.pth",
    "synchformer_state_dict.pth",
]

HF_LOCAL_MODELS = {
    # local directory names expected under models-root (or its hf_cache)
    "google/siglip2-base-patch16-512": ["google", "siglip2-base-patch16-512"],
    "laion/larger_clap_general": ["laion", "larger_clap_general"],
}


def check_models(models_root: Path):
    missing = []
    for fn in REQUIRED_FILES:
        if not (models_root / fn).exists():
            missing.append(fn)
    # try to find HF models locally
    hf_ok = True
    for repo, parts in HF_LOCAL_MODELS.items():
        local_dir = models_root.joinpath(*parts)
        if not local_dir.exists():
            hf_ok = False
    return missing, hf_ok


def print_device_info():
    """Print detailed device and memory information."""
    print("\n" + "="*50)
    print("设备信息检查")
    print("="*50)

    # CUDA availability
    print(f"CUDA 可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA 设备数量: {torch.cuda.device_count()}")
        print(f"当前 CUDA 设备: {torch.cuda.current_device()}")
        print(f"设备名称: {torch.cuda.get_device_name()}")

        # Memory info
        memory_allocated = torch.cuda.memory_allocated() / 1024**3
        memory_reserved = torch.cuda.memory_reserved() / 1024**3
        memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3

        print(f"显存总量: {memory_total:.2f} GB")
        print(f"已分配显存: {memory_allocated:.2f} GB")
        print(f"已保留显存: {memory_reserved:.2f} GB")
        print(f"可用显存: {memory_total - memory_reserved:.2f} GB")
    else:
        print("CUDA 不可用，将使用 CPU")
    print("="*50 + "\n")


def print_model_device_info(model_dict):
    """Print device information for all models."""
    print("\n" + "="*50)
    print("模型设备分布检查")
    print("="*50)

    for name, model in model_dict.items():
        if hasattr(model, 'device'):
            print(f"{name}: {model.device}")
        elif hasattr(model, 'parameters'):
            try:
                device = next(model.parameters()).device
                print(f"{name}: {device}")
            except StopIteration:
                print(f"{name}: 无参数")
        else:
            print(f"{name}: 非模型对象")

    print("="*50 + "\n")


def monitor_memory_usage(stage_name):
    """Monitor GPU memory usage at different stages."""
    if torch.cuda.is_available():
        memory_allocated = torch.cuda.memory_allocated() / 1024**3
        memory_reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"[{stage_name}] 显存使用: 已分配 {memory_allocated:.2f} GB, 已保留 {memory_reserved:.2f} GB")
    else:
        print(f"[{stage_name}] CPU 模式")


def force_gpu_usage():
    """Force GPU usage and clear cache."""
    if torch.cuda.is_available():
        # Clear cache
        torch.cuda.empty_cache()
        gc.collect()

        # Create a small tensor on GPU to ensure CUDA is initialized
        dummy = torch.randn(1, device='cuda')
        del dummy
        torch.cuda.empty_cache()

        print("GPU 缓存已清理，CUDA 已初始化")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--models-root", default=os.path.join("models", "HunyuanVideoFoley"))
    parser.add_argument("--config", default=None, help="Optional path to config YAML if needed by load_model")
    args = parser.parse_args()

    pass

    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    models_root = Path(args.models_root)
    models_root.mkdir(parents=True, exist_ok=True)

    # Prepare HF offline env to avoid network
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HOME", str(models_root / "hf_cache"))

    # Check required files
    missing, hf_ok = check_models(models_root)
    if missing or not hf_ok:
        print("[error] Missing required models for full inference.")
        if missing:
            print(" - Missing local files:")
            for fn in missing:
                print("   ", models_root / fn)
        if not hf_ok:
            print(" - Missing local HF model directories:")
            for repo, parts in HF_LOCAL_MODELS.items():
                print("   ", models_root.joinpath(*parts), f"(expected for {repo})")
        print("[hint] 请将上述权重与HF模型目录放入 ComfyUI/models/HunyuanVideoFoley/ 后重试。")
        sys.exit(2)

    # Import vendor pipeline (robust to folder name with hyphen)
    vendor_dir = Path(__file__).resolve().parent
    parent_dir = vendor_dir.parent
    if str(parent_dir) not in sys.path:
        sys.path.insert(0, str(parent_dir))
    from vendor import inference as vendor_infer
    from vendor import model_utils as vendor_model_utils

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model dict and cfg
    config_path = args.config
    if config_path is None:
        # 默认寻找 models_root 下的 config.yaml（如项目提供）
        default_cfg = models_root / "config.yaml"
        if default_cfg.exists():
            config_path = str(default_cfg)
        else:
            print("[warning] 未提供配置文件，将尝试使用模型默认配置加载。")
            # vendor_model_utils.load_model 需要 config_path；此处直接失败更清晰
            print("[error] 缺少配置文件 config.yaml。请将其放入:", default_cfg)
            sys.exit(3)

    model_dict, cfg = vendor_model_utils.load_model(str(models_root), config_path, device)

    # Run inference end-to-end
    audio, sr = vendor_infer.infer(args.video, args.prompt, model_dict, cfg)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_wav = out_dir / "hunyuanfoley_e2e.wav"

    try:
        import torchaudio
        torchaudio.save(str(out_wav), audio.unsqueeze(0), sample_rate=sr)
        print("[info] saved:", out_wav)
    except Exception as e:
        print("[error] saving audio failed:", e)
        sys.exit(4)


if __name__ == "__main__":
    main()

