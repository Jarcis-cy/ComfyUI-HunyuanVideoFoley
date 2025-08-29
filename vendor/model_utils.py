from __future__ import annotations

import os
import torch
from torchvision import transforms
from torchvision.transforms import v2
from transformers import AutoTokenizer, AutoModel, ClapTextModelWithProjection, AutoImageProcessor
from tqdm import tqdm

from .config_utils import load_yaml, AttributeDict
from .schedulers.scheduling_flow_match_discrete import FlowMatchDiscreteScheduler
from .models.hifi_foley import HunyuanVideoFoley
from .models.dac_vae.model.dac import DAC
# Prefer original project's Synchformer if available (to strictly match official weights)
try:
    from hunyuanvideo_foley.models.synchformer.synchformer import Synchformer as OriginalSynchformer
    _USE_ORIGINAL_SYNCHFORMER = True
except Exception:
    from .models.synchformer.synchformer import Synchformer as OriginalSynchformer
    _USE_ORIGINAL_SYNCHFORMER = False


def load_state_dict(model, model_path):
    state_dict = torch.load(model_path, map_location=lambda storage, loc: storage, weights_only=False)
    model.load_state_dict(state_dict, strict=True)
    return model


def load_model(model_path, config_path, device):
    cfg = load_yaml(config_path)

    foley_model = HunyuanVideoFoley(cfg, dtype=torch.bfloat16, device=device).to(device=device, dtype=torch.bfloat16)
    foley_model = load_state_dict(foley_model, os.path.join(model_path, "hunyuanvideo_foley.pth"))
    foley_model.eval()


    dac_path = os.path.join(model_path, "vae_128d_48k.pth")
    dac_model = DAC.load(dac_path)
    dac_model = dac_model.to(device)
    dac_model.requires_grad_(False)
    dac_model.eval()



    siglip2_preprocess = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    # Prefer local HF model directories under models_root (offline-friendly)
    siglip2_dir = os.path.join(model_path, "google", "siglip2-base-patch16-512")
    clap_dir = os.path.join(model_path, "laion", "larger_clap_general")

    if not os.path.isdir(siglip2_dir):
        raise FileNotFoundError(f"Missing local SigLIP2 dir: {siglip2_dir}")
    if not os.path.isdir(clap_dir):
        raise FileNotFoundError(f"Missing local CLAP dir: {clap_dir}")

    siglip2_model = AutoModel.from_pretrained(siglip2_dir, local_files_only=True).to(device).eval()

    clap_tokenizer = AutoTokenizer.from_pretrained(clap_dir, local_files_only=True)
    clap_model = ClapTextModelWithProjection.from_pretrained(clap_dir, local_files_only=True).to(device)

    # 转换函数：将原始 Synchformer/ViT 风格键名映射到 vendor 的 nn.MultiheadAttention/MLP 键名
    def _convert_synchformer_state_dict_for_vendor(sd: dict, model: torch.nn.Module) -> dict:
        mapped = {}
        for k, v in sd.items():
            nk = k
            if ".attn.qkv.weight" in nk:
                nk = nk.replace(".attn.qkv.weight", ".attn.in_proj_weight")
            elif ".attn.qkv.bias" in nk:
                nk = nk.replace(".attn.qkv.bias", ".attn.in_proj_bias")
            elif ".attn.proj.weight" in nk:
                nk = nk.replace(".attn.proj.weight", ".attn.out_proj.weight")
            elif ".attn.proj.bias" in nk:
                nk = nk.replace(".attn.proj.bias", ".attn.out_proj.bias")

            if ".mlp.fc1.weight" in nk:
                nk = nk.replace(".mlp.fc1.weight", ".mlp.0.weight")
            elif ".mlp.fc1.bias" in nk:
                nk = nk.replace(".mlp.fc1.bias", ".mlp.0.bias")
            elif ".mlp.fc2.weight" in nk:
                nk = nk.replace(".mlp.fc2.weight", ".mlp.3.weight")
            elif ".mlp.fc2.bias" in nk:
                nk = nk.replace(".mlp.fc2.bias", ".mlp.3.bias")

            mapped[nk] = v
        target_keys = set(model.state_dict().keys())
        return {k: v for k, v in mapped.items() if k in target_keys}

    syncformer_path = os.path.join(model_path, "synchformer_state_dict.pth")
    syncformer_preprocess = v2.Compose([
        v2.Resize(224, interpolation=v2.InterpolationMode.BICUBIC),
        v2.CenterCrop(224),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    syncformer_model = OriginalSynchformer()
    raw_sd = torch.load(syncformer_path, weights_only=False, map_location="cpu")
    if _USE_ORIGINAL_SYNCHFORMER:
        # 官方实现：严格加载失败直接报错
        syncformer_model.load_state_dict(raw_sd, strict=True)
    else:
        # vendor 实现：做键名转换后严格加载失败直接报错
        state_dict = _convert_synchformer_state_dict_for_vendor(raw_sd, syncformer_model)
        syncformer_model.load_state_dict(state_dict, strict=True)

    syncformer_model = syncformer_model.to(device).eval()

    model_dict = AttributeDict({
        'foley_model': foley_model,
        'dac_model': dac_model,
        'siglip2_preprocess': siglip2_preprocess,
        'siglip2_model': siglip2_model,
        'clap_tokenizer': clap_tokenizer,
        'clap_model': clap_model,
        'syncformer_preprocess': syncformer_preprocess,
        'syncformer_model': syncformer_model,
        'device': device,
    })
    return model_dict, cfg


def retrieve_timesteps(scheduler, num_inference_steps, device, **kwargs):
    scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
    timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


def prepare_latents(scheduler, batch_size, num_channels_latents, length, dtype, device):
    from diffusers.utils.torch_utils import randn_tensor
    shape = (batch_size, num_channels_latents, int(length))
    latents = randn_tensor(shape, device=device, dtype=dtype)
    if hasattr(scheduler, "init_noise_sigma"):
        latents = latents * scheduler.init_noise_sigma
    return latents


@torch.no_grad()
def denoise_process(visual_feats, text_feats, audio_len_in_s, model_dict, cfg, guidance_scale=4.5, num_inference_steps=50, batch_size=1):
    target_dtype = model_dict.foley_model.dtype
    autocast_enabled = target_dtype != torch.float32
    device = model_dict.device

    scheduler = FlowMatchDiscreteScheduler(
        shift=cfg.diffusion_config.sample_flow_shift,
        reverse=cfg.diffusion_config.flow_reverse,
        solver=cfg.diffusion_config.flow_solver,
        use_flux_shift=cfg.diffusion_config.sample_use_flux_shift,
        flux_base_shift=cfg.diffusion_config.flux_base_shift,
        flux_max_shift=cfg.diffusion_config.flux_max_shift,
    )

    timesteps, num_inference_steps = retrieve_timesteps(scheduler, num_inference_steps, device)

    latents = prepare_latents(
        scheduler,
        batch_size=batch_size,
        num_channels_latents=cfg.model_config.model_kwargs.audio_vae_latent_dim,
        length=audio_len_in_s * cfg.model_config.model_kwargs.audio_frame_rate,
        dtype=target_dtype,
        device=device,
    )

    # Denoise loop with progress bar
    for i, t in tqdm(enumerate(timesteps), total=len(timesteps), desc="Denoising steps"):
        latent_input = torch.cat([latents] * 2) if guidance_scale > 1.0 else latents
        latent_input = scheduler.scale_model_input(latent_input, t)
        t_expand = t.repeat(latent_input.shape[0])

        siglip2_feat = visual_feats.siglip2_feat.repeat(batch_size, 1, 1)
        uncond_siglip2_feat = model_dict.foley_model.get_empty_clip_sequence(bs=batch_size, len=siglip2_feat.shape[1]).to(device)
        if guidance_scale is not None and guidance_scale > 1.0:
            siglip2_feat_input = torch.cat([uncond_siglip2_feat, siglip2_feat], dim=0)
        else:
            siglip2_feat_input = siglip2_feat

        syncformer_feat = visual_feats.syncformer_feat.repeat(batch_size, 1, 1)
        uncond_syncformer_feat = model_dict.foley_model.get_empty_sync_sequence(bs=batch_size, len=syncformer_feat.shape[1]).to(device)
        if guidance_scale is not None and guidance_scale > 1.0:
            syncformer_feat_input = torch.cat([uncond_syncformer_feat, syncformer_feat], dim=0)
        else:
            syncformer_feat_input = syncformer_feat

        text_feat_repeated = text_feats.text_feat.repeat(batch_size, 1, 1)
        uncond_text_feat_repeated = text_feats.uncond_text_feat.repeat(batch_size, 1, 1)
        if guidance_scale is not None and guidance_scale > 1.0:
            text_feat_input = torch.cat([uncond_text_feat_repeated, text_feat_repeated], dim=0)
        else:
            text_feat_input = text_feat_repeated

        with torch.autocast(device_type=device.type, enabled=autocast_enabled, dtype=target_dtype):
            noise_pred = model_dict.foley_model(
                x=latent_input,
                t=t_expand,
                cond=text_feat_input,
                clip_feat=siglip2_feat_input,
                sync_feat=syncformer_feat_input,
                return_dict=True,
            )["x"]
        noise_pred = noise_pred.to(dtype=torch.float32)
        if guidance_scale is not None and guidance_scale > 1.0:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
        latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]

    with torch.no_grad():
        audio = model_dict.dac_model.decode(latents)
        audio = audio.float().cpu()
    audio = audio[:, : int(audio_len_in_s * model_dict.dac_model.sample_rate)]
    return audio, model_dict.dac_model.sample_rate

