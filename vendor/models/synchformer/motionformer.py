import logging
from pathlib import Path

import torch

# Optional deps with graceful fallback
try:
    from omegaconf import OmegaConf
    _HAS_OMEGACONF = True
except Exception as e:
    OmegaConf = None
    _HAS_OMEGACONF = False
    _OMEGACONF_IMPORT_ERROR = e

try:
    from timm.layers import trunc_normal_
    _HAS_TIMM = True
except Exception as e:
    _HAS_TIMM = False
    def trunc_normal_(tensor, std=0.02):
        # naive init fallback
        with torch.no_grad():
            return tensor.normal_(mean=0.0, std=std)
    _TIMM_IMPORT_ERROR = e

from torch import nn

from .utils import check_if_file_exists_else_download
from .video_model_builder import VisionTransformer


FILE2URL = {
    # cfg
    "motionformer_224_16x4.yaml": "https://raw.githubusercontent.com/facebookresearch/Motionformer/bf43d50/configs/SSV2/motionformer_224_16x4.yaml",
    "joint_224_16x4.yaml": "https://raw.githubusercontent.com/facebookresearch/Motionformer/bf43d50/configs/SSV2/joint_224_16x4.yaml",
    "divided_224_16x4.yaml": "https://raw.githubusercontent.com/facebookresearch/Motionformer/bf43d50/configs/SSV2/divided_224_16x4.yaml",
    # ckpt
    "ssv2_motionformer_224_16x4.pyth": "https://dl.fbaipublicfiles.com/motionformer/ssv2_motionformer_224_16x4.pyth",
    "ssv2_joint_224_16x4.pyth": "https://dl.fbaipublicfiles.com/motionformer/ssv2_joint_224_16x4.pyth",
    "ssv2_divided_224_16x4.pyth": "https://dl.fbaipublicfiles.com/motionformer/ssv2_divided_224_16x4.pyth",
}


class MotionFormer(VisionTransformer):
    """Vendorized MotionFormer with no-download, no-required-ckpt behavior.
    - If ckpt_path is provided and exists, it will be loaded.
    - If missing or None, we skip weight loading but still build the module graph so forward works.
    """

    def __init__(
        self,
        extract_features: bool = False,
        ckpt_path: str = None,
        factorize_space_time: bool = None,
        agg_space_module: str = None,
        agg_time_module: str = None,
        add_global_repr: bool = True,
        agg_segments_module: str = None,
        max_segments: int = None,
    ):
        self.extract_features = extract_features
        self.ckpt_path = ckpt_path
        self.factorize_space_time = factorize_space_time

        was_pt_on_avclip = False
        if self.ckpt_path is not None and Path(self.ckpt_path).exists():
            ckpt = torch.load(self.ckpt_path, map_location="cpu")
            mformer_ckpt2cfg = {
                "ssv2_motionformer_224_16x4.pyth": "motionformer_224_16x4.yaml",
                "ssv2_joint_224_16x4.pyth": "joint_224_16x4.yaml",
                "ssv2_divided_224_16x4.pyth": "divided_224_16x4.yaml",
            }
            if self.ckpt_path.endswith(tuple(mformer_ckpt2cfg.keys())):
                cfg_fname = mformer_ckpt2cfg[Path(self.ckpt_path).name]
            else:
                # attempt to infer from stored args
                s1_cfg = ckpt.get("args", None)
                if s1_cfg is not None:
                    try:
                        s1_vfeat_extractor_ckpt_path = s1_cfg.model.params.vfeat_extractor.params.ckpt_path
                        if s1_vfeat_extractor_ckpt_path is not None:
                            cfg_fname = mformer_ckpt2cfg[Path(s1_vfeat_extractor_ckpt_path).name]
                        else:
                            cfg_fname = "divided_224_16x4.yaml"
                    except Exception:
                        cfg_fname = "divided_224_16x4.yaml"
                else:
                    cfg_fname = "divided_224_16x4.yaml"
        else:
            cfg_fname = "divided_224_16x4.yaml"

        if cfg_fname in ["motionformer_224_16x4.yaml", "divided_224_16x4.yaml"]:
            pos_emb_type = "separate"
        elif cfg_fname == "joint_224_16x4.yaml":
            pos_emb_type = "joint"
        else:
            pos_emb_type = "separate"

        self.mformer_cfg_path = Path(__file__).absolute().parent / cfg_fname

        # Prepare config: prefer YAML+OmegaConf if available; otherwise synthesize a minimal config
        if _HAS_OMEGACONF and self.mformer_cfg_path.exists():
            mformer_cfg = OmegaConf.load(self.mformer_cfg_path)
            logging.info(f"Loading MotionFormer config from {self.mformer_cfg_path.absolute()}")
            # patch the cfg (from the default cfg defined in the repo `Motionformer/slowfast/config/defaults.py`)
            mformer_cfg.VIT.ATTN_DROPOUT = 0.0
            mformer_cfg.VIT.POS_EMBED = pos_emb_type
            mformer_cfg.VIT.USE_ORIGINAL_TRAJ_ATTN_CODE = True
            mformer_cfg.VIT.APPROX_ATTN_TYPE = "none"
            mformer_cfg.VIT.APPROX_ATTN_DIM = 64
        else:
            # Synthesize a minimal config equivalent to divided_224_16x4.yaml for import-time friendliness
            from types import SimpleNamespace as _NS
            VIT = _NS(
                PATCH_SIZE=16,
                PATCH_SIZE_TEMP=2,
                CHANNELS=3,
                EMBED_DIM=768,
                DEPTH=12,
                NUM_HEADS=12,
                MLP_RATIO=4,
                QKV_BIAS=True,
                VIDEO_INPUT=True,
                TEMPORAL_RESOLUTION=8,
                USE_MLP=True,
                DROP=0.0,
                POS_DROPOUT=0.0,
                DROP_PATH=0.2,
                ATTN_LAYER="divided",
                ATTN_DROPOUT=0.0,
                POS_EMBED=pos_emb_type,
                USE_ORIGINAL_TRAJ_ATTN_CODE=True,
                APPROX_ATTN_TYPE="none",
                APPROX_ATTN_DIM=64,
                HEAD_DROPOUT=0.0,
                HEAD_ACT="tanh",
            )
            DATA = _NS(TRAIN_CROP_SIZE=224)
            MODEL = _NS(NUM_CLASSES=174)
            TRAIN = _NS(DATASET="Ssv2")
            mformer_cfg = _NS(VIT=VIT, DATA=DATA, MODEL=MODEL, TRAIN=TRAIN)

        # finally init VisionTransformer with the cfg
        super().__init__(mformer_cfg)

        # load the ckpt now if ckpt is provided and file exists
        if (self.ckpt_path is not None) and Path(self.ckpt_path).exists():
            model_state = ckpt.get("model_state", None)
            if isinstance(model_state, dict):
                _ckpt_load_status = self.load_state_dict(model_state, strict=False)
                if len(_ckpt_load_status.missing_keys) > 0 or len(_ckpt_load_status.unexpected_keys) > 0:
                    logging.warning(
                        f"Loading exact vfeat_extractor ckpt from {self.ckpt_path} had missing or unexpected keys."
                    )
                else:
                    logging.info(f"Loading vfeat_extractor ckpt from {self.ckpt_path} succeeded.")

        if self.extract_features:
            assert isinstance(self.norm, nn.LayerNorm), "early x[:, 1:, :] may not be safe for per-tr weights"
            self.pre_logits = nn.Identity()
            self.head = nn.Identity()
            self.head_drop = nn.Identity()
            transf_enc_layer_kwargs = dict(
                d_model=self.embed_dim,
                nhead=self.num_heads,
                activation=nn.GELU(),
                batch_first=True,
                dim_feedforward=self.mlp_ratio * self.embed_dim,
                dropout=self.drop_rate,
                layer_norm_eps=1e-6,
                norm_first=True,
            )
            if self.factorize_space_time:
                if agg_space_module == "TransformerEncoderLayer":
                    self.spatial_attn_agg = SpatialTransformerEncoderLayer(**transf_enc_layer_kwargs)
                elif agg_space_module == "AveragePooling":
                    self.spatial_attn_agg = AveragePooling(
                        avg_pattern="BS D t h w -> BS D t", then_permute_pattern="BS D t -> BS t D"
                    )
                if agg_time_module == "TransformerEncoderLayer":
                    self.temp_attn_agg = TemporalTransformerEncoderLayer(**transf_enc_layer_kwargs)
                elif agg_time_module == "AveragePooling":
                    self.temp_attn_agg = AveragePooling(avg_pattern="BS t D -> BS D")
                elif agg_time_module and "Identity" in str(agg_time_module):
                    self.temp_attn_agg = nn.Identity()
                else:
                    self.temp_attn_agg = nn.Identity()
            self.add_global_repr = add_global_repr
            if add_global_repr:
                if agg_segments_module == "TransformerEncoderLayer":
                    pos_max_len = max_segments if max_segments is not None else 16
                    self.global_attn_agg = TemporalTransformerEncoderLayer(
                        add_pos_emb=True,
                        pos_emb_drop=mformer_cfg.VIT.POS_DROPOUT,
                        pos_max_len=pos_max_len,
                        **transf_enc_layer_kwargs,
                    )
                elif agg_segments_module == "AveragePooling":
                    self.global_attn_agg = AveragePooling(avg_pattern="B S D -> B D")

        # patch_embed is not used in MotionFormer, only patch_embed_3d, because cfg.VIT.PATCH_SIZE_TEMP > 1
        # but it used to calculate the number of patches, so we need to keep it
        self.patch_embed.requires_grad_(False)

    def forward(self, x):
        """
        x is of shape (B, S, C, T, H, W)
        """
        B, S, C, T, H, W = x.shape
        orig_shape = (B, S, C, T, H, W)
        x = x.view(B * S, C, T, H, W)
        x = self.forward_segments(x, orig_shape=orig_shape)
        x = x.view(B, S, *x.shape[1:])
        return x

    def forward_segments(self, x, orig_shape: tuple) -> torch.Tensor:
        x, x_mask = self.forward_features(x)
        assert self.extract_features
        x = x[:, 1:, :]  # drop CLS
        x = self.norm(x)
        x = self.pre_logits(x)
        if self.factorize_space_time:
            x = self.restore_spatio_temp_dims(x, orig_shape)
            x = self.spatial_attn_agg(x, x_mask)
            x = self.temp_attn_agg(x)
        return x

    def restore_spatio_temp_dims(self, feats: torch.Tensor, orig_shape: tuple) -> torch.Tensor:
        B, S, C, T, H, W = orig_shape
        D = self.embed_dim
        t = max(1, T // getattr(self.patch_embed_3d, "z_block_size", 2))
        h = getattr(self.patch_embed_3d, "height", H // 16)
        w = getattr(self.patch_embed_3d, "width", W // 16)
        feats = feats.permute(0, 2, 1)
        feats = feats.view(B * S, D, t, h, w)
        return feats


class BaseEncoderLayer(nn.TransformerEncoderLayer):
    def __init__(
        self,
        add_pos_emb: bool = False,
        pos_emb_drop: float = None,
        pos_max_len: int = None,
        *args_transformer_enc,
        **kwargs_transformer_enc,
    ):
        super().__init__(*args_transformer_enc, **kwargs_transformer_enc)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.self_attn.embed_dim))
        trunc_normal_(self.cls_token, std=0.02)
        self.add_pos_emb = add_pos_emb
        if add_pos_emb:
            self.pos_max_len = 1 + pos_max_len
            self.pos_emb = nn.Parameter(torch.zeros(1, self.pos_max_len, self.self_attn.embed_dim))
            self.pos_drop = nn.Dropout(pos_emb_drop if pos_emb_drop is not None else 0.0)
            trunc_normal_(self.pos_emb, std=0.02)
        self.apply(self._init_weights)

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor = None):
        batch_dim = x.shape[0]
        cls_tokens = self.cls_token.expand(batch_dim, -1, -1)
        x = torch.cat((cls_tokens, x), dim=-2)
        if x_mask is not None:
            cls_mask = torch.ones((batch_dim, 1), dtype=torch.bool, device=x_mask.device)
            x_mask_w_cls = torch.cat((cls_mask, x_mask), dim=-1)
            B, N = x_mask_w_cls.shape
            x_mask_w_cls = (
                x_mask_w_cls.reshape(B, 1, 1, N)
                .expand(-1, self.self_attn.num_heads, N, -1)
                .reshape(B * self.self_attn.num_heads, N, N)
            )
            assert x_mask_w_cls.dtype == torch.bool
            x_mask_w_cls = ~x_mask_w_cls
        else:
            x_mask_w_cls = None
        if self.add_pos_emb:
            seq_len = x.shape[1]
            assert self.pos_emb is not None
            assert seq_len <= self.pos_max_len, f"Seq len ({seq_len}) > pos_max_len ({self.pos_max_len})"
            x = x + self.pos_emb[:, :seq_len, :]
            x = self.pos_drop(x)
        x = super().forward(src=x, src_mask=x_mask_w_cls)
        x = x[:, 0, :]
        return x

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"cls_token", "pos_emb"}


class SpatialTransformerEncoderLayer(BaseEncoderLayer):
    def forward(self, x: torch.Tensor, x_mask: torch.Tensor = None) -> torch.Tensor:
        BS, D, t, h, w = x.shape
        x = x.permute(0, 2, 3, 4, 1).reshape(BS * t, h * w, D)
        if x_mask is not None:
            x_mask = x_mask.permute(0, 2, 3, 1).reshape(BS * t, h * w)
        x = super().forward(x=x, x_mask=x_mask)
        x = x.view(BS, t, D)
        return x


class TemporalTransformerEncoderLayer(BaseEncoderLayer):
    def forward(self, x):
        x = super().forward(x)
        return x


class AveragePooling(nn.Module):
    def __init__(self, avg_pattern: str, then_permute_pattern: str = None) -> None:
        super().__init__()
        self.reduce_fn = "mean"
        self.avg_pattern = avg_pattern
        self.then_permute_pattern = then_permute_pattern

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor = None) -> torch.Tensor:
        # simple reduction using view/mean to avoid einops dependency
        if self.avg_pattern == "BS D t h w -> BS D t":
            BS, D, t, h, w = x.shape
            x = x.view(BS, D, t, h * w).mean(dim=-1)
            if self.then_permute_pattern == "BS D t -> BS t D":
                x = x.permute(0, 2, 1)
            return x
        elif self.avg_pattern == "BS t D -> BS D":
            x = x.mean(dim=1)
            return x
        elif self.avg_pattern == "B S D -> B D":
            x = x.mean(dim=1)
            return x
        else:
            # fallback
            return x

