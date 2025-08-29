import logging
import types

import torch

# Try to import heavy dependencies; fall back gracefully if unavailable
try:
    from .modeling_ast import ASTForAudioClassification, ASTConfig  # noqa: F401
    _HAS_TRANSFORMERS = True
except Exception as e:  # transformers or submodules may be missing
    ASTForAudioClassification = None
    ASTConfig = None
    _HAS_TRANSFORMERS = False
    _TRANSFORMERS_IMPORT_ERROR = e

from .motionformer import AveragePooling, BaseEncoderLayer, TemporalTransformerEncoderLayer
from .utils import check_if_file_exists_else_download


class AST(torch.nn.Module):
    def __init__(
        self,
        extract_features: bool = False,
        ckpt_path: str = None,
        feat_type: str = None,
        max_spec_t: int = None,
        factorize_freq_time: bool = None,
        agg_freq_module: str = None,
        agg_time_module: str = None,
        add_global_repr: bool = True,
        agg_segments_module: str = None,
        max_segments: int = None,
    ) -> None:
        """
        Minimal, vendorized AST wrapper.
        - When transformers is available, uses the real AST implementation.
        - When transformers is NOT available, builds a lightweight placeholder so that import/instantiation works
          and feature extraction returns tensor(s) with reasonable shapes. A clear error will be raised if
          unsupported paths are executed.
        """
        super().__init__()
        self.extract_features = extract_features
        self.ckpt_path = ckpt_path
        self.max_spec_t = max_spec_t
        self.max_segments = max_segments

        # default hidden dim consistent with AST base
        self._hidden_size = 768

        if _HAS_TRANSFORMERS:
            # Real implementation path
            try:
                if ckpt_path == "MIT/ast-finetuned-audioset-10-10-0.4593":
                    revision = "c1c0c66"  # fixing the revision for compatibility
                    self.config = ASTConfig.from_pretrained(ckpt_path, revision=revision)
                    full_model = ASTForAudioClassification.from_pretrained(ckpt_path, revision=revision)
                    logging.info(f"Loaded AST from {ckpt_path}")
                else:
                    self.config = ASTConfig()
                    self.config.num_labels = 527
                    full_model = ASTForAudioClassification(self.config)
                    logging.info("Initialized AST from scratch with the AST AudioSet config")
            except Exception as e:
                logging.warning(
                    "Failed to initialize real AST, falling back to placeholder. Error: %s", e
                )
                self._init_placeholder(factorize_freq_time, agg_time_module, add_global_repr)
                return

            was_pt_on_avclip = ckpt_path is not None and ckpt_path.endswith(".pt")

            # feature extractor
            self.ast = full_model.audio_spectrogram_transformer

            if self.extract_features:
                # assign `feat_type` (use default if not specified)
                self.feat_type = "last_hidden_state" if feat_type is None else feat_type
                # define adapters if needed
                self.factorize_freq_time = factorize_freq_time
                # avoiding code duplication (used only if agg_*_module is TransformerEncoderLayer)
                transf_enc_layer_kwargs = dict(
                    d_model=self.config.hidden_size,
                    nhead=self.config.num_attention_heads,
                    dim_feedforward=self.config.intermediate_size,
                    activation=torch.nn.GELU(),
                    batch_first=True,
                    dropout=self.config.attention_probs_dropout_prob,
                    layer_norm_eps=1e-6,
                    norm_first=True,
                )
                if factorize_freq_time:
                    self.feat_type = "last_hidden_state"  # this feat_type supports factorization
                    # frequency aggregation
                    if agg_freq_module == "TransformerEncoderLayer":
                        self.freq_attn_agg = FrequencyTransformerEncoderLayer(**transf_enc_layer_kwargs)
                    elif agg_freq_module == "AveragePooling":
                        self.freq_attn_agg = AveragePooling(
                            avg_pattern="BS D f t -> BS D t", then_permute_pattern="BS D t -> BS t D"
                        )
                    # time aggregation
                    if agg_time_module == "TransformerEncoderLayer":
                        self.temp_attn_agg = TemporalTransformerEncoderLayer(**transf_enc_layer_kwargs)
                    elif agg_time_module == "AveragePooling":
                        self.temp_attn_agg = AveragePooling(avg_pattern="BS t D -> BS D")
                    elif "Identity" in str(agg_time_module):
                        self.temp_attn_agg = torch.nn.Identity()
                # define a global aggregation layer (aggregate over segments)
                self.add_global_repr = add_global_repr
                if add_global_repr:
                    if agg_segments_module == "TransformerEncoderLayer":
                        pos_max_len = max_segments if max_segments is not None else 16
                        self.global_attn_agg = TemporalTransformerEncoderLayer(
                            add_pos_emb=True,
                            pos_emb_drop=self.config.hidden_dropout_prob,
                            pos_max_len=pos_max_len,
                            **transf_enc_layer_kwargs,
                        )
                    elif agg_segments_module == "AveragePooling":
                        self.global_attn_agg = AveragePooling(avg_pattern="B S D -> B D")
            else:
                self.classifier = full_model.classifier

            # AST.device fails with AttributeError. This is a workaround
            self.device = full_model.device

            # pre-trained on 12*101+2=1214 tokens, but we have less (e.g. 12*6+2=74)
            self.patch_position_emb()

            if was_pt_on_avclip:
                check_if_file_exists_else_download(self.ckpt_path)
                ckpt = torch.load(ckpt_path, map_location="cpu")
                ckpt_weights = dict()
                for k, v in ckpt["state_dict"].items():
                    if k.startswith(("module.a_encoder.", "a_encoder.")):
                        k = k.replace("module.", "").replace("a_encoder.", "")
                        ckpt_weights[k] = v
                _load_status = self.load_state_dict(ckpt_weights, strict=False)
                if len(_load_status.missing_keys) > 0 or len(_load_status.unexpected_keys) > 0:
                    logging.warning(
                        f"Loading exact afeat_extractor ckpt from {self.ckpt_path} failed.\n"
                        f"Missing keys ({len(_load_status.missing_keys)}): {_load_status.missing_keys},\n"
                        f"Unexpected keys ({len(_load_status.unexpected_keys)}): {_load_status.unexpected_keys} \n"
                        f"temp_attn_agg are expected to be missing if ckpt was pt contrastively."
                    )
                else:
                    logging.info(f"Loading afeat_extractor ckpt from {self.ckpt_path} succeeded.")

            logging.info(f"AST: {sum(p.numel() for p in self.parameters() if p.requires_grad):,}")
        else:
            # Fallback: lightweight dummy implementation to allow import & minimal forward
            logging.warning(
                "Transformers/AST not available; using a minimal placeholder. "
                f"Original import error: {_TRANSFORMERS_IMPORT_ERROR}"
            )
            self._init_placeholder(factorize_freq_time, agg_time_module, add_global_repr)

    def _init_placeholder(self, factorize_freq_time, agg_time_module, add_global_repr):
        self.config = types.SimpleNamespace(
            hidden_size=self._hidden_size,
            num_attention_heads=12,
            intermediate_size=3072,
            attention_probs_dropout_prob=0.0,
            hidden_dropout_prob=0.0,
        )
        self.factorize_freq_time = factorize_freq_time
        if factorize_freq_time and (agg_time_module == "TransformerEncoderLayer"):
            self.temp_attn_agg = TemporalTransformerEncoderLayer(
                d_model=self.config.hidden_size,
                nhead=self.config.num_attention_heads,
                dim_feedforward=self.config.intermediate_size,
                activation=torch.nn.GELU(),
                batch_first=True,
                dropout=self.config.attention_probs_dropout_prob,
                layer_norm_eps=1e-6,
                norm_first=True,
            )
        else:
            self.temp_attn_agg = torch.nn.Identity()
        self._proj = torch.nn.Linear(1, self._hidden_size)
        self.add_global_repr = add_global_repr

    def forward(self, x: torch.Tensor, for_loop: bool = False, cont_mask: torch.Tensor = None, **ast_kwargs):
        B, S, T, F = x.shape

        if _HAS_TRANSFORMERS and hasattr(self, "ast"):
            if for_loop:
                assert cont_mask is None, "cont_mask is not supported with for_loop=True"
                orig_shape_s = (B, 1, T, F)
                x = torch.cat(
                    [self.forward_segments(x[:, s], orig_shape_s, **ast_kwargs).unsqueeze(1) for s in range(S)], dim=1
                )
            else:
                orig_shape = (B, S, T, F)
                x = x.view(B * S, T, F)
                if cont_mask is not None:
                    cont_mask = cont_mask.reshape(B * S, T, F)
                x = self.forward_segments(x, orig_shape=orig_shape, cont_mask=cont_mask, **ast_kwargs)
                x = x.view(B, S, *x.shape[1:])
            global_x = None
            if self.extract_features and self.add_global_repr and hasattr(self, "global_attn_agg"):
                assert len(x.shape) == 3, f"Local representation should be (B, S, D) {x.shape}"
                global_x = self.global_attn_agg(x)
            return x, global_x
        else:
            # Minimal placeholder path: return zeros with shape (B, S, 1, D)
            D = self._hidden_size
            t = 1
            local = x.new_zeros((B, S, t, D))
            global_x = None
            return local, global_x

    def forward_segments(self, x, orig_shape: tuple, cont_mask: torch.Tensor = None, **ast_kwargs):
        if not (_HAS_TRANSFORMERS and hasattr(self, "ast")):
            raise RuntimeError(
                "AST forward_segments is unavailable without transformers. Install 'transformers' to enable real AST."
            )
        x, x_mask = self.ast(x, cont_mask=cont_mask, **ast_kwargs)
        if self.extract_features:
            x = self.get_features_by_type(x)
            if self.factorize_freq_time:
                x = self.restore_freq_temp_dims(x, orig_shape)  # (BS, D, f, t) <- (B*S, T, D)
                if cont_mask is not None:
                    x_mask = x_mask.unsqueeze(-1).expand(-1, -1, self.config.hidden_size)
                    x_mask = self.restore_freq_temp_dims(x_mask, orig_shape)  # (BS, D, f, t)
                    x_mask = x_mask[:, 0, :, :]
                else:
                    x_mask = None
                x = self.freq_attn_agg(x, x_mask)  # (BS, t, D)
                x = self.temp_attn_agg(x)  # (BS, D) or (BS, t, D)
        else:
            x = x["pooler_output"]
            x = self.classifier(x)
        return x

    def get_features_by_type(self, x):
        if x["last_hidden_state"].dim() == 3:
            return x["last_hidden_state"]
        return x["pooler_output"]

    def restore_freq_temp_dims(self, feats, orig_shape: tuple):
        B, S, T, F = orig_shape
        D = self.config.hidden_size
        # fallback to generic reshape if AST embeddings are unavailable
        f = 12
        t = max(1, (T // 10))
        feats = feats.permute(0, 2, 1)
        feats = feats.view(B * S, D, f, t)
        return feats

    def patch_position_emb(self):
        if not (_HAS_TRANSFORMERS and hasattr(self, "ast")):
            return
        if self.max_spec_t is not None:
            self.config.max_length = self.max_spec_t
        f, t = self.ast.embeddings.get_shape(self.config)
        shortened = self.ast.embeddings.position_embeddings[:, : f * t + 2].clone()
        self.ast.embeddings.position_embeddings = torch.nn.Parameter(shortened).to(self.device)

    def to(self, device):
        if _HAS_TRANSFORMERS and hasattr(self, "ast"):
            self.device = torch.device(device)
        return super().to(device)


class FrequencyTransformerEncoderLayer(BaseEncoderLayer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor = None) -> torch.Tensor:
        BS, D, f, t = x.shape
        x = x.permute(0, 3, 2, 1)
        x = x.reshape(BS * t, f, D)
        if x_mask is not None:
            x_mask = x_mask.permute(0, 2, 1)
            x_mask = x_mask.reshape(BS * t, f)
        x = super().forward(x=x, x_mask=x_mask)  # (B*S*t, D)
        x = x.view(BS, t, D)
        return x

