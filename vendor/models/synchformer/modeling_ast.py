# Copyright (c) Tencent.
# Licensed under the Apache License, Version 2.0.
"""
Audio Spectrogram Transformer (AST) model implementation.
Based on Hugging Face transformers library.
"""

import math
import warnings
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, MSELoss

try:
    from transformers import PretrainedConfig, PreTrainedModel
    from transformers.modeling_outputs import BaseModelOutput, SequenceClassifierOutput
    from transformers.modeling_utils import find_pruneable_heads_and_indices, prune_linear_layer
    _HAS_TRANSFORMERS = True
except ImportError:
    # Fallback base classes if transformers is not available
    PretrainedConfig = object
    PreTrainedModel = nn.Module
    BaseModelOutput = dict
    SequenceClassifierOutput = dict
    _HAS_TRANSFORMERS = False


class ASTConfig(PretrainedConfig):
    """
    Configuration class for AST model.
    """
    model_type = "ast"

    def __init__(
        self,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        patch_size=16,
        frequency_stride=10,
        time_stride=10,
        max_length=1024,
        num_mel_bins=128,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.patch_size = patch_size
        self.frequency_stride = frequency_stride
        self.time_stride = time_stride
        self.max_length = max_length
        self.num_mel_bins = num_mel_bins


class ASTEmbeddings(nn.Module):
    """
    Construct the patch and position embeddings for AST.
    """

    def __init__(self, config):
        super().__init__()
        
        self.config = config
        self.patch_embeddings = nn.Conv2d(
            1, config.hidden_size, 
            kernel_size=(config.patch_size, config.patch_size),
            stride=(config.frequency_stride, config.time_stride)
        )
        
        # Calculate number of patches
        self.frequency_out_dimension = (config.num_mel_bins - config.patch_size) // config.frequency_stride + 1
        self.time_out_dimension = (config.max_length - config.patch_size) // config.time_stride + 1
        self.num_patches = self.frequency_out_dimension * self.time_out_dimension
        
        # Add CLS token and distillation token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        self.distillation_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        
        # Position embeddings
        self.position_embeddings = nn.Parameter(
            torch.zeros(1, self.num_patches + 2, config.hidden_size)
        )
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def get_shape(self, config):
        """Get the shape of frequency and time dimensions."""
        f = (config.num_mel_bins - config.patch_size) // config.frequency_stride + 1
        t = (config.max_length - config.patch_size) // config.time_stride + 1
        return f, t

    def forward(self, input_values, cont_mask=None):
        batch_size = input_values.shape[0]
        
        # Create patch embeddings
        embeddings = self.patch_embeddings(input_values)
        embeddings = embeddings.flatten(2).transpose(1, 2)
        
        # Add CLS and distillation tokens
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        distillation_tokens = self.distillation_token.expand(batch_size, -1, -1)
        embeddings = torch.cat((cls_tokens, distillation_tokens, embeddings), dim=1)
        
        # Add position embeddings
        embeddings = embeddings + self.position_embeddings
        embeddings = self.dropout(embeddings)
        
        return embeddings


class ASTSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask=None, head_mask=None, output_attentions=False):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        return outputs


class ASTSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = hidden_states + input_tensor
        return hidden_states


class ASTAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self = ASTSelfAttention(config)
        self.output = ASTSelfOutput(config)

    def forward(self, hidden_states, attention_mask=None, head_mask=None, output_attentions=False):
        self_outputs = self.self(hidden_states, attention_mask, head_mask, output_attentions)
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]
        return outputs


class ASTIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.intermediate_act_fn = nn.GELU()

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class ASTOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = hidden_states + input_tensor
        return hidden_states


class ASTLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = ASTAttention(config)
        self.intermediate = ASTIntermediate(config)
        self.output = ASTOutput(config)
        self.layernorm_before = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.layernorm_after = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states, attention_mask=None, head_mask=None, output_attentions=False):
        self_attention_outputs = self.attention(
            self.layernorm_before(hidden_states),
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]

        hidden_states = attention_output + hidden_states

        layer_output = self.layernorm_after(hidden_states)
        layer_output = self.intermediate(layer_output)
        layer_output = self.output(layer_output, hidden_states)

        outputs = (layer_output,) + outputs
        return outputs


class ASTEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([ASTLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None

            layer_outputs = layer_module(hidden_states, attention_mask, layer_head_mask, output_attentions)

            hidden_states = layer_outputs[0]
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
        
        if _HAS_TRANSFORMERS:
            return BaseModelOutput(
                last_hidden_state=hidden_states,
                hidden_states=all_hidden_states,
                attentions=all_self_attentions,
            )
        else:
            return {
                "last_hidden_state": hidden_states,
                "hidden_states": all_hidden_states,
                "attentions": all_self_attentions,
            }


class ASTModel(PreTrainedModel):
    config_class = ASTConfig

    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.embeddings = ASTEmbeddings(config)
        self.encoder = ASTEncoder(config)
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.init_weights()

    def init_weights(self):
        """Initialize weights."""
        if hasattr(super(), 'init_weights'):
            super().init_weights()
        else:
            # Fallback initialization
            for module in self.modules():
                if isinstance(module, nn.Linear):
                    module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
                    if module.bias is not None:
                        module.bias.data.zero_()
                elif isinstance(module, nn.LayerNorm):
                    module.bias.data.zero_()
                    module.weight.data.fill_(1.0)

    def forward(
        self,
        input_values=None,
        attention_mask=None,
        head_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        cont_mask=None,
        **kwargs
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_values is None:
            raise ValueError("You have to specify input_values")

        embedding_output = self.embeddings(input_values, cont_mask=cont_mask)

        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        if return_dict and _HAS_TRANSFORMERS:
            sequence_output = encoder_outputs.last_hidden_state
        else:
            sequence_output = encoder_outputs[0] if isinstance(encoder_outputs, tuple) else encoder_outputs["last_hidden_state"]
            
        sequence_output = self.layernorm(sequence_output)

        pooled_output = sequence_output.mean(dim=1)

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        if _HAS_TRANSFORMERS:
            return BaseModelOutput(
                last_hidden_state=sequence_output,
                pooler_output=pooled_output,
                hidden_states=encoder_outputs.hidden_states,
                attentions=encoder_outputs.attentions,
            )
        else:
            return {
                "last_hidden_state": sequence_output,
                "pooler_output": pooled_output,
                "hidden_states": encoder_outputs.get("hidden_states"),
                "attentions": encoder_outputs.get("attentions"),
            }


class ASTForAudioClassification(PreTrainedModel):
    config_class = ASTConfig

    def __init__(self, config):
        super().__init__(config)

        self.num_labels = config.num_labels
        self.audio_spectrogram_transformer = ASTModel(config)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    def init_weights(self):
        """Initialize weights."""
        if hasattr(super(), 'init_weights'):
            super().init_weights()
        else:
            # Fallback initialization
            for module in self.modules():
                if isinstance(module, nn.Linear):
                    module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
                    if module.bias is not None:
                        module.bias.data.zero_()

    def forward(
        self,
        input_values=None,
        attention_mask=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        cont_mask=None,
        **kwargs
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.audio_spectrogram_transformer(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cont_mask=cont_mask,
        )

        if return_dict and _HAS_TRANSFORMERS:
            pooled_output = outputs.pooler_output
        else:
            pooled_output = outputs[1] if isinstance(outputs, tuple) else outputs["pooler_output"]

        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = nn.BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        if _HAS_TRANSFORMERS:
            return SequenceClassifierOutput(
                loss=loss,
                logits=logits,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )
        else:
            return {
                "loss": loss,
                "logits": logits,
                "hidden_states": outputs.get("hidden_states"),
                "attentions": outputs.get("attentions"),
            }
