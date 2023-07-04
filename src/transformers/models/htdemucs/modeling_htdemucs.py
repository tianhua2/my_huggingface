# coding=utf-8
# Copyright 2023 Meta AI and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" PyTorch HT Demucs model."""
from dataclasses import dataclass

import math
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn

from ...activations import ACT2FN
from ...modeling_outputs import ModelOutput
from ...modeling_utils import PreTrainedModel

@dataclass
class HtdemucsBaseModelOutput(ModelOutput):
    """
    Base class for Htdemuc's model outputs, with potential hidden states and attentions.

    Args:
        last_freq_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of frequency hidden-states at the output of the last layer of the model.
        last_temp_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of temporal hidden-states at the output of the last layer of the model.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    last_freq_hidden_state: torch.FloatTensor = None
    last_temp_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None

# Copied from transformers.models.bart.modeling_bart._expand_mask
def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)

class HtdemucsAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads

        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )
        self.scaling = self.head_dim**-0.5

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Input shape: Batch x Time x Channel"""

        # if key_value_states are provided this layer is used as a cross-attention layer
        is_cross_attention = key_value_states is not None

        bsz, tgt_len, _ = hidden_states.size()

        # get query proj
        query_states = self.q_proj(hidden_states) * self.scaling

        # get key, value proj
        if is_cross_attention:
            # cross_attentions
            key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
            value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
        else:
            # self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.reshape(*proj_shape)
        value_states = value_states.reshape(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        if output_attentions:
            # this operation is a bit awkward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to be reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None

        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.bmm(attn_probs, value_states)

        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz * self.num_heads, tgt_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)

        # Use the `embed_dim` from the config (stored in the class) rather than `hidden_state` because `attn_output` can be
        # partitioned across GPUs when using tensor-parallelism.
        attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights_reshaped


class HtdemucsEncoderBlock(nn.Module):
    def __init__(self, config: HtdemucsConfig, is_cross_attn=False):
        super().__init__()
        self.embed_dim = config.d_model
        self.is_cross_attn = is_cross_attn

        self.attn = HtdemucsAttention(
            self.embed_dim,
            config.decoder_attention_heads,
            dropout=config.attention_dropout,
        )

        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout

        self.attn_layer_norm = nn.LayerNorm(self.embed_dim)

        if is_cross_attn:
            self.cross_attn_layer_norm = nn.LayerNorm(self.embed_dim)

        self.layer_scale_1 = (
            nn.Parameter(torch.ones(self.embed_dim) * config.layer_scale_init_value)
            if config.attn_layer_scale
            else nn.Identity()
        )  # TODO(SG): can remove attn_layer_scale from config
        self.layer_scale_2 = (
            nn.Parameter(torch.ones(self.embed_dim) * config.layer_scale_init_value)
            if config.attn_layer_scale
            else nn.Identity()
        )

        self.fc1 = nn.Linear(self.embed_dim, config.decoder_ffn_dim)
        self.fc2 = nn.Linear(config.decoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

        self.group_norm = nn.GroupNorm(num_groups=1, num_channels=self.embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            hidden_states (`torch.FloatTensor` of shape `(batch, seq_len, embed_dim)`):
                Inputs to the layer.
            attention_mask (`torch.FloatTensor` of shape `(batch, 1, tgt_len, src_len)`, *optional*):
                Attention mask, where padding elements are indicated by very large negative values.
            encoder_hidden_states (`torch.FloatTensor` of shape `(batch, seq_len, embed_dim)`, *optional*):
                Cross attention input to the layer. Only used for cross attention layers.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        residual = hidden_states
        hidden_states = self.attn_layer_norm(hidden_states)

        if encoder_hidden_states is not None:
            encoder_hidden_states = self.cross_attn_layer_norm(encoder_hidden_states)

        # Cross attention
        hidden_states, attn_weights = self.attn(
            hidden_states=hidden_states,
            key_value_states=encoder_hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
        )
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = self.layer_scale_1 * hidden_states
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)

        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        hidden_states = self.layer_scale_2 * hidden_states
        hidden_states = residual + hidden_states

        hidden_states = self.group_norm(hidden_states)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs


class HtdemucsEncoderLayer(nn.Module):
    def __init__(self, config: HtdemucsConfig):
        super().__init__()

        self.freq_self_attn = HtdemucsEncoderBlock(config)
        self.temp_self_attn = HtdemucsEncoderBlock(config)

        self.freq_cross_attn = HtdemucsEncoderBlock(config, is_cross_attn=True)
        self.temp_self_attn = HtdemucsEncoderBlock(config, is_cross_attn=True)

    def forward(
        self,
        freq_hidden_states: torch.FloatTensor,
        temp_hidden_states: torch.FloatTensor,
        freq_attention_mask: torch.LongTensor = None,
        temp_attention_mask: torch.LongTensor = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
    ):
        all_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None

        freq_layer_outputs = self.freq_self_attn(
            freq_hidden_states, attention_mask=freq_attention_mask, output_attentions=output_attentions
        )
        temp_layer_outputs = self.temp_self_attn(
            temp_hidden_states, attention_mask=temp_attention_mask, output_attentions=output_attentions
        )

        freq_residual = freq_hidden_states

        freq_hidden_states = freq_layer_outputs[0]
        temp_hidden_states = temp_hidden_states[0]

        if output_attentions:
            all_attentions += (freq_layer_outputs[1], temp_layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states += (freq_hidden_states, temp_hidden_states,)

        freq_layer_outputs = self.freq_cross_attn(
            freq_hidden_states,
            attention_mask=freq_attention_mask,
            encoder_hidden_states=temp_hidden_states,
            output_attentions=output_attentions,
        )
        temp_layer_outputs = self.temp_cross_attn(
            temp_hidden_states,
            attention_mask=temp_attention_mask,
            encoder_hidden_states=freq_residual,
            output_attentions=output_attentions,
        )

        freq_hidden_states = freq_layer_outputs[0]
        temp_hidden_states = temp_hidden_states[0]

        if output_attentions:
            all_attentions += (freq_layer_outputs[1], temp_layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states += (freq_hidden_states, temp_hidden_states,)

        return (
            freq_hidden_states,
            temp_hidden_states,
            all_attentions,
            all_hidden_states
        )


class HtdemucsPreTrainedModel(PreTrainedModel):
    config_class = HtdemucsConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        std = self.config.init_std
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, HtdemucsEncoder):
            module.gradient_checkpointing = value


# Copied from transformers.models.musicgen.modeling_musicgen.MusicgenSinusoidalPositionalEmbedding
class HtdemucsSinusoidalPositionalEmbedding(nn.Module):
    """This module produces sinusoidal positional embeddings of any length."""

    def __init__(self, num_positions: int, embedding_dim: int):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.make_weights(num_positions, embedding_dim)

    def make_weights(self, num_embeddings: int, embedding_dim: int):
        emb_weights = self.get_embedding(num_embeddings, embedding_dim)
        if hasattr(self, "weights"):
            # in forward put the weights on the correct dtype and device of the param
            emb_weights = emb_weights.to(dtype=self.weights.dtype, device=self.weights.device)

        self.weights = nn.Parameter(emb_weights)
        self.weights.requires_grad = False
        self.weights.detach_()

    @staticmethod
    def get_embedding(num_embeddings: int, embedding_dim: int):
        """
        Build sinusoidal embeddings. This matches the implementation in tensor2tensor, but differs slightly from the
        description in Section 3.5 of "Attention Is All You Need".
        """
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        emb = torch.arange(num_embeddings, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.cos(emb), torch.sin(emb)], dim=1).view(num_embeddings, -1)
        if embedding_dim % 2 == 1:
            # zero pad
            emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
        return emb.to(torch.get_default_dtype())

    @torch.no_grad()
    def forward(self, input_ids: torch.Tensor, past_key_values_length: int = 0):
        seq_len = input_ids.size(-1)
        # Create the position ids from the input token ids.
        position_ids = (torch.arange(seq_len) + past_key_values_length).to(input_ids.device)
        # expand embeddings if needed
        if seq_len > self.weights.size(0):
            self.make_weights(seq_len + self.offset, self.embedding_dim)
        return self.weights.index_select(0, position_ids.view(-1)).detach()

class Htdemucs2dSinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.max_len = config.max_source_positions
        self.d_model = config.hidden_size
        self.num_stems = config.num_stems

        self.pos_emb = None
        self.extend_pe(torch.tensor(0.0).expand(1, self.max_len))

    def extend_pe(self, x):
        seq_len = x.size(-1)
        # Reset the positional encodings
        if self.pos_emb is not None:
            if self.pos_emb.size(-1) >= seq_len:
                if self.pos_emb.dtype != x.dtype or self.pos_emb.device != x.device:
                    self.pos_emb = self.pos_emb.to(dtype=x.dtype, device=x.device)
                return

        pos_emb = torch.zeros(seq_len, self.num_stems, self.d_model)
        height_position = torch.arange(0, self.num_stems, dtype=torch.float32).unsqueeze(1)
        width_position = torch.arange(0, x.size(1), dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float32) * -(math.log(10000.0) / self.d_model)
        )

        pos_emb[:, :, 0:self.d_model:2] = torch.sin(width_position * div_term)
        pos_emb[:, :, 1:self.d_model:2] = torch.cos(width_position * div_term)
        pos_emb[:, :, self.d_model::2] = torch.sin(height_position * div_term)
        pos_emb[:, :, self.d_model + 1::2] = torch.cos(height_position * div_term)

        self.pos_emb = pos_emb.to(device=x.device, dtype=x.dtype)

    def forward(self, hidden_states: torch.Tensor):
        self.extend_pe(hidden_states)
        seq_len = hidden_states.size(-1)
        relative_position_embeddings = self.pos_emb[:, :, -seq_len:seq_len]
        return relative_position_embeddings


class HtdemucsEncoder(HtdemucsPreTrainedModel):
    def __init__(self, config: HtdemucsConfig):
        super().__init__(config)

        self.dropout = config.dropout
        self.layerdrop = config.encoder_layerdrop

        embed_dim = config.d_model

        self.layers = nn.ModuleList([HtdemucsEncoderLayer(config) for _ in range(config.num_hidden_layers // 2)])

        self.freq_pos_embedding = Htdemucs2dSinusoidalPositionalEmbedding(config)
        self.temp_pos_embedding = HtdemucsSinusoidalPositionalEmbedding(config.max_position_embeddings, embedding_dim=embed_dim)

        self.freq_layernorm_embedding = nn.LayerNorm(embed_dim)
        self.temp_layernorm_embedding = nn.LayerNorm(embed_dim)

        self.freq_layer_norm = nn.LayerNorm(embed_dim)
        self.temp_layer_norm = nn.LayerNorm(embed_dim)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_features: torch.FloatTensor,
        input_values: torch.FloatTensor,
        freq_attention_mask: torch.LongTensor = None,
        temp_attention_mask: torch.LongTensor = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, HtdemucsBaseModelOutput]:
        r"""
        Args:
            input_features (`torch.FloatTensor` of shape `(batch_size, feature_size, sequence_length)`):
                Float values mel features extracted from the raw speech waveform. Raw speech waveform can be obtained by
                loading a `.flac` or `.wav` audio file into an array of type `List[float]` or a `numpy.ndarray`, *e.g.* via
                the soundfile library (`pip install soundfile`). To prepare the array into `input_features`, the
                [`AutoFeatureExtractor`] should be used for extracting the mel features, padding and conversion into a
                tensor of type `torch.FloatTensor`. See [`~HtdemucsProcessor.__call__`]
            input_values (`torch.FloatTensor` of shape `(batch_size, sequence_length)`):
                Float values of input raw speech waveform. Values can be obtained by loading a `.flac` or `.wav` audio file
                into an array of type `List[float]` or a `numpy.ndarray`, *e.g.* via the soundfile library (`pip install
                soundfile`). To prepare the array into `input_values`, the [`AutoProcessor`] should be used for padding and
                conversion into a tensor of type `torch.FloatTensor`. See [`HtdemucsProcessor.__call__`] for details.
            freq_attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on spatial (frequency) padding token indices. Mask values selected
                in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            temp_attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on temporal padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        freq_positions = self.freq_pos_embedding(input_features)
        temp_positions = self.temp_pos_embedding(input_values)

        freq_hidden_states = input_features + freq_positions.to(input_features.device)
        temp_hidden_states = input_values + temp_positions.to(input_values.device)

        freq_hidden_states = self.freq_layernorm_embedding(freq_hidden_states)
        temp_hidden_states = self.temp_layernorm_embedding(temp_hidden_states)

        freq_hidden_states = nn.functional.dropout(freq_hidden_states, p=self.dropout, training=self.training)
        temp_hidden_states = nn.functional.dropout(temp_hidden_states, p=self.dropout, training=self.training)

        # expand attention_mask from [bsz, seq_len] to [bsz, 1, tgt_seq_len, src_seq_len]
        if freq_attention_mask is not None:
            freq_attention_mask = _expand_mask(freq_attention_mask, freq_hidden_states.dtype)
        if temp_attention_mask is not None:
            temp_attention_mask = _expand_mask(temp_attention_mask, freq_hidden_states.dtype)

        all_hidden_states = (freq_hidden_states, temp_hidden_states,) if output_hidden_states else None
        all_attentions = () if output_attentions else None

        for idx, layer in enumerate(self.layers):
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            to_drop = False
            if self.training:
                dropout_probability = torch.rand([])
                if dropout_probability < self.layerdrop:  # skip the layer
                    to_drop = True

            if to_drop:
                layer_outputs = (None, None, (None,), (None,))

            else:
                if self.gradient_checkpointing and self.training:

                    def create_custom_forward(module):
                        def custom_forward(*inputs):
                            return module(*inputs, output_attentions, output_hidden_states)

                        return custom_forward

                    layer_outputs = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(layer),
                        freq_hidden_states,
                        temp_hidden_states,
                        freq_attention_mask,
                        temp_attention_mask,
                    )
                else:
                    layer_outputs = layer(
                        freq_hidden_states,
                        temp_hidden_states,
                        freq_attention_mask=freq_attention_mask,
                        temp_attention_mask=temp_attention_mask,
                        output_attentions=output_attentions,
                        output_hidden_states=output_hidden_states,
                    )
                freq_hidden_states = layer_outputs[0]
                temp_hidden_states = layer_outputs[1]

            if output_attentions:
                all_attentions = all_attentions + layer_outputs[2]

            if output_hidden_states:
                all_hidden_states = all_hidden_states + layer_outputs[3]

            if not return_dict:
                return tuple(v for v in [freq_hidden_states, temp_hidden_states, all_hidden_states, all_attentions] if v is not None)
            return HtdemucsBaseModelOutput(
                last_freq_hidden_state=freq_hidden_states, last_temp_hidden_state=temp_hidden_states, hidden_states=all_hidden_states, attentions=all_attentions,
            )