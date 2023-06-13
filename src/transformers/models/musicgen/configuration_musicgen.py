# coding=utf-8
# Copyright 2023 Facebook and The HuggingFace Inc. team. All rights reserved.
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
""" Musicgen model configuration"""
import copy

from ...configuration_utils import PretrainedConfig
from ...utils import logging


logger = logging.get_logger(__name__)

MUSICGEN_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "facebook/musicgen-small": "https://huggingface.co/facebook/musicgen-small/resolve/main/config.json",
    # See all Musicgen models at https://huggingface.co/models?filter=musicgen
}


class MusicgenEncoderConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`MusicgenEncoder`], which is the same as that of
    the [`T5EncoderModel`]. It is used to instantiate a MusicGen encoder according to the specified arguments, defining
    the model architecture. Instantiating a configuration with the defaults will yield a similar configuration to that
    of the MusicGen [facebook/musicgen-small](https://huggingface.co/facebook/musicgen-small) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Arguments:
        vocab_size (`int`, *optional*, defaults to 32128):
            Vocabulary size of the T5 model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`T5Model`] or [`TFT5Model`].
        d_model (`int`, *optional*, defaults to 512):
            Size of the encoder layers and the pooler layer.
        d_kv (`int`, *optional*, defaults to 64):
            Size of the key, query, value projections per attention head. The `inner_dim` of the projection layer will
            be defined as `num_heads * d_kv`.
        d_ff (`int`, *optional*, defaults to 2048):
            Size of the intermediate feed forward layer in each `MusicgenEncoderBlock`.
        num_layers (`int`, *optional*, defaults to 6):
            Number of hidden layers in the Transformer encoder.
        num_heads (`int`, *optional*, defaults to 8):
            Number of attention heads for each attention layer in the Transformer encoder.
        relative_attention_num_buckets (`int`, *optional*, defaults to 32):
            The number of buckets to use for each attention layer.
        relative_attention_max_distance (`int`, *optional*, defaults to 128):
            The maximum distance of the longer sequences for the bucket separation.
        dropout_rate (`float`, *optional*, defaults to 0.1):
            The ratio for all dropout layers.
        layer_norm_eps (`float`, *optional*, defaults to 1e-6):
            The epsilon used by the layer normalization layers.
        initializer_factor (`float`, *optional*, defaults to 1):
            A factor for initializing all weight matrices (should be kept to 1, used internally for initialization
            testing).
        feed_forward_proj (`string`, *optional*, defaults to `"relu"`):
            Type of feed forward layer to be used. Should be one of `"relu"` or `"gated-gelu"`.
    """
    model_type = "musicgen_encoder"
    keys_to_ignore_at_inference = ["past_key_values"]
    attribute_map = {"hidden_size": "d_model", "num_attention_heads": "num_heads", "num_hidden_layers": "num_layers"}

    def __init__(
        self,
        vocab_size=32128,
        d_model=512,
        d_kv=64,
        d_ff=2048,
        num_layers=6,
        num_heads=8,
        relative_attention_num_buckets=32,
        relative_attention_max_distance=128,
        dropout_rate=0.1,
        layer_norm_epsilon=1e-6,
        initializer_factor=1.0,
        feed_forward_proj="relu",
        is_encoder_decoder=False,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.d_kv = d_kv
        self.d_ff = d_ff
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.relative_attention_num_buckets = relative_attention_num_buckets
        self.relative_attention_max_distance = relative_attention_max_distance
        self.dropout_rate = dropout_rate
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_factor = initializer_factor
        self.feed_forward_proj = feed_forward_proj

        act_info = self.feed_forward_proj.split("-")
        self.dense_act_fn = act_info[-1]
        self.is_gated_act = act_info[0] == "gated"

        if len(act_info) > 1 and act_info[0] != "gated" or len(act_info) > 2:
            raise ValueError(
                f"`feed_forward_proj`: {feed_forward_proj} is not a valid activation function of the dense layer."
                "Please make sure `feed_forward_proj` is of the format `gated-{ACT_FN}` or `{ACT_FN}`, e.g. "
                "'gated-gelu' or 'relu'"
            )

        # for backwards compatibility
        if feed_forward_proj == "gated-gelu":
            self.dense_act_fn = "gelu_new"

        super().__init__(
            is_encoder_decoder=is_encoder_decoder,
            **kwargs,
        )


class MusicgenDecoderConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of an [`MusicgenDecoder`]. It is used to instantiate a
    MusicGen decoder according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the MusicGen
    [facebook/musicgen-small](https://huggingface.co/facebook/musicgen-small) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 2048):
            Vocabulary size of the MusicgenDecoder model. Defines the number of different tokens that can be
            represented by the `inputs_ids` passed when calling [`MusicgenDecoder`].
        d_model (`int`, *optional*, defaults to 1024):
            Dimensionality of the layers and the pooler layer.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of decoder layers.
        num_attention_heads (`int`, *optional*, defaults to 16):
            Number of attention heads for each attention layer in the Transformer block.
        ffn_dim (`int`, *optional*, defaults to 4096):
            Dimensionality of the "intermediate" (often named feed-forward) layer in the Transformer block.
        activation_function (`str` or `function`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the decoder and pooler. If string, `"gelu"`,
            `"relu"`, `"silu"` and `"gelu_new"` are supported.
        dropout (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        activation_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for activations inside the fully connected layer.
        classifier_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for classifier.
        max_position_embeddings (`int`, *optional*, defaults to 1024):
            The maximum sequence length that this model might ever be used with. Typically, set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        initializer_factor (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layerdrop (`float`, *optional*, defaults to 0.0):
            The LayerDrop probability for the decoder. See the [LayerDrop paper](see https://arxiv.org/abs/1909.11556)
            for more details.
        scale_embedding (`bool`, *optional*, defaults to `False`):
            Scale embeddings by diving by sqrt(d_model).
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether the model should return the last key/values attentions (not used by all models)
        num_codebooks (`int`, *optional*, defaults to 4):
            The number of parallel codebooks forwarded to the model.
    """
    model_type = "musicgen_decoder"
    keys_to_ignore_at_inference = ["past_key_values"]
    attribute_map = {"num_attention_heads": "num_heads", "num_hidden_layers": "num_layers"}

    def __init__(
        self,
        vocab_size=2048,
        max_position_embeddings=1024,
        num_hidden_layers=12,
        ffn_dim=4096,
        num_attention_heads=16,
        layerdrop=0.0,
        use_cache=True,
        activation_function="gelu",
        d_model=1024,
        dropout=0.1,
        attention_dropout=0.0,
        activation_dropout=0.0,
        initializer_factor=0.02,
        classifier_dropout=0.0,
        scale_embedding=False,
        num_codebooks=4,
        pad_token_id=2048,
        bos_token_id=2048,
        eos_token_id=None,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.d_model = d_model
        self.ffn_dim = ffn_dim
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.activation_dropout = activation_dropout
        self.activation_function = activation_function
        self.initializer_factor = initializer_factor
        self.layerdrop = layerdrop
        self.classifier_dropout = classifier_dropout
        self.use_cache = use_cache
        self.scale_embedding = scale_embedding  # scale factor will be sqrt(d_model) if True
        self.num_codebooks = num_codebooks
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs,
        )


class MusicgenConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`MusicgenModel`]. It is used to instantiate a
    MusicGen model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the MusicGen
    [facebook/musicgen-small](https://huggingface.co/facebook/musicgen-small) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        encoder_config (`dict`, *optional*):
            Dictionary of configuration options used to initialize [`MusicgenEncoderConfig`].
        decoder_config (`dict`, *optional*):
            Dictionary of configuration options used to initialize [`MusicgenDecoderConfig`].
        initializer_factor (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.

        kwargs (*optional*):
            Dictionary of keyword arguments.

    Example:

    ```python
    >>> from transformers import (
    ...     MusicgenConfig,
    ...     MusicgenEncoderConfig,
    ...     MusicgenDecoderConfig,
    ...     MusicgenForConditionalGeneration,
    ... )

    >>> # Initializing a Musicgen config with facebook/musicgen-small style configuration
    >>> configuration = MusicgenConfig()

    >>> # Initializing a MusicgenForConditionalGeneration (with random weights) from the facebook/musicgen-small style configuration
    >>> model = MusicgenForConditionalGeneration(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config

    >>> # We can also initialize a MusicgenConfig from a MusicgenEncoderConfig and MusicgenDecoderConfig

    >>> # Initializing encoder and decoder model configurations
    >>> encoder_config = MusicgenEncoderConfig()
    >>> decoder_config = MusicgenDecoderConfig()

    >>> config = MusicgenConfig.from_encoder_decoder_configs(encoder_config, decoder_config)
    ```"""

    model_type = "musicgen"
    is_composition = True

    def __init__(self, encoder_config=None, decoder_config=None, initializer_factor=0.02, use_cache=True, **kwargs):
        super().__init__(**kwargs)
        if encoder_config is None:
            encoder_config = {}
            logger.info("encoder_config is None. initializing the MusicgenEncoderConfig with default values.")

        if decoder_config is None:
            decoder_config = {}
            logger.info("decoder_config is None. Initializing the MusicgenDecoderConfig with default values.")

        self.encoder_config = MusicgenEncoderConfig(**encoder_config)
        self.decoder_config = MusicgenDecoderConfig(**decoder_config)

        self.initializer_factor = initializer_factor
        self.is_encoder_decoder = True
        self.use_cache = use_cache
        self.bos_token_id = self.decoder_config.bos_token_id
        self.pad_token_id = self.decoder_config.pad_token_id
        self.eos_token_id = self.decoder_config.eos_token_id

    @classmethod
    def from_encoder_decoder_configs(
        cls,
        encoder_config: MusicgenEncoderConfig,
        decoder_config: MusicgenDecoderConfig,
        **kwargs,
    ):
        r"""
        Instantiate a [`MusicgenConfig`] (or a derived class) from Musicgen encoder and decoder configurations.

        Returns:
            [`MusicgenConfig`]: An instance of a configuration object
        """

        return cls(
            encoder_config=encoder_config.to_dict(),
            decoder_config=decoder_config.to_dict(),
            **kwargs,
        )

    def to_dict(self):
        """
        Serializes this instance to a Python dictionary. Override the default [`~PretrainedConfig.to_dict`].

        Returns:
            `Dict[str, any]`: Dictionary of all the attributes that make up this configuration instance,
        """
        output = copy.deepcopy(self.__dict__)
        output["encoder_config"] = self.encoder_config.to_dict()
        output["decoder_config"] = self.decoder_config.to_dict()
        output["model_type"] = self.__class__.model_type
        return output
