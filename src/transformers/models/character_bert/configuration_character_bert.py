# coding=utf-8
# Copyright Hicham EL BOUKKOURI, Olivier FERRET, Thomas LAVERGNE, Hiroshi NOJI,
# Pierre ZWEIGENBAUM, Junichi TSUJII and The HuggingFace Inc. team.
# All rights reserved.
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
""" CharacterBERT model configuration"""

from ...configuration_utils import PretrainedConfig
from ...utils import logging


logger = logging.get_logger(__name__)

CHARACTER_BERT_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "helboukkouri/character-bert": "https://huggingface.co/helboukkouri/character-bert/resolve/main/config.json",
    "helboukkouri/character-bert-medical": "https://huggingface.co/helboukkouri/character-bert-medical/resolve/main/config.json",
    # See all CharacterBERT models at https://huggingface.co/models?filter=character_bert
}


class CharacterBertConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`CharacterBertModel`]. It is
    used to instantiate an CharacterBERT model according to the specified arguments, defining the model architecture.
    Instantiating a configuration with the defaults will yield a similar configuration to that of the CharacterBERT
    [helboukkouri/character-bert](https://huggingface.co/helboukkouri/character-bert) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model
    outputs. Read the documentation from [`PretrainedConfig`] for more information.


    Args:
        character_embeddings_dim (`int`, *optional*, defaults to `16`):
            The size of the character embeddings.
        cnn_activation (`str`, *optional*, defaults to `"relu"`):
            The activation function to apply to the cnn representations.
        cnn_filters (:
            obj:*list(list(int))*, *optional*, defaults to `[[1, 32], [2, 32], [3, 64], [4, 128], [5, 256], [6, 512], [7, 1024]]`): The list of CNN filters to use in the CharacterCNN module.
        num_highway_layers (`int`, *optional*, defaults to `2`):
            The number of Highway layers to apply to the CNNs output.
        max_word_length (`int`, *optional*, defaults to `50`):
            The maximum token length in characters (actually, in bytes as any non-ascii characters will be converted to
            a sequence of utf-8 bytes).
        hidden_size (`int`, *optional*, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (`int`, *optional*, defaults to 3072):
            Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        hidden_act (`str` or `function`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string,
            `"gelu"`, `"relu"`, `"selu"` and `"gelu_new"` are supported.
        hidden_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout probabilitiy for all fully connected layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the attention probabilities.
        max_position_embeddings (`int`, *optional*, defaults to 512):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        type_vocab_size (`int`, *optional*, defaults to 2):
            The vocabulary size of the `token_type_ids` passed when calling
            [`CharacterBertModel`] or [`TFCharacterBertModel`].
        mlm_vocab_size (`int`, *optional*, defaults to 100000):
            Size of the output vocabulary for MLM.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.

    Example:

    ```python

    ```

            >>> from transformers import CharacterBertModel, CharacterBertConfig

            >>> # Initializing a CharacterBERT helboukkouri/character-bert style configuration
            >>> configuration = CharacterBertConfig()

            >>> # Initializing a model from the helboukkouri/character-bert style configuration
            >>> model = CharacterBertModel(configuration)

            >>> # Accessing the model configuration
            >>> configuration = model.config
    """
    model_type = "character_bert"

    def __init__(
        self,
        character_embeddings_dim=16,
        cnn_activation="relu",
        cnn_filters=None,
        num_highway_layers=2,
        max_word_length=50,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=2,
        mlm_vocab_size=100000,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        is_encoder_decoder=False,
        use_cache=True,
        **kwargs
    ):
        tie_word_embeddings = kwargs.pop("tie_word_embeddings", False)
        if tie_word_embeddings:
            raise ValueError(
                "Cannot tie word embeddings in CharacterBERT. Please set " "`config.tie_word_embeddings=False`."
            )
        super().__init__(
            type_vocab_size=type_vocab_size,
            layer_norm_eps=layer_norm_eps,
            use_cache=use_cache,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
        if cnn_filters is None:
            cnn_filters = [[1, 32], [2, 32], [3, 64], [4, 128], [5, 256], [6, 512], [7, 1024]]
        self.character_embeddings_dim = character_embeddings_dim
        self.cnn_activation = cnn_activation
        self.cnn_filters = cnn_filters
        self.num_highway_layers = num_highway_layers
        self.max_word_length = max_word_length
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.mlm_vocab_size = mlm_vocab_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
