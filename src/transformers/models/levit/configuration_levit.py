# coding=utf-8
# Copyright 2022 Meta Inc. and The HuggingFace Inc. team. All rights reserved.
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
""" LeViT model configuration"""

from ...configuration_utils import PretrainedConfig
from ...utils import logging


logger = logging.get_logger(__name__)

LEVIT_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "anugunj/levit-128S": "https://huggingface.co/anugunj/levit-128S/resolve/main/config.json",
    # See all LeViT models at https://huggingface.co/models?filter=levit
}


class LevitConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`LevitModel`]. It is used to instantiate an LeViT
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the LeViT
    [facebook/levit-base-192](https://huggingface.co/facebook/levit-base-192) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        image_size (`int`, *optional*, defaults to 224):
            The size of the input image.
        num_channels (`int`, *optional*, defaults to 3):
            Number of channels in the input image.
        kernel_size (`int`, *optional*, defaults to 3):
            The kernel size for the initial convolution layers of patch embedding.
        stride (`int`, *optional*, defaults to 2):
            The stride size for the initial convolution layers of patch embedding.
        padding (`int`, *optional*, defaults to 1):
            The padding size for the initial convolution layers of patch embedding.
        patch_size (`int`, *optional*, defaults to 16):
            The patch size for embeddings.
        embed_dim (`List[int]`, *optional*, defaults to `[128, 256, 384]`):
            Dimension of each of the encoder blocks.
        num_heads (`List[int]`, *optional*, defaults to `[4, 8, 12]`):
            Number of attention heads for each attention layer in each block of the Transformer encoder.
        depth (`List[int]`, *optional*, defaults to `[4, 4, 4]`):
            The number of layers in each encoder block.
        key_dim (`List[int]`, *optional*, defaults to `[16, 16, 16]`):
            The size of key in each of the encoder blocks.
        drop_path_rate (`int`, *optional*, defaults to 0):
            The dropout probability for stochastic depth, used in the blocks of the Transformer encoder.
        mlp_ratios (`List[int]`, *optional*, defaults to `[2, 2, 2]`):
            Ratio of the size of the hidden layer compared to the size of the input layer of the Mix FFNs in the
            encoder blocks.
        attention_drop_rate (`List[int]`, *optional*, defaults to `[2, 2, 2]`):
            The dropout ratio for the attention probabilities.
        distillation (`bool`, *optional*, defaults to True):
            The value is set to True to use distillation else set to False.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.

    Example:

    ```python
    >>> from transformers import LevitModel, LevitConfig

    >>> # Initializing a LeViT levit-base-192 style configuration
    >>> configuration = LevitConfig()

    >>> # Initializing a model from the levit-base-192 style configuration
    >>> model = LevitModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""
    model_type = "levit"

    def __init__(
        self,
        image_size=224,
        num_channels=3,
        kernel_size=3,
        stride=2,
        padding=1,
        patch_size=16,
        embed_dim=[128, 256, 384],
        num_heads=[4, 8, 12],
        depth=[4, 4, 4],
        key_dim=[16, 16, 16],
        drop_path_rate=0,
        mlp_ratio=[2, 2, 2],
        attention_ratio=[2, 2, 2],
        distillation=True,
        initializer_range=0.02,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.image_size = image_size
        self.num_channels = num_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.depth = depth
        self.key_dim = key_dim
        self.drop_path = drop_path_rate
        self.patch_size = patch_size
        self.attention_ratio = attention_ratio
        self.mlp_ratio = mlp_ratio
        self.distillation = distillation
        self.initializer_range = initializer_range
        self.down_ops = [
            ["Subsample", key_dim[0], embed_dim[0] // key_dim[0], 4, 2, 2],
            ["Subsample", key_dim[0], embed_dim[1] // key_dim[0], 4, 2, 2],
        ]
