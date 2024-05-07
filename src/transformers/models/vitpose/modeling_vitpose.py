# coding=utf-8
# Copyright 2024 University of Sydney and The HuggingFace Inc. team. All rights reserved.
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
""" PyTorch ViTPose model."""

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn

from ...modeling_utils import PreTrainedModel
from ...utils import (
    ModelOutput,
    logging,
)
from ...utils.backbone_utils import load_backbone
from .configuration_vitpose import ViTPoseConfig


logger = logging.get_logger(__name__)

# General docstring
_CONFIG_FOR_DOC = "ViTPoseConfig"


@dataclass
class PoseEstimatorOutput(ModelOutput):
    """
    Class for outputs of pose estimation models.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Classification (or regression if config.num_labels==1) loss.
        heatmaps (`torch.FloatTensor` of shape `(batch_size, num_keypoints, height, width)`):
            Heatmaps.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each stage) of shape `(batch_size, sequence_length, hidden_size)`. Hidden-states
            (also called feature maps) of the model at the output of each stage.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, patch_size,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    loss: Optional[torch.FloatTensor] = None
    heatmaps: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None


class ViTPosePreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = ViTPoseConfig
    base_model_prefix = "vit"
    main_input_name = "pixel_values"
    supports_gradient_checkpointing = True

    def _init_weights(self, module: Union[nn.Linear, nn.Conv2d, nn.LayerNorm]) -> None:
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            # Upcast the input in `fp32` and cast it back to desired `dtype` to avoid
            # `trunc_normal_cpu` not implemented in `half` issues
            module.weight.data = nn.init.trunc_normal_(
                module.weight.data.to(torch.float32), mean=0.0, std=self.config.initializer_range
            ).to(module.weight.dtype)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


VITPOSE_START_DOCSTRING = r"""
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it
    as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`ViTPoseConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

VITPOSE_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Pixel values can be obtained using [`ViTPoseImageProcessor`]. See
            [`ViTPoseImageProcessor.__call__`] for details.

        head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


def flip_back(output_flipped, flip_pairs, target_type="GaussianHeatmap"):
    """Flip the flipped heatmaps back to the original form.

    Args:
        output_flipped (`np.ndarray` of shape `(batch_size, num_keypoints, height, width)`):
            The output heatmaps obtained from the flipped images.
        flip_pairs (list[tuple()):
            Pairs of keypoints which are mirrored (for example, left ear -- right ear).
        target_type (str):
            GaussianHeatmap or CombinedTarget

    Returns:
        np.ndarray: heatmaps that flipped back to the original image
    """
    if output_flipped.ndim != 4:
        raise ValueError("output_flipped should be [batch_size, num_keypoints, height, width]")
    shape_ori = output_flipped.shape
    channels = 1
    if target_type.lower() == "CombinedTarget".lower():
        channels = 3
        output_flipped[:, 1::3, ...] = -output_flipped[:, 1::3, ...]
    output_flipped = output_flipped.reshape(shape_ori[0], -1, channels, shape_ori[2], shape_ori[3])
    output_flipped_back = output_flipped.copy()

    # Swap left-right parts
    for left, right in flip_pairs.tolist():
        output_flipped_back[:, left, ...] = output_flipped[:, right, ...]
        output_flipped_back[:, right, ...] = output_flipped[:, left, ...]
    output_flipped_back = output_flipped_back.reshape(shape_ori)
    # Flip horizontally
    output_flipped_back = output_flipped_back[..., ::-1]
    return output_flipped_back


class ViTPoseSimpleDecoder(nn.Module):
    """
    Simple decoding head consisting of a ReLU activation, 4x upsampling and a 3x3 convolution, turning the
    feature maps into heatmaps.
    """

    def __init__(self, config) -> None:
        super().__init__()

        self.scale_factor = config.scale_factor
        self.conv = nn.Conv2d(config.backbone_hidden_size, config.num_labels, kernel_size=3, stride=1, padding=1)

    def forward(self, hidden_state, flip_pairs) -> torch.Tensor:
        # Transform input: ReLu + upsample
        hidden_state = nn.functional.relu(hidden_state)
        hidden_state = nn.functional.interpolate(
            hidden_state, scale_factor=self.scale_factor, mode="bilinear", align_corners=False
        )

        heatmaps = self.conv(hidden_state)

        if flip_pairs is not None:
            heatmaps = flip_back(heatmaps.detach().cpu().numpy(), flip_pairs)

        return heatmaps


class ViTPoseClassicDecoder(nn.Module):
    """
    Classic decoding head consisting of a 2 deconvolutional blocks, followed by a 1x1 convolution layer,
    turning the feature maps into heatmaps.
    """

    def __init__(self, config):
        super().__init__()

        self.deconv1 = nn.ConvTranspose2d(
            config.backbone_hidden_size, 256, kernel_size=4, stride=2, padding=1, bias=False
        )
        self.batchnorm1 = nn.BatchNorm2d(256)
        self.relu1 = nn.ReLU()

        self.deconv2 = nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1, bias=False)
        self.batchnorm2 = nn.BatchNorm2d(256)
        self.relu2 = nn.ReLU()

        self.conv = nn.Conv2d(256, config.num_labels, kernel_size=1, stride=1, padding=0)

    def forward(self, hidden_state, flip_pairs):
        hidden_state = self.deconv1(hidden_state)
        hidden_state = self.batchnorm1(hidden_state)
        hidden_state = self.relu1(hidden_state)

        hidden_state = self.deconv2(hidden_state)
        hidden_state = self.batchnorm2(hidden_state)
        hidden_state = self.relu2(hidden_state)

        heatmaps = self.conv(hidden_state)

        if flip_pairs is not None:
            heatmaps = flip_back(heatmaps.detach().cpu().numpy(), flip_pairs)

        return heatmaps


class ViTPoseForPoseEstimation(ViTPosePreTrainedModel):
    def __init__(self, config: ViTPoseConfig) -> None:
        super().__init__(config)

        self.backbone = load_backbone(config)
        # add backbone attributes
        config.backbone_hidden_size = self.backbone.config.hidden_size
        config.image_size = self.backbone.config.image_size
        config.patch_size = self.backbone.config.patch_size
        self.head = ViTPoseSimpleDecoder(config) if config.use_simple_decoder else ViTPoseClassicDecoder(config)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.backbone.get_input_embeddings()

    def forward(
        self,
        pixel_values: torch.Tensor,
        dataset_index: Optional[torch.Tensor] = None,
        flip_pairs: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[tuple, PoseEstimatorOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions

        loss = None
        if labels is not None:
            raise NotImplementedError("Training is not yet supported")

        outputs = self.backbone.forward_with_filtered_kwargs(
            pixel_values,
            dataset_index=dataset_index,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            return_dict=return_dict,
        )

        # Turn output hidden states in tensor of shape (batch_size, num_channels, height, width)
        sequence_output = outputs.feature_maps[-1] if return_dict else outputs[0][-1]
        batch_size = sequence_output.shape[0]
        patch_height = self.config.image_size[0] // self.config.patch_size[0]
        patch_width = self.config.image_size[1] // self.config.patch_size[1]
        sequence_output = (
            sequence_output.permute(0, 2, 1).reshape(batch_size, -1, patch_height, patch_width).contiguous()
        )

        heatmaps = self.head(sequence_output, flip_pairs=flip_pairs)

        if not return_dict:
            if output_hidden_states:
                output = (heatmaps,) + outputs[1:]
            else:
                output = (heatmaps,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return PoseEstimatorOutput(
            loss=loss,
            heatmaps=heatmaps,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
