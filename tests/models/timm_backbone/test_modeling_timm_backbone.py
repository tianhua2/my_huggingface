# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
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

import inspect
import unittest

from transformers import AutoBackbone, TimmBackbone, TimmBackboneConfig
from transformers.configuration_utils import PretrainedConfig
from transformers.testing_utils import require_timm, require_torch, torch_device
from transformers.utils.import_utils import is_torch_available

from ...test_backbone_common import BackboneTesterMixin
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, floats_tensor


if is_torch_available():
    import torch


@require_timm
@require_torch
class TimmBackboneModelTester:
    def __init__(
        self,
        parent,
        out_indices=None,
        out_features=None,
        stage_names=None,
        backbone="resnet50",
        batch_size=3,
        image_size=32,
        num_channels=3,
        is_training=True,
        use_pretrained_backbone=True,
    ):
        self.parent = parent
        self.out_indices = out_indices if out_indices is not None else [4]
        self.stage_names = stage_names
        self.out_features = out_features
        self.backbone = backbone
        self.batch_size = batch_size
        self.image_size = image_size
        self.num_channels = num_channels
        self.use_pretrained_backbone = use_pretrained_backbone
        self.is_training = is_training

    def prepare_config_and_inputs(self):
        pixel_values = floats_tensor([self.batch_size, self.num_channels, self.image_size, self.image_size])
        config = self.get_config()

        return config, pixel_values

    def get_config(self):
        return TimmBackboneConfig(
            image_size=self.image_size,
            num_channels=self.num_channels,
            out_features=self.out_features,
            out_indices=self.out_indices,
            stage_names=self.stage_names,
            use_pretrained_backbone=self.use_pretrained_backbone,
            backbone=self.backbone,
        )

    def create_and_check_model(self, config, pixel_values):
        model = TimmBackbone(config=config)
        model.to(torch_device)
        model.eval()
        with torch.no_grad():
            result = model(pixel_values)
        self.parent.assertEqual(
            result.feature_map[-1].shape,
            (self.batch_size, model.channels[-1], 14, 14),
        )

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, pixel_values = config_and_inputs
        inputs_dict = {"pixel_values": pixel_values}
        return config, inputs_dict


class TimmBackboneModelTest(ModelTesterMixin, BackboneTesterMixin, unittest.TestCase):
    all_model_classes = (TimmBackbone,)
    test_resize_embeddings = False
    test_head_masking = False
    test_pruning = False
    has_attentions = False

    def setUp(self):
        self.model_tester = TimmBackboneModelTester(self)
        self.config_tester = ConfigTester(self, config_class=PretrainedConfig, has_text_modality=False)

    def test_config(self):
        self.config_tester.create_and_test_config_to_json_string()
        self.config_tester.create_and_test_config_to_json_file()
        self.config_tester.create_and_test_config_from_and_save_pretrained()
        self.config_tester.create_and_test_config_with_num_labels()
        self.config_tester.check_config_can_be_init_without_params()
        self.config_tester.check_config_arguments_init()

    @require_timm
    def test_timm_transformer_backbone_equivalence(self):
        timm_checkpoint = "resnet50"
        transformers_checkpoint = "microsoft/resnet-50"

        timm_model = AutoBackbone.from_pretrained(timm_checkpoint, use_timm_backbone=True)
        transformers_model = AutoBackbone.from_pretrained(transformers_checkpoint)

        self.assertEqual(timm_model.out_indices, transformers_model.out_indices)
        self.assertEqual(len(timm_model.out_features), len(transformers_model.out_features))
        self.assertEqual(len(timm_model.stage_names), len(transformers_model.stage_names))
        self.assertEqual(timm_model.channels, transformers_model.channels)

        timm_model = AutoBackbone.from_pretrained(timm_checkpoint, use_timm_backbone=True, out_indices=(1, 2, 3))
        transformers_model = AutoBackbone.from_pretrained(transformers_checkpoint, out_indices=(1, 2, 3))

        self.assertEqual(timm_model.out_indices, transformers_model.out_indices)
        self.assertEqual(len(timm_model.out_features), len(transformers_model.out_features))
        self.assertEqual(timm_model.channels, transformers_model.channels)

    @unittest.skip("TimmBackbone doesn't support feed forward chunking")
    def test_feed_forward_chunking(self):
        pass

    @unittest.skip("TimmBackbone doesn't have num_hidden_layers attribute")
    def test_hidden_states_output(self):
        pass

    @unittest.skip("TimmBackbone initialization is managed on the timm side")
    def test_initialization(self):
        pass

    @unittest.skip("TimmBackbone models doesn't have inputs_embeds")
    def test_inputs_embeds(self):
        pass

    @unittest.skip("TimmBackbone models doesn't have inputs_embeds")
    def test_model_common_attributes(self):
        pass

    @unittest.skip("TimmBackbone model cannot be created without specifying a backbone checkpoint")
    def test_from_pretrained_no_checkpoint(self):
        pass

    @unittest.skip("Only checkpoints on timm can be loaded into TimmBackbone")
    def test_save_load(self):
        pass

    @unittest.skip("model weights aren't tied in TimmBackbone.")
    def test_tie_model_weights(self):
        pass

    @unittest.skip("model weights aren't tied in TimmBackbone.")
    def test_tied_model_weights_key_ignore(self):
        pass

    @unittest.skip("TimmBackbone doesn't have hidden size info in its configuration.")
    def test_channels(self):
        pass

    def test_forward_signature(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            signature = inspect.signature(model.forward)
            # signature.parameters is an OrderedDict => so arg_names order is deterministic
            arg_names = [*signature.parameters.keys()]

            expected_arg_names = ["pixel_values"]
            self.assertListEqual(arg_names[:1], expected_arg_names)

    def test_retain_grad_hidden_states_attentions(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.output_hidden_states = True
        config.output_attentions = self.has_attentions

        # no need to test all models as different heads yield the same functionality
        model_class = self.all_model_classes[0]
        model = model_class(config)
        model.to(torch_device)

        inputs = self._prepare_for_class(inputs_dict, model_class)
        outputs = model(**inputs)
        output = outputs[0][-1]

        # Encoder-/Decoder-only models
        hidden_states = outputs.hidden_states[0]
        hidden_states.retain_grad()

        if self.has_attentions:
            attentions = outputs.attentions[0]
            attentions.retain_grad()

        output.flatten()[0].backward(retain_graph=True)

        self.assertIsNotNone(hidden_states.grad)

        if self.has_attentions:
            self.assertIsNotNone(attentions.grad)
