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
"""Convert VITS checkpoint."""

import argparse
import os
import torch

from transformers import (
    VITSConfig,
    # VITSFeatureExtractor,
    VITSModel,
    # VITSProcessor,
    # VITSTokenizer,
    logging,
)
from transformers.tokenization_utils import AddedToken


logging.set_verbosity_info()
logger = logging.get_logger("transformers.models.speecht5")

# MAPPING_SPEECH_ENCODER_PRENET = {
#     "speech_encoder_prenet.layer_norm": "speecht5.encoder.prenet.feature_projection.layer_norm",
#     "speech_encoder_prenet.post_extract_proj": "speecht5.encoder.prenet.feature_projection.projection",
#     "speech_encoder_prenet.pos_conv.0": "speecht5.encoder.prenet.pos_conv_embed.conv",
#     "speech_encoder_prenet.mask_emb": "speecht5.encoder.prenet.masked_spec_embed",
# }
MAPPING_TEXT_ENCODER = {
    "enc_p.emb": "text_encoder.embed_tokens",
    # "text_encoder_prenet.encoder_prenet.1.alpha": "speecht5.encoder.prenet.encode_positions.alpha",
    "enc_p.encoder.attn_layers.*.conv_k": "text_encoder.encoder.layers.*.attention.k_proj",
    "enc_p.encoder.attn_layers.*.conv_v": "text_encoder.encoder.layers.*.attention.v_proj",
    "enc_p.encoder.attn_layers.*.conv_q": "text_encoder.encoder.layers.*.attention.q_proj",
    "enc_p.encoder.attn_layers.*.conv_o": "text_encoder.encoder.layers.*.attention.out_proj",
    "enc_p.encoder.attn_layers.*.emb_rel_k": "text_encoder.encoder.layers.*.attention.emb_rel_k",
    "enc_p.encoder.attn_layers.*.emb_rel_v": "text_encoder.encoder.layers.*.attention.emb_rel_v",
    "enc_p.encoder.norm_layers_1.*.gamma": "text_encoder.encoder.layers.*.layer_norm.weight",
    "enc_p.encoder.norm_layers_1.*.beta": "text_encoder.encoder.layers.*.layer_norm.bias",
    "enc_p.encoder.ffn_layers.*.conv_1": "text_encoder.encoder.layers.*.feed_forward.conv_1",
    "enc_p.encoder.ffn_layers.*.conv_2": "text_encoder.encoder.layers.*.feed_forward.conv_2",
    "enc_p.encoder.norm_layers_2.*.gamma": "text_encoder.encoder.layers.*.final_layer_norm.weight",
    "enc_p.encoder.norm_layers_2.*.beta": "text_encoder.encoder.layers.*.final_layer_norm.bias",
    "enc_p.proj": "text_encoder.project",

    # "encoder.layers.*.final_layer_norm": "speecht5.encoder.wrapped_encoder.layers.*.final_layer_norm",
    # "encoder.layer_norm": "speecht5.encoder.wrapped_encoder.layer_norm",
    # "encoder.pos_emb.pe_k": "speecht5.encoder.wrapped_encoder.embed_positions.pe_k",
}
# MAPPING_DECODER = {
#     "decoder.layers.*.self_attn.k_proj": "speecht5.decoder.wrapped_decoder.layers.*.self_attn.k_proj",
#     "decoder.layers.*.self_attn.v_proj": "speecht5.decoder.wrapped_decoder.layers.*.self_attn.v_proj",
#     "decoder.layers.*.self_attn.q_proj": "speecht5.decoder.wrapped_decoder.layers.*.self_attn.q_proj",
#     "decoder.layers.*.self_attn.out_proj": "speecht5.decoder.wrapped_decoder.layers.*.self_attn.out_proj",
#     "decoder.layers.*.self_attn_layer_norm": "speecht5.decoder.wrapped_decoder.layers.*.self_attn_layer_norm",
#     "decoder.layers.*.encoder_attn.k_proj": "speecht5.decoder.wrapped_decoder.layers.*.encoder_attn.k_proj",
#     "decoder.layers.*.encoder_attn.v_proj": "speecht5.decoder.wrapped_decoder.layers.*.encoder_attn.v_proj",
#     "decoder.layers.*.encoder_attn.q_proj": "speecht5.decoder.wrapped_decoder.layers.*.encoder_attn.q_proj",
#     "decoder.layers.*.encoder_attn.out_proj": "speecht5.decoder.wrapped_decoder.layers.*.encoder_attn.out_proj",
#     "decoder.layers.*.encoder_attn_layer_norm": "speecht5.decoder.wrapped_decoder.layers.*.encoder_attn_layer_norm",
#     "decoder.layers.*.fc1": "speecht5.decoder.wrapped_decoder.layers.*.feed_forward.intermediate_dense",
#     "decoder.layers.*.fc2": "speecht5.decoder.wrapped_decoder.layers.*.feed_forward.output_dense",
#     "decoder.layers.*.final_layer_norm": "speecht5.decoder.wrapped_decoder.layers.*.final_layer_norm",
# }
MAPPING = {
    **MAPPING_TEXT_ENCODER,
}
TOP_LEVEL_KEYS = []
IGNORE_KEYS = [
    # "encoder.version",
    # "encoder.layers.*.norm_k.weight",
    # "encoder.layers.*.norm_k.bias",
    # "decoder.version",
    # "decoder.layers.*.norm_k.weight",
    # "decoder.layers.*.norm_k.bias",
    # "decoder.pos_emb.pe_k",
    # "speech_encoder_prenet.embed_positions._float_tensor",
    # "text_decoder_prenet.embed_positions._float_tensor",
    # "encoder.proj",
]


def set_recursively(hf_pointer, key, value, full_name, weight_type):
    for attribute in key.split("."):
        hf_pointer = getattr(hf_pointer, attribute)

    if weight_type is not None:
        hf_shape = getattr(hf_pointer, weight_type).shape
    else:
        hf_shape = hf_pointer.shape

    # strip off the kernel dimension at the end (original weights are Conv1d)
    if key.endswith(".k_proj") or key.endswith(".v_proj") or key.endswith(".q_proj") or key.endswith(".out_proj"):
        value = value.squeeze(-1)

    if hf_shape != value.shape:
        raise ValueError(
            f"Shape of hf {key + '.' + weight_type if weight_type is not None else ''} is {hf_shape}, but should be"
            f" {value.shape} for {full_name}"
        )

    if weight_type == "weight":
        hf_pointer.weight.data = value
    elif weight_type == "weight_g":
        hf_pointer.weight_g.data = value
    elif weight_type == "weight_v":
        hf_pointer.weight_v.data = value
    elif weight_type == "bias":
        hf_pointer.bias.data = value
    elif weight_type == "running_mean":
        hf_pointer.running_mean.data = value
    elif weight_type == "running_var":
        hf_pointer.running_var.data = value
    elif weight_type == "num_batches_tracked":
        hf_pointer.num_batches_tracked.data = value
    else:
        hf_pointer.data = value

    logger.info(f"{key + ('.' + weight_type if weight_type is not None else '')} was initialized from {full_name}.")


def should_ignore(name, ignore_keys):
    for key in ignore_keys:
        if key.endswith(".*"):
            if name.startswith(key[:-1]):
                return True
        elif ".*." in key:
            prefix, suffix = key.split(".*.")
            if prefix in name and suffix in name:
                return True
        elif key in name:
            return True
    return False


def recursively_load_weights(fairseq_dict, hf_model):
    unused_weights = []

    # if task == "s2t":
    #     feature_encoder = hf_model.speecht5.encoder.prenet.feature_encoder
    #     MAPPING = MAPPING_S2T
    #     IGNORE_KEYS = IGNORE_KEYS_S2T
    # elif task == "t2s":
    #     feature_encoder = None
    #     MAPPING = MAPPING_T2S
    #     IGNORE_KEYS = IGNORE_KEYS_T2S
    # elif task == "s2s":
    #     feature_encoder = hf_model.speecht5.encoder.prenet.feature_encoder
    #     MAPPING = MAPPING_S2S
    #     IGNORE_KEYS = IGNORE_KEYS_S2S
    # else:
    #     raise ValueError(f"Unsupported task: {task}")

    for name, value in fairseq_dict.items():
        if should_ignore(name, IGNORE_KEYS):
            logger.info(f"{name} was ignored")
            continue

        is_used = False
        if False and "conv_layers" in name:
            load_conv_layer(
                name,
                value,
                feature_encoder,
                unused_weights,
                hf_model.config.feat_extract_norm == "group",
            )
            is_used = True
        else:
            for key, mapped_key in MAPPING.items():
                # mapped_key = "speecht5." + mapped_key if mapped_key not in TOP_LEVEL_KEYS else mapped_key

                # print(key, mapped_key)

                if "*" in key:
                    # if key.endswith(".*"):
                    #     if key[:-2] in name:
                    #         key = key#[:-1]
                    # else:
                    prefix, suffix = key.split(".*.")
                    if prefix in name and suffix in name:
                        key = suffix

                # if key in name or key.split("w2v_model.")[-1] == name.split(".")[0]:
                if key in name:
                    # print(key, name, mapped_key)
                    is_used = True
                    if "*" in mapped_key:
                        # print(key, name, mapped_key)
                        # print(name.split(key))
                        # print(name.split(key)[0])
                        # print(name.split(key)[0].split("."))
                        layer_index = name.split(key)[0].split(".")[-2]
                        mapped_key = mapped_key.replace("*", layer_index)
                    if "weight_g" in name:
                        weight_type = "weight_g"
                    elif "weight_v" in name:
                        weight_type = "weight_v"
                    elif "bias" in name:
                        weight_type = "bias"
                    elif "weight" in name:
                        weight_type = "weight"
                    elif "running_mean" in name:
                        weight_type = "running_mean"
                    elif "running_var" in name:
                        weight_type = "running_var"
                    elif "num_batches_tracked" in name:
                        weight_type = "num_batches_tracked"
                    else:
                        weight_type = None
                    set_recursively(hf_model, mapped_key, value, name, weight_type)
                continue
        if not is_used:
            unused_weights.append(name)

    #MIH logger.warning(f"Unused weights: {unused_weights}")


def load_conv_layer(full_name, value, feature_extractor, unused_weights, use_group_norm):
    name = full_name.split("conv_layers.")[-1]
    items = name.split(".")
    layer_id = int(items[0])
    type_id = int(items[1])

    if type_id == 0:
        if "bias" in name:
            if value.shape != feature_extractor.conv_layers[layer_id].conv.bias.data.shape:
                raise ValueError(
                    f"{full_name} has size {value.shape}, but"
                    f" {feature_extractor.conv_layers[layer_id].conv.bias.data.shape} was found."
                )
            feature_extractor.conv_layers[layer_id].conv.bias.data = value
            logger.info(f"Feat extract conv layer {layer_id} was initialized from {full_name}.")
        elif "weight" in name:
            if value.shape != feature_extractor.conv_layers[layer_id].conv.weight.data.shape:
                raise ValueError(
                    f"{full_name} has size {value.shape}, but"
                    f" {feature_extractor.conv_layers[layer_id].conv.weight.data.shape} was found."
                )
            feature_extractor.conv_layers[layer_id].conv.weight.data = value
            logger.info(f"Feat extract conv layer {layer_id} was initialized from {full_name}.")
    elif (type_id == 2 and not use_group_norm) or (type_id == 2 and layer_id == 0 and use_group_norm):
        if "bias" in name:
            if value.shape != feature_extractor.conv_layers[layer_id].layer_norm.bias.data.shape:
                raise ValueError(
                    f"{full_name} has size {value.shape}, but"
                    f" {feature_extractor.conv_layers[layer_id].layer_norm.bias.data.shape} was found."
                )
            feature_extractor.conv_layers[layer_id].layer_norm.bias.data = value
            logger.info(f"Feat extract layer norm weight of layer {layer_id} was initialized from {full_name}.")
        elif "weight" in name:
            if value.shape != feature_extractor.conv_layers[layer_id].layer_norm.weight.data.shape:
                raise ValueError(
                    f"{full_name} has size {value.shape}, but"
                    f" {feature_extractor.conv_layers[layer_id].layer_norm.weight.data.shape} was found."
                )
            feature_extractor.conv_layers[layer_id].layer_norm.weight.data = value
            logger.info(f"Feat extract layer norm weight of layer {layer_id} was initialized from {full_name}.")
    else:
        unused_weights.append(full_name)


@torch.no_grad()
def convert_checkpoint(
    checkpoint_path,
    pytorch_dump_folder_path,
    config_path=None,
    vocab_path=None,
    repo_id=None,
):
    """
    Copy/paste/tweak model's weights to transformers design.
    """
    if config_path is not None:
        config = VITSConfig.from_pretrained(config_path)
    else:
        config = VITSConfig()

    # if task == "s2t":
    #     config.max_length = config.max_text_positions
    #     model = SpeechT5ForSpeechToText(config)
    # elif task == "t2s":
    #     config.max_speech_positions = 1876
    #     config.max_text_positions = 600
    #     config.max_length = config.max_speech_positions
    #     model = SpeechT5ForTextToSpeech(config)
    # elif task == "s2s":
    #     config.max_speech_positions = 1876
    #     config.max_length = config.max_speech_positions
    #     model = SpeechT5ForSpeechToSpeech(config)
    # else:
    #     raise ValueError(f"Unknown task name: {task}")

    model = VITSModel(config)

    # if vocab_path:
    #     tokenizer = SpeechT5Tokenizer(vocab_path, model_max_length=config.max_text_positions)

    #     # Mask token behaves like a normal word, i.e. include the space before it
    #     mask_token = AddedToken("<mask>", lstrip=True, rstrip=False)
    #     tokenizer.mask_token = mask_token
    #     tokenizer.add_special_tokens({"mask_token": mask_token})
    #     tokenizer.add_tokens(["<ctc_blank>"])

    # feature_extractor = VITSFeatureExtractor()
    # processor = VITSProcessor(tokenizer=tokenizer, feature_extractor=feature_extractor)
    # processor.save_pretrained(pytorch_dump_folder_path)

    orig_checkpoint = torch.load(os.path.join(checkpoint_path, "G_100000.pth"), map_location=torch.device("cpu"))
    recursively_load_weights(orig_checkpoint["model"], model)

    model.save_pretrained(pytorch_dump_folder_path)

    # if repo_id:
    #     print("Pushing to the hub...")
    #     processor.push_to_hub(repo_id)
    #     model.push_to_hub(repo_id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", required=True, default=None, type=str, help="Path to original checkpoint")
    parser.add_argument("--vocab_path", default=None, type=str, help="Path to TODO")
    parser.add_argument("--config_path", default=None, type=str, help="Path to hf config.json of model to convert")
    parser.add_argument(
        "--pytorch_dump_folder_path", required=True, default=None, type=str, help="Path to the output PyTorch model."
    )
    parser.add_argument(
        "--push_to_hub", default=None, type=str, help="Where to upload the converted model on the 🤗 hub."
    )

    args = parser.parse_args()
    convert_checkpoint(
        args.checkpoint_path,
        args.pytorch_dump_folder_path,
        args.config_path,
        args.vocab_path,
        args.push_to_hub,
    )
