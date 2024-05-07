<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# CogVLM

## Overview

The CogVLM model was proposed in [CogVLM: Visual Expert for Pretrained Language Models](https://arxiv.org/abs/2311.03079) by Weihan Wang, Qingsong Lv, Wenmeng Yu, Wenyi Hong, Ji Qi, Yan Wang, Junhui Ji, Zhuoyi Yang, Lei Zhao, Xixuan Song, Jiazheng Xu, Bin Xu, Juanzi Li, Yuxiao Dong, Ming Ding, Jie Tang. CogVLM adds separate QKV and MLP weights to a frozen large language model, enabling a strong multimodal foundation model that performs well on various multimodal benchmarks.

The abstract from the paper is the following:

*We introduce CogVLM, a powerful open-source visual language foundation model. Different from the popular shallow alignment method which maps image features into the input space of language model, CogVLM bridges the gap between the frozen pretrained language model and image encoder by a trainable visual expert module in the attention and FFN layers. As a result, CogVLM enables deep fusion of vision language features without sacrificing any performance on NLP tasks. CogVLM-17B achieves state-of-the-art performance on 10 classic cross-modal benchmarks, including NoCaps, Flicker30k captioning, RefCOCO, RefCOCO+, RefCOCOg, Visual7W, GQA, ScienceQA, VizWiz VQA and TDIUC, and ranks the 2nd on VQAv2, OKVQA, TextVQA, COCO captioning, etc., surpassing or matching PaLI-X 55B.*

Tips:

- One can use [`CogvlmProcessor`] to prepare images and text for the model.

This model was contributed by [nielsr](https://huggingface.co/nielsr).
The original code can be found [here](https://github.com/THUDM/CogVLM).


## CogvlmConfig

[[autodoc]] CogvlmConfig

## CogvlmVisionConfig

[[autodoc]] CogvlmVisionConfig

## CogvlmProcessor

[[autodoc]] CogvlmProcessor

## CogvlmModel

[[autodoc]] CogvlmModel
    - forward

## CogvlmForCausalLM

[[autodoc]] CogvlmForCausalLM
    - forward
    - generate