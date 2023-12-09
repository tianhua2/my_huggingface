# Copyright 2020 The HuggingFace Team. All rights reserved.
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

from typing import TYPE_CHECKING

from ...utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_tf_available,
    is_tokenizers_available,
    is_torch_available,
    is_vision_available,
)


_import_structure = {
    "configuration_layoutlm": ["LAYOUTLM_PRETRAINED_CONFIG_ARCHIVE_MAP", "LayoutLMConfig", "LayoutLMOnnxConfig"],
    "processing_layoutlm": ["LayoutLMProcessor"],
    "tokenization_layoutlm": ["LayoutLMTokenizer"],
}

try:
    if not is_tokenizers_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["tokenization_layoutlm_fast"] = ["LayoutLMTokenizerFast"]

try:
    if not is_vision_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["image_processing_layoutlm"] = ["LayoutLMImageProcessor"]

try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_layoutlm"] = [
        "LAYOUTLM_PRETRAINED_MODEL_ARCHIVE_LIST",
        "LayoutLMForMaskedLM",
        "LayoutLMForSequenceClassification",
        "LayoutLMForTokenClassification",
        "LayoutLMForQuestionAnswering",
        "LayoutLMModel",
        "LayoutLMPreTrainedModel",
    ]

try:
    if not is_tf_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_tf_layoutlm"] = [
        "TF_LAYOUTLM_PRETRAINED_MODEL_ARCHIVE_LIST",
        "TFLayoutLMForMaskedLM",
        "TFLayoutLMForSequenceClassification",
        "TFLayoutLMForTokenClassification",
        "TFLayoutLMForQuestionAnswering",
        "TFLayoutLMMainLayer",
        "TFLayoutLMModel",
        "TFLayoutLMPreTrainedModel",
    ]


if TYPE_CHECKING:
    from .configuration_layoutlm import LAYOUTLM_PRETRAINED_CONFIG_ARCHIVE_MAP, LayoutLMConfig, LayoutLMOnnxConfig
    from .processing_layoutlm import LayoutLMProcessor
    from .tokenization_layoutlm import LayoutLMTokenizer

    try:
        if not is_tokenizers_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .tokenization_layoutlm_fast import LayoutLMTokenizerFast

    try:
        if not is_vision_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .image_processing_layoutlm import LayoutLMImageProcessor

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_layoutlm import (
            LAYOUTLM_PRETRAINED_MODEL_ARCHIVE_LIST,
            LayoutLMForMaskedLM,
            LayoutLMForQuestionAnswering,
            LayoutLMForSequenceClassification,
            LayoutLMForTokenClassification,
            LayoutLMModel,
            LayoutLMPreTrainedModel,
        )
    try:
        if not is_tf_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_tf_layoutlm import (
            TF_LAYOUTLM_PRETRAINED_MODEL_ARCHIVE_LIST,
            TFLayoutLMForMaskedLM,
            TFLayoutLMForQuestionAnswering,
            TFLayoutLMForSequenceClassification,
            TFLayoutLMForTokenClassification,
            TFLayoutLMMainLayer,
            TFLayoutLMModel,
            TFLayoutLMPreTrainedModel,
        )

else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
