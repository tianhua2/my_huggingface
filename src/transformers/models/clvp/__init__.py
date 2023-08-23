# Copyright 2023 The HuggingFace Team. All rights reserved.
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
    is_torch_available,
)


_import_structure = {
    "configuration_clvp": [
        "CLVP_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "CLVPConfig",
        "CLVPSpeechConfig",
        "CLVPTextConfig",
        "CLVPAutoRegressiveConfig",
    ],
    "feature_extraction_clvp": ["CLVPFeatureExtractor"],
    "processing_clvp": ["CLVPProcessor"],
    "tokenization_clvp": ["CLVPTokenizer"],
}


try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_clvp"] = [
        "CLVP_PRETRAINED_MODEL_ARCHIVE_LIST",
        "CLVPModel",
        "CLVPPreTrainedModel",
        "CLVPTextModel",
        "CLVPTextModelWithProjection",
        "CLVPSpeechModel",
        "CLVPSpeechModelWithProjection",
    ]


if TYPE_CHECKING:
    from .configuration_clvp import (
        CLVP_PRETRAINED_CONFIG_ARCHIVE_MAP,
        CLVPConfig,
        CLVPSpeechConfig,
        CLVPTextConfig,
        CLVPAutoRegressiveConfig,
    )
    from .feature_extraction_clvp import CLVPFeatureExtractor
    from .processing_clvp import CLVPProcessor
    from .tokenization_clvp import CLVPTokenizer

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_clvp import (
            CLVP_PRETRAINED_MODEL_ARCHIVE_LIST,
            CLVPModel,
            CLVPPreTrainedModel,
            CLVPSpeechModel,
            CLVPSpeechModelWithProjection,
            CLVPTextModel,
            CLVPTextModelWithProjection,
        )

else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
