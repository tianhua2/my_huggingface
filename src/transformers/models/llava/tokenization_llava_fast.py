# coding=utf-8
# Copyright 2022 Haotian Liu, Chunyuan Li, Qingyang Wu, Yong Jae Lee and The HuggingFace Inc. team. All rights reserved.
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
"""Tokenization classes for LLaVA."""
from typing import List, Optional

from tokenizers import ByteLevelBPETokenizer

from ...tokenization_utils_fast import PreTrainedTokenizerFast
from ...utils import logging
from .tokenization_llava import LlaVaTokenizer


logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {"vocab_file": "vocab.txt", "tokenizer_file": "tokenizer.json"}

PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "liuhaotian/llava-v1.5-13b": "https://huggingface.co/liuhaotian/llava-v1.5-13b/resolve/main/vocab.txt",
    },
    "tokenizer_file": {
        "liuhaotian/llava-v1.5-13b": "https://huggingface.co/liuhaotian/llava-v1.5-13b/resolve/main/tokenizer.json",
    },
}

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "liuhaotian/llava-v1.5-13b": 1024,
}


class LlaVaTokenizerFast(PreTrainedTokenizerFast):
    """
    Construct a "fast" LLaVA tokenizer (backed by HuggingFace's *tokenizers* library).

    Args:
        vocab_file (`str`):
            Path to the vocabulary file.
        merges_file (`<fill_type>`): <fill_docstring>
        unk_token (`<fill_type>`, *optional*, defaults to `"<|endoftext|>"`): <fill_docstring>
        bos_token (`<fill_type>`, *optional*, defaults to `"<|endoftext|>"`): <fill_docstring>
        eos_token (`<fill_type>`, *optional*, defaults to `"<|endoftext|>"`): <fill_docstring>
        add_prefix_space (`<fill_type>`, *optional*, defaults to `False`): <fill_docstring>
        trim_offsets (`<fill_type>`, *optional*, defaults to `True`): <fill_docstring>
    """

    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    slow_tokenizer_class = LlaVaTokenizer

    def __init__(
        self,
        vocab_file,
        merges_file,
        unk_token="<|endoftext|>",
        bos_token="<|endoftext|>",
        eos_token="<|endoftext|>",
        add_prefix_space=False,
        trim_offsets=True,
        **kwargs,
    ):
        super().__init__(
            ByteLevelBPETokenizer(
                vocab_file=vocab_file,
                merges_file=merges_file,
                add_prefix_space=add_prefix_space,
                trim_offsets=trim_offsets,
            ),
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            **kwargs,
        )
        self.add_prefix_space = add_prefix_space

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        output = [self.bos_token_id] + token_ids_0 + [self.eos_token_id]
        if token_ids_1 is None:
            return output

        return output + [self.eos_token_id] + token_ids_1 + [self.eos_token_id]

    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Create a mask from the two sequences passed to be used in a sequence-pair classification task. LLaVA does not
        make use of token type ids, therefore a list of zeros is returned.

        Args:
            token_ids_0 (`List[int]`):
                List of IDs.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]`: List of zeros.
        """
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]

        if token_ids_1 is None:
            return len(cls + token_ids_0 + sep) * [0]
        return len(cls + token_ids_0 + sep + sep + token_ids_1 + sep) * [0]
