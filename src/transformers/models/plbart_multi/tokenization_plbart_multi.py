# coding=utf-8
# Copyright 2020 The Facebook AI Research Team Authors and The HuggingFace Inc. team.
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

from contextlib import contextmanager
from typing import List, Optional

from ...tokenization_utils import BatchEncoding
from ...utils import logging
from ..xlm_roberta.tokenization_xlm_roberta import XLMRobertaTokenizer


logger = logging.get_logger(__name__)


VOCAB_FILES_NAMES = {"vocab_file": "sentencepiece.bpe.model", "tokenizer_file": "tokenizer.json"}

PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "uclanlp/plbart-multi_task-all": "https://huggingface.co/uclanlp/plbart-multi_task-all/resolve/main/sentencepiece.bpe.model",
        "uclanlp/plbart-multi_task-compiled": "https://huggingface.co/uclanlp/plbart-multi_task-compiled/resolve/main/sentencepiece.bpe.model",
        "uclanlp/plbart-multi_task-dynamic": "https://huggingface.co/uclanlp/plbart-multi_task-dynamic/resolve/main/sentencepiece.bpe.model",
        "uclanlp/plbart-multi_task-go": "https://huggingface.co/uclanlp/plbart-multi_task-go/resolve/main/sentencepiece.bpe.model",
        "uclanlp/plbart-multi_task-interpreted": "https://huggingface.co/uclanlp/plbart-multi_task-interpreted/resolve/main/sentencepiece.bpe.model",
        "uclanlp/plbart-multi_task-java": "https://huggingface.co/uclanlp/plbart-multi_task-java/resolve/main/sentencepiece.bpe.model",
        "uclanlp/plbart-multi_task-js": "https://huggingface.co/uclanlp/plbart-multi_task-js/resolve/main/sentencepiece.bpe.model",
        "uclanlp/plbart-multi_task-php": "https://huggingface.co/uclanlp/plbart-multi_task-php/resolve/main/sentencepiece.bpe.model",
        "uclanlp/plbart-multi_task-python": "https://huggingface.co/uclanlp/plbart-multi_task-python/resolve/main/sentencepiece.bpe.model",
        "uclanlp/plbart-multi_task-ruby": "https://huggingface.co/uclanlp/plbart-multi_task-ruby/resolve/main/sentencepiece.bpe.model",
        "uclanlp/plbart-multi_task-static": "https://huggingface.co/uclanlp/plbart-multi_task-static/resolve/main/sentencepiece.bpe.model",
        "uclanlp/plbart-multi_task-strong": "https://huggingface.co/uclanlp/plbart-multi_task-strong/resolve/main/sentencepiece.bpe.model",
        "uclanlp/plbart-multi_task-weak": "https://huggingface.co/uclanlp/plbart-multi_task-weak/resolve/main/sentencepiece.bpe.model",
        "uclanlp/plbart-single_task-all-generation": "https://huggingface.co/uclanlp/plbart-single_task-all-generation/resolve/main/sentencepiece.bpe.model",
        "uclanlp/plbart-single_task-all-summarization": "https://huggingface.co/uclanlp/plbart-single_task-all-summarization/resolve/main/sentencepiece.bpe.model",
        "uclanlp/plbart-single_task-compiled-generation": "https://huggingface.co/uclanlp/plbart-single_task-compiled-generation/resolve/main/sentencepiece.bpe.model",
        "uclanlp/plbart-single_task-compiled-summarization": "https://huggingface.co/uclanlp/plbart-single_task-compiled-summarization/resolve/main/sentencepiece.bpe.model",
        "uclanlp/plbart-single_task-dynamic-generation": "https://huggingface.co/uclanlp/plbart-single_task-dynamic-generation/resolve/main/sentencepiece.bpe.model",
        "uclanlp/plbart-single_task-dynamic-summarization": "https://huggingface.co/uclanlp/plbart-single_task-dynamic-summarization/resolve/main/sentencepiece.bpe.model",
        "uclanlp/plbart-single_task-en_go": "https://huggingface.co/uclanlp/plbart-single_task-en_go/resolve/main/sentencepiece.bpe.model",
        "uclanlp/plbart-single_task-en_java": "https://huggingface.co/uclanlp/plbart-single_task-en_java/resolve/main/sentencepiece.bpe.model",
        "uclanlp/plbart-single_task-en_js": "https://huggingface.co/uclanlp/plbart-single_task-en_js/resolve/main/sentencepiece.bpe.model",
        "uclanlp/plbart-single_task-en_php": "https://huggingface.co/uclanlp/plbart-single_task-en_php/resolve/main/sentencepiece.bpe.model",
        "uclanlp/plbart-single_task-en_python": "https://huggingface.co/uclanlp/plbart-single_task-en_python/resolve/main/sentencepiece.bpe.model",
        "uclanlp/plbart-single_task-en_ruby": "https://huggingface.co/uclanlp/plbart-single_task-en_ruby/resolve/main/sentencepiece.bpe.model",
        "uclanlp/plbart-single_task-go_en": "https://huggingface.co/uclanlp/plbart-single_task-go_en/resolve/main/sentencepiece.bpe.model",
        "uclanlp/plbart-single_task-interpreted-generation": "https://huggingface.co/uclanlp/plbart-single_task-interpreted-generation/resolve/main/sentencepiece.bpe.model",
        "uclanlp/plbart-single_task-interpreted-summarization": "https://huggingface.co/uclanlp/plbart-single_task-interpreted-summarization/resolve/main/sentencepiece.bpe.model",
        "uclanlp/plbart-single_task-java_en": "https://huggingface.co/uclanlp/plbart-single_task-java_en/resolve/main/sentencepiece.bpe.model",
        "uclanlp/plbart-single_task-js_en": "https://huggingface.co/uclanlp/plbart-single_task-js_en/resolve/main/sentencepiece.bpe.model",
        "uclanlp/plbart-single_task-php_en": "https://huggingface.co/uclanlp/plbart-single_task-php_en/resolve/main/sentencepiece.bpe.model",
        "uclanlp/plbart-single_task-python_en": "https://huggingface.co/uclanlp/plbart-single_task-python_en/resolve/main/sentencepiece.bpe.model",
        "uclanlp/plbart-single_task-ruby_en": "https://huggingface.co/uclanlp/plbart-single_task-ruby_en/resolve/main/sentencepiece.bpe.model",
        "uclanlp/plbart-single_task-static-generation": "https://huggingface.co/uclanlp/plbart-single_task-static-generation/resolve/main/sentencepiece.bpe.model",
        "uclanlp/plbart-single_task-static-summarization": "https://huggingface.co/uclanlp/plbart-single_task-static-summarization/resolve/main/sentencepiece.bpe.model",
        "uclanlp/plbart-single_task-strong-generation": "https://huggingface.co/uclanlp/plbart-single_task-strong-generation/resolve/main/sentencepiece.bpe.model",
        "uclanlp/plbart-single_task-strong-summarization": "https://huggingface.co/uclanlp/plbart-single_task-strong-summarization/resolve/main/sentencepiece.bpe.model",
        "uclanlp/plbart-single_task-weak-generation": "https://huggingface.co/uclanlp/plbart-single_task-weak-generation/resolve/main/sentencepiece.bpe.model",
        "uclanlp/plbart-single_task-weak-summarization": "https://huggingface.co/uclanlp/plbart-single_task-weak-summarization/resolve/main/sentencepiece.bpe.model",
    }
}

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "uclanlp/plbart-multi_task-all": 1024,
    "uclanlp/plbart-multi_task-compiled": 1024,
    "uclanlp/plbart-multi_task-dynamic": 1024,
    "uclanlp/plbart-multi_task-go": 1024,
    "uclanlp/plbart-multi_task-interpreted": 1024,
    "uclanlp/plbart-multi_task-java": 1024,
    "uclanlp/plbart-multi_task-js": 1024,
    "uclanlp/plbart-multi_task-php": 1024,
    "uclanlp/plbart-multi_task-python": 1024,
    "uclanlp/plbart-multi_task-ruby": 1024,
    "uclanlp/plbart-multi_task-static": 1024,
    "uclanlp/plbart-multi_task-strong": 1024,
    "uclanlp/plbart-multi_task-weak": 1024,
    "uclanlp/plbart-single_task-all-generation": 1024,
    "uclanlp/plbart-single_task-all-summarization": 1024,
    "uclanlp/plbart-single_task-compiled-generation": 1024,
    "uclanlp/plbart-single_task-compiled-summarization": 1024,
    "uclanlp/plbart-single_task-dynamic-generation": 1024,
    "uclanlp/plbart-single_task-dynamic-summarization": 1024,
    "uclanlp/plbart-single_task-en_go": 1024,
    "uclanlp/plbart-single_task-en_java": 1024,
    "uclanlp/plbart-single_task-en_js": 1024,
    "uclanlp/plbart-single_task-en_php": 1024,
    "uclanlp/plbart-single_task-en_python": 1024,
    "uclanlp/plbart-single_task-en_ruby": 1024,
    "uclanlp/plbart-single_task-go_en": 1024,
    "uclanlp/plbart-single_task-interpreted-generation": 1024,
    "uclanlp/plbart-single_task-interpreted-summarization": 1024,
    "uclanlp/plbart-single_task-java_en": 1024,
    "uclanlp/plbart-single_task-js_en": 1024,
    "uclanlp/plbart-single_task-php_en": 1024,
    "uclanlp/plbart-single_task-python_en": 1024,
    "uclanlp/plbart-single_task-ruby_en": 1024,
    "uclanlp/plbart-single_task-static-generation": 1024,
    "uclanlp/plbart-single_task-static-summarization": 1024,
    "uclanlp/plbart-single_task-strong-generation": 1024,
    "uclanlp/plbart-single_task-strong-summarization": 1024,
    "uclanlp/plbart-single_task-weak-generation": 1024,
    "uclanlp/plbart-single_task-weak-summarization": 1024,
}

FAIRSEQ_LANGUAGE_CODES = [
    "java",
    "python",
    "en_XX",
    "javascript",
    "php",
    "ruby",
    "go",
]


class PLBartMultiTokenizer(XLMRobertaTokenizer):
    """
    Construct an PLBART tokenizer.

    :class:`~transformers.PLBartMultiTokenizer` is a subclass of :class:`~transformers.XLMRobertaTokenizer`. Refer to
    superclass :class:`~transformers.XLMRobertaTokenizer` for usage examples and documentation concerning the
    initialization parameters and other methods.

    The tokenization method is ``<tokens> <eos> <language code>`` for source language documents, and ``<language code>
    <tokens> <eos>``` for target language documents.

    Examples::

        >>> from transformers import PLBartMultiTokenizer >>> tokenizer =
        PLBartMultiTokenizer.from_pretrained('facebook/mbart-large-en-ro', src_lang="en_XX", tgt_lang="ro_RO") >>>
        example_english_phrase = " UN Chief Says There Is No Military Solution in Syria" >>>
        expected_translation_romanian = "Şeful ONU declară că nu există o soluţie militară în Siria" >>> inputs =
        tokenizer(example_english_phrase, return_tensors="pt) >>> with tokenizer.as_target_tokenizer(): ... labels =
        tokenizer(expected_translation_romanian, return_tensors="pt") >>> inputs["labels"] = labels["input_ids"]
    """

    vocab_files_names = VOCAB_FILES_NAMES
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP

    prefix_tokens: List[int] = []
    suffix_tokens: List[int] = []

    def __init__(
        self, *args, tokenizer_file=None, src_lang=None, tgt_lang=None, additional_special_tokens=None, **kwargs
    ):
        super().__init__(
            *args,
            tokenizer_file=tokenizer_file,
            src_lang=src_lang,
            tgt_lang=tgt_lang,
            additional_special_tokens=additional_special_tokens,
            **kwargs,
        )

        self.sp_model_size = len(self.sp_model)
        self.lang_code_to_id = {
            code: self.sp_model_size + i + self.fairseq_offset for i, code in enumerate(FAIRSEQ_LANGUAGE_CODES)
        }
        self.id_to_lang_code = {v: k for k, v in self.lang_code_to_id.items()}
        # self.fairseq_tokens_to_ids["<mask>"] = len(self.sp_model) + len(self.lang_code_to_id) + self.fairseq_offset

        self.fairseq_tokens_to_ids.update(self.lang_code_to_id)
        self.fairseq_ids_to_tokens = {v: k for k, v in self.fairseq_tokens_to_ids.items()}
        self._additional_special_tokens = list(self.lang_code_to_id.keys())

        if additional_special_tokens is not None:
            # Only add those special tokens if they are not already there.
            self._additional_special_tokens.extend(
                [t for t in additional_special_tokens if t not in self._additional_special_tokens]
            )

        self._src_lang = src_lang if src_lang is not None else "en_XX"
        self.cur_lang_code_id = self.lang_code_to_id[self._src_lang]
        self.tgt_lang = tgt_lang
        self.set_src_lang_special_tokens(self._src_lang)

    @property
    def vocab_size(self):
        return len(self.sp_model) + len(self.lang_code_to_id) + self.fairseq_offset  # + 1  # Plus 1 for the mask token

    @property
    def src_lang(self) -> str:
        return self._src_lang

    @src_lang.setter
    def src_lang(self, new_src_lang: str) -> None:
        self._src_lang = new_src_lang
        self.set_src_lang_special_tokens(self._src_lang)

    def get_special_tokens_mask(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None, already_has_special_tokens: bool = False
    ) -> List[int]:
        """
        Retrieve sequence ids from a token list that has no special tokens added. This method is called when adding
        special tokens using the tokenizer ``prepare_for_model`` method.

        Args:
            token_ids_0 (:obj:`List[int]`):
                List of IDs.
            token_ids_1 (:obj:`List[int]`, `optional`):
                Optional second list of IDs for sequence pairs.
            already_has_special_tokens (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not the token list is already formatted with special tokens for the model.

        Returns:
            :obj:`List[int]`: A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.
        """

        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=True
            )

        prefix_ones = [1] * len(self.prefix_tokens)
        suffix_ones = [1] * len(self.suffix_tokens)
        if token_ids_1 is None:
            return prefix_ones + ([0] * len(token_ids_0)) + suffix_ones
        return prefix_ones + ([0] * len(token_ids_0)) + ([0] * len(token_ids_1)) + suffix_ones

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
        adding special tokens. An PLBART sequence has the following format, where ``X`` represents the sequence:

        - ``input_ids`` (for encoder) ``X [eos, src_lang_code]``
        - ``decoder_input_ids``: (for decoder) ``X [eos, tgt_lang_code]``

        BOS is never used. Pairs of sequences are not the expected use case, but they will be handled without a
        separator.

        Args:
            token_ids_0 (:obj:`List[int]`):
                List of IDs to which the special tokens will be added.
            token_ids_1 (:obj:`List[int]`, `optional`):
                Optional second list of IDs for sequence pairs.

        Returns:
            :obj:`List[int]`: List of `input IDs <../glossary.html#input-ids>`__ with the appropriate special tokens.
        """
        if token_ids_1 is None:
            return self.prefix_tokens + token_ids_0 + self.suffix_tokens
        # We don't expect to process pairs, but leave the pair logic for API consistency
        return self.prefix_tokens + token_ids_0 + token_ids_1 + self.suffix_tokens

    def _build_translation_inputs(
        self, raw_inputs, return_tensors: str, src_lang: Optional[str], tgt_lang: Optional[str], **extra_kwargs
    ):
        """Used by translation pipeline, to prepare inputs for the generate function"""
        if src_lang is None or tgt_lang is None:
            raise ValueError("Translation requires a `src_lang` and a `tgt_lang` for this model")
        self.src_lang = src_lang
        inputs = self(raw_inputs, add_special_tokens=True, return_tensors=return_tensors, **extra_kwargs)
        tgt_lang_id = self.convert_tokens_to_ids(tgt_lang)
        inputs["forced_bos_token_id"] = tgt_lang_id
        return inputs

    def prepare_seq2seq_batch(
        self,
        src_texts: List[str],
        src_lang: str = "en_XX",
        tgt_texts: Optional[List[str]] = None,
        tgt_lang: str = "python",
        **kwargs,
    ) -> BatchEncoding:
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        return super().prepare_seq2seq_batch(src_texts, tgt_texts, **kwargs)

    @contextmanager
    def as_target_tokenizer(self):
        """
        Temporarily sets the tokenizer for encoding the targets. Useful for tokenizer associated to
        sequence-to-sequence models that need a slightly different processing for the labels.
        """
        self.set_tgt_lang_special_tokens(self.tgt_lang)
        yield
        self.set_src_lang_special_tokens(self.src_lang)

    def set_src_lang_special_tokens(self, src_lang) -> None:
        """Reset the special tokens to the source lang setting. No prefix and suffix=[eos, src_lang_code]."""
        self.cur_lang_code = self.lang_code_to_id[src_lang]
        self.prefix_tokens = []
        self.suffix_tokens = [self.eos_token_id, self.cur_lang_code]

    def set_tgt_lang_special_tokens(self, lang: str) -> None:
        """Reset the special tokens to the target language setting. No prefix and suffix=[eos, tgt_lang_code]."""
        self.cur_lang_code = self.lang_code_to_id[lang]
        self.prefix_tokens = []
        self.suffix_tokens = [self.eos_token_id, self.cur_lang_code]
