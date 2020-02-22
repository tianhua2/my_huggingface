from enum import Enum
from transformers.pipelines import Pipeline, PipelineDataFormat, pipeline


class ModelType(str, Enum):
    BERT = "bert"
    GPT = "gpt"
    GPT2 = "gpt2"
    TRANSFORMER_XL = "transfo_xl"
    XLNET = "xlnet"
    XLM = "xlm"


class SupportedFormat(str, Enum):
    INFER = "infer"
    JSON = "json"
    CSV = "csv"
    PIPE = "pipe"


class SupportedTask(str, Enum):
    FEATURE_EXTRACTION = "feature-extraction"
    SENTIMENT_ANALYSIS = "sentiment-analysis"
    NER = "ner"
    QUESTION_ANSWERING = "question-answering"
    FILL_MASK = "fill-mask"