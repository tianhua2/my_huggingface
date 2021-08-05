# THIS FILE HAS BEEN AUTOGENERATED. To update:
# 1. modify: models/auto/modeling_auto.py
# 2. run: python utils/class_mapping_update.py
from collections import OrderedDict


MODEL_FOR_QUESTION_ANSWERING_MAPPING_NAMES = OrderedDict(
    [
        ("RemBertConfig", "RemBertForQuestionAnswering"),
        ("CanineConfig", "CanineForQuestionAnswering"),
        ("RoFormerConfig", "RoFormerForQuestionAnswering"),
        ("BigBirdPegasusConfig", "BigBirdPegasusForQuestionAnswering"),
        ("BigBirdConfig", "BigBirdForQuestionAnswering"),
        ("ConvBertConfig", "ConvBertForQuestionAnswering"),
        ("LEDConfig", "LEDForQuestionAnswering"),
        ("DistilBertConfig", "DistilBertForQuestionAnswering"),
        ("AlbertConfig", "AlbertForQuestionAnswering"),
        ("CamembertConfig", "CamembertForQuestionAnswering"),
        ("BartConfig", "BartForQuestionAnswering"),
        ("MBartConfig", "MBartForQuestionAnswering"),
        ("LongformerConfig", "LongformerForQuestionAnswering"),
        ("XLMRobertaConfig", "XLMRobertaForQuestionAnswering"),
        ("RobertaConfig", "RobertaForQuestionAnswering"),
        ("SqueezeBertConfig", "SqueezeBertForQuestionAnswering"),
        ("BertConfig", "BertForQuestionAnswering"),
        ("XLNetConfig", "XLNetForQuestionAnsweringSimple"),
        ("FlaubertConfig", "FlaubertForQuestionAnsweringSimple"),
        ("MegatronBertConfig", "MegatronBertForQuestionAnswering"),
        ("MobileBertConfig", "MobileBertForQuestionAnswering"),
        ("XLMConfig", "XLMForQuestionAnsweringSimple"),
        ("ElectraConfig", "ElectraForQuestionAnswering"),
        ("ReformerConfig", "ReformerForQuestionAnswering"),
        ("FunnelConfig", "FunnelForQuestionAnswering"),
        ("LxmertConfig", "LxmertForQuestionAnswering"),
        ("MPNetConfig", "MPNetForQuestionAnswering"),
        ("DebertaConfig", "DebertaForQuestionAnswering"),
        ("DebertaV2Config", "DebertaV2ForQuestionAnswering"),
        ("IBertConfig", "IBertForQuestionAnswering"),
    ]
)


MODEL_FOR_CAUSAL_LM_MAPPING_NAMES = OrderedDict(
    [
        ("RemBertConfig", "RemBertForCausalLM"),
        ("RoFormerConfig", "RoFormerForCausalLM"),
        ("BigBirdPegasusConfig", "BigBirdPegasusForCausalLM"),
        ("GPTNeoConfig", "GPTNeoForCausalLM"),
        ("BigBirdConfig", "BigBirdForCausalLM"),
        ("CamembertConfig", "CamembertForCausalLM"),
        ("XLMRobertaConfig", "XLMRobertaForCausalLM"),
        ("RobertaConfig", "RobertaForCausalLM"),
        ("BertConfig", "BertLMHeadModel"),
        ("OpenAIGPTConfig", "OpenAIGPTLMHeadModel"),
        ("GPT2Config", "GPT2LMHeadModel"),
        ("TransfoXLConfig", "TransfoXLLMHeadModel"),
        ("XLNetConfig", "XLNetLMHeadModel"),
        ("XLMConfig", "XLMWithLMHeadModel"),
        ("CTRLConfig", "CTRLLMHeadModel"),
        ("ReformerConfig", "ReformerModelWithLMHead"),
        ("BertGenerationConfig", "BertGenerationDecoder"),
        ("XLMProphetNetConfig", "XLMProphetNetForCausalLM"),
        ("ProphetNetConfig", "ProphetNetForCausalLM"),
        ("BartConfig", "BartForCausalLM"),
        ("MBartConfig", "MBartForCausalLM"),
        ("PegasusConfig", "PegasusForCausalLM"),
        ("MarianConfig", "MarianForCausalLM"),
        ("BlenderbotConfig", "BlenderbotForCausalLM"),
        ("BlenderbotSmallConfig", "BlenderbotSmallForCausalLM"),
        ("MegatronBertConfig", "MegatronBertForCausalLM"),
    ]
)


MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING_NAMES = OrderedDict(
    [
        ("ViTConfig", "ViTForImageClassification"),
        ("DeiTConfig", "('DeiTForImageClassification', 'DeiTForImageClassificationWithTeacher')"),
        ("BeitConfig", "BeitForImageClassification"),
    ]
)


MODEL_FOR_MASKED_LM_MAPPING_NAMES = OrderedDict(
    [
        ("RemBertConfig", "RemBertForMaskedLM"),
        ("RoFormerConfig", "RoFormerForMaskedLM"),
        ("BigBirdConfig", "BigBirdForMaskedLM"),
        ("Wav2Vec2Config", "Wav2Vec2ForMaskedLM"),
        ("ConvBertConfig", "ConvBertForMaskedLM"),
        ("LayoutLMConfig", "LayoutLMForMaskedLM"),
        ("DistilBertConfig", "DistilBertForMaskedLM"),
        ("AlbertConfig", "AlbertForMaskedLM"),
        ("BartConfig", "BartForConditionalGeneration"),
        ("MBartConfig", "MBartForConditionalGeneration"),
        ("CamembertConfig", "CamembertForMaskedLM"),
        ("XLMRobertaConfig", "XLMRobertaForMaskedLM"),
        ("LongformerConfig", "LongformerForMaskedLM"),
        ("RobertaConfig", "RobertaForMaskedLM"),
        ("SqueezeBertConfig", "SqueezeBertForMaskedLM"),
        ("BertConfig", "BertForMaskedLM"),
        ("MegatronBertConfig", "MegatronBertForMaskedLM"),
        ("MobileBertConfig", "MobileBertForMaskedLM"),
        ("FlaubertConfig", "FlaubertWithLMHeadModel"),
        ("XLMConfig", "XLMWithLMHeadModel"),
        ("ElectraConfig", "ElectraForMaskedLM"),
        ("ReformerConfig", "ReformerForMaskedLM"),
        ("FunnelConfig", "FunnelForMaskedLM"),
        ("MPNetConfig", "MPNetForMaskedLM"),
        ("TapasConfig", "TapasForMaskedLM"),
        ("DebertaConfig", "DebertaForMaskedLM"),
        ("DebertaV2Config", "DebertaV2ForMaskedLM"),
        ("IBertConfig", "IBertForMaskedLM"),
    ]
)


MODEL_FOR_MULTIPLE_CHOICE_MAPPING_NAMES = OrderedDict(
    [
        ("RemBertConfig", "RemBertForMultipleChoice"),
        ("CanineConfig", "CanineForMultipleChoice"),
        ("RoFormerConfig", "RoFormerForMultipleChoice"),
        ("BigBirdConfig", "BigBirdForMultipleChoice"),
        ("ConvBertConfig", "ConvBertForMultipleChoice"),
        ("CamembertConfig", "CamembertForMultipleChoice"),
        ("ElectraConfig", "ElectraForMultipleChoice"),
        ("XLMRobertaConfig", "XLMRobertaForMultipleChoice"),
        ("LongformerConfig", "LongformerForMultipleChoice"),
        ("RobertaConfig", "RobertaForMultipleChoice"),
        ("SqueezeBertConfig", "SqueezeBertForMultipleChoice"),
        ("BertConfig", "BertForMultipleChoice"),
        ("DistilBertConfig", "DistilBertForMultipleChoice"),
        ("MegatronBertConfig", "MegatronBertForMultipleChoice"),
        ("MobileBertConfig", "MobileBertForMultipleChoice"),
        ("XLNetConfig", "XLNetForMultipleChoice"),
        ("AlbertConfig", "AlbertForMultipleChoice"),
        ("XLMConfig", "XLMForMultipleChoice"),
        ("FlaubertConfig", "FlaubertForMultipleChoice"),
        ("FunnelConfig", "FunnelForMultipleChoice"),
        ("MPNetConfig", "MPNetForMultipleChoice"),
        ("IBertConfig", "IBertForMultipleChoice"),
    ]
)


MODEL_FOR_NEXT_SENTENCE_PREDICTION_MAPPING_NAMES = OrderedDict(
    [
        ("BertConfig", "BertForNextSentencePrediction"),
        ("MegatronBertConfig", "MegatronBertForNextSentencePrediction"),
        ("MobileBertConfig", "MobileBertForNextSentencePrediction"),
    ]
)


MODEL_FOR_OBJECT_DETECTION_MAPPING_NAMES = OrderedDict(
    [
        ("DetrConfig", "DetrForObjectDetection"),
    ]
)


MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING_NAMES = OrderedDict(
    [
        ("BigBirdPegasusConfig", "BigBirdPegasusForConditionalGeneration"),
        ("M2M100Config", "M2M100ForConditionalGeneration"),
        ("LEDConfig", "LEDForConditionalGeneration"),
        ("BlenderbotSmallConfig", "BlenderbotSmallForConditionalGeneration"),
        ("MT5Config", "MT5ForConditionalGeneration"),
        ("T5Config", "T5ForConditionalGeneration"),
        ("PegasusConfig", "PegasusForConditionalGeneration"),
        ("MarianConfig", "MarianMTModel"),
        ("MBartConfig", "MBartForConditionalGeneration"),
        ("BlenderbotConfig", "BlenderbotForConditionalGeneration"),
        ("BartConfig", "BartForConditionalGeneration"),
        ("FSMTConfig", "FSMTForConditionalGeneration"),
        ("EncoderDecoderConfig", "EncoderDecoderModel"),
        ("XLMProphetNetConfig", "XLMProphetNetForConditionalGeneration"),
        ("ProphetNetConfig", "ProphetNetForConditionalGeneration"),
    ]
)


MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING_NAMES = OrderedDict(
    [
        ("RemBertConfig", "RemBertForSequenceClassification"),
        ("CanineConfig", "CanineForSequenceClassification"),
        ("RoFormerConfig", "RoFormerForSequenceClassification"),
        ("BigBirdPegasusConfig", "BigBirdPegasusForSequenceClassification"),
        ("BigBirdConfig", "BigBirdForSequenceClassification"),
        ("ConvBertConfig", "ConvBertForSequenceClassification"),
        ("LEDConfig", "LEDForSequenceClassification"),
        ("DistilBertConfig", "DistilBertForSequenceClassification"),
        ("AlbertConfig", "AlbertForSequenceClassification"),
        ("CamembertConfig", "CamembertForSequenceClassification"),
        ("XLMRobertaConfig", "XLMRobertaForSequenceClassification"),
        ("MBartConfig", "MBartForSequenceClassification"),
        ("BartConfig", "BartForSequenceClassification"),
        ("LongformerConfig", "LongformerForSequenceClassification"),
        ("RobertaConfig", "RobertaForSequenceClassification"),
        ("SqueezeBertConfig", "SqueezeBertForSequenceClassification"),
        ("LayoutLMConfig", "LayoutLMForSequenceClassification"),
        ("BertConfig", "BertForSequenceClassification"),
        ("XLNetConfig", "XLNetForSequenceClassification"),
        ("MegatronBertConfig", "MegatronBertForSequenceClassification"),
        ("MobileBertConfig", "MobileBertForSequenceClassification"),
        ("FlaubertConfig", "FlaubertForSequenceClassification"),
        ("XLMConfig", "XLMForSequenceClassification"),
        ("ElectraConfig", "ElectraForSequenceClassification"),
        ("FunnelConfig", "FunnelForSequenceClassification"),
        ("DebertaConfig", "DebertaForSequenceClassification"),
        ("DebertaV2Config", "DebertaV2ForSequenceClassification"),
        ("GPT2Config", "GPT2ForSequenceClassification"),
        ("GPTNeoConfig", "GPTNeoForSequenceClassification"),
        ("OpenAIGPTConfig", "OpenAIGPTForSequenceClassification"),
        ("ReformerConfig", "ReformerForSequenceClassification"),
        ("CTRLConfig", "CTRLForSequenceClassification"),
        ("TransfoXLConfig", "TransfoXLForSequenceClassification"),
        ("MPNetConfig", "MPNetForSequenceClassification"),
        ("TapasConfig", "TapasForSequenceClassification"),
        ("IBertConfig", "IBertForSequenceClassification"),
    ]
)


MODEL_FOR_TABLE_QUESTION_ANSWERING_MAPPING_NAMES = OrderedDict(
    [
        ("TapasConfig", "TapasForQuestionAnswering"),
    ]
)


MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING_NAMES = OrderedDict(
    [
        ("RemBertConfig", "RemBertForTokenClassification"),
        ("CanineConfig", "CanineForTokenClassification"),
        ("RoFormerConfig", "RoFormerForTokenClassification"),
        ("BigBirdConfig", "BigBirdForTokenClassification"),
        ("ConvBertConfig", "ConvBertForTokenClassification"),
        ("LayoutLMConfig", "LayoutLMForTokenClassification"),
        ("DistilBertConfig", "DistilBertForTokenClassification"),
        ("CamembertConfig", "CamembertForTokenClassification"),
        ("FlaubertConfig", "FlaubertForTokenClassification"),
        ("XLMConfig", "XLMForTokenClassification"),
        ("XLMRobertaConfig", "XLMRobertaForTokenClassification"),
        ("LongformerConfig", "LongformerForTokenClassification"),
        ("RobertaConfig", "RobertaForTokenClassification"),
        ("SqueezeBertConfig", "SqueezeBertForTokenClassification"),
        ("BertConfig", "BertForTokenClassification"),
        ("MegatronBertConfig", "MegatronBertForTokenClassification"),
        ("MobileBertConfig", "MobileBertForTokenClassification"),
        ("XLNetConfig", "XLNetForTokenClassification"),
        ("AlbertConfig", "AlbertForTokenClassification"),
        ("ElectraConfig", "ElectraForTokenClassification"),
        ("FunnelConfig", "FunnelForTokenClassification"),
        ("MPNetConfig", "MPNetForTokenClassification"),
        ("DebertaConfig", "DebertaForTokenClassification"),
        ("DebertaV2Config", "DebertaV2ForTokenClassification"),
        ("IBertConfig", "IBertForTokenClassification"),
    ]
)


MODEL_MAPPING_NAMES = OrderedDict(
    [
        ("BeitConfig", "BeitModel"),
        ("RemBertConfig", "RemBertModel"),
        ("VisualBertConfig", "VisualBertModel"),
        ("CanineConfig", "CanineModel"),
        ("RoFormerConfig", "RoFormerModel"),
        ("CLIPConfig", "CLIPModel"),
        ("BigBirdPegasusConfig", "BigBirdPegasusModel"),
        ("DeiTConfig", "DeiTModel"),
        ("LukeConfig", "LukeModel"),
        ("DetrConfig", "DetrModel"),
        ("GPTNeoConfig", "GPTNeoModel"),
        ("BigBirdConfig", "BigBirdModel"),
        ("Speech2TextConfig", "Speech2TextModel"),
        ("ViTConfig", "ViTModel"),
        ("Wav2Vec2Config", "Wav2Vec2Model"),
        ("HubertConfig", "HubertModel"),
        ("M2M100Config", "M2M100Model"),
        ("ConvBertConfig", "ConvBertModel"),
        ("LEDConfig", "LEDModel"),
        ("BlenderbotSmallConfig", "BlenderbotSmallModel"),
        ("RetriBertConfig", "RetriBertModel"),
        ("MT5Config", "MT5Model"),
        ("T5Config", "T5Model"),
        ("PegasusConfig", "PegasusModel"),
        ("MarianConfig", "MarianModel"),
        ("MBartConfig", "MBartModel"),
        ("BlenderbotConfig", "BlenderbotModel"),
        ("DistilBertConfig", "DistilBertModel"),
        ("AlbertConfig", "AlbertModel"),
        ("CamembertConfig", "CamembertModel"),
        ("XLMRobertaConfig", "XLMRobertaModel"),
        ("BartConfig", "BartModel"),
        ("LongformerConfig", "LongformerModel"),
        ("RobertaConfig", "RobertaModel"),
        ("LayoutLMConfig", "LayoutLMModel"),
        ("SqueezeBertConfig", "SqueezeBertModel"),
        ("BertConfig", "BertModel"),
        ("OpenAIGPTConfig", "OpenAIGPTModel"),
        ("GPT2Config", "GPT2Model"),
        ("MegatronBertConfig", "MegatronBertModel"),
        ("MobileBertConfig", "MobileBertModel"),
        ("TransfoXLConfig", "TransfoXLModel"),
        ("XLNetConfig", "XLNetModel"),
        ("FlaubertConfig", "FlaubertModel"),
        ("FSMTConfig", "FSMTModel"),
        ("XLMConfig", "XLMModel"),
        ("CTRLConfig", "CTRLModel"),
        ("ElectraConfig", "ElectraModel"),
        ("ReformerConfig", "ReformerModel"),
        ("FunnelConfig", "('FunnelModel', 'FunnelBaseModel')"),
        ("LxmertConfig", "LxmertModel"),
        ("BertGenerationConfig", "BertGenerationEncoder"),
        ("DebertaConfig", "DebertaModel"),
        ("DebertaV2Config", "DebertaV2Model"),
        ("DPRConfig", "DPRQuestionEncoder"),
        ("XLMProphetNetConfig", "XLMProphetNetModel"),
        ("ProphetNetConfig", "ProphetNetModel"),
        ("MPNetConfig", "MPNetModel"),
        ("TapasConfig", "TapasModel"),
        ("IBertConfig", "IBertModel"),
    ]
)


MODEL_WITH_LM_HEAD_MAPPING_NAMES = OrderedDict(
    [
        ("RemBertConfig", "RemBertForMaskedLM"),
        ("RoFormerConfig", "RoFormerForMaskedLM"),
        ("BigBirdPegasusConfig", "BigBirdPegasusForConditionalGeneration"),
        ("GPTNeoConfig", "GPTNeoForCausalLM"),
        ("BigBirdConfig", "BigBirdForMaskedLM"),
        ("Speech2TextConfig", "Speech2TextForConditionalGeneration"),
        ("Wav2Vec2Config", "Wav2Vec2ForMaskedLM"),
        ("M2M100Config", "M2M100ForConditionalGeneration"),
        ("ConvBertConfig", "ConvBertForMaskedLM"),
        ("LEDConfig", "LEDForConditionalGeneration"),
        ("BlenderbotSmallConfig", "BlenderbotSmallForConditionalGeneration"),
        ("LayoutLMConfig", "LayoutLMForMaskedLM"),
        ("T5Config", "T5ForConditionalGeneration"),
        ("DistilBertConfig", "DistilBertForMaskedLM"),
        ("AlbertConfig", "AlbertForMaskedLM"),
        ("CamembertConfig", "CamembertForMaskedLM"),
        ("XLMRobertaConfig", "XLMRobertaForMaskedLM"),
        ("MarianConfig", "MarianMTModel"),
        ("FSMTConfig", "FSMTForConditionalGeneration"),
        ("BartConfig", "BartForConditionalGeneration"),
        ("LongformerConfig", "LongformerForMaskedLM"),
        ("RobertaConfig", "RobertaForMaskedLM"),
        ("SqueezeBertConfig", "SqueezeBertForMaskedLM"),
        ("BertConfig", "BertForMaskedLM"),
        ("OpenAIGPTConfig", "OpenAIGPTLMHeadModel"),
        ("GPT2Config", "GPT2LMHeadModel"),
        ("MegatronBertConfig", "MegatronBertForCausalLM"),
        ("MobileBertConfig", "MobileBertForMaskedLM"),
        ("TransfoXLConfig", "TransfoXLLMHeadModel"),
        ("XLNetConfig", "XLNetLMHeadModel"),
        ("FlaubertConfig", "FlaubertWithLMHeadModel"),
        ("XLMConfig", "XLMWithLMHeadModel"),
        ("CTRLConfig", "CTRLLMHeadModel"),
        ("ElectraConfig", "ElectraForMaskedLM"),
        ("EncoderDecoderConfig", "EncoderDecoderModel"),
        ("ReformerConfig", "ReformerModelWithLMHead"),
        ("FunnelConfig", "FunnelForMaskedLM"),
        ("MPNetConfig", "MPNetForMaskedLM"),
        ("TapasConfig", "TapasForMaskedLM"),
        ("DebertaConfig", "DebertaForMaskedLM"),
        ("DebertaV2Config", "DebertaV2ForMaskedLM"),
        ("IBertConfig", "IBertForMaskedLM"),
    ]
)
