from typing import *
import torch
import numpy as np
import transformers
from allennlp.modules.seq2vec_encoders import Seq2VecEncoder
from torch import nn
from huggingface_hub import PyTorchModelHubMixin


class BertSeq2VecEncoderForPairs(transformers.BertPreTrainedModel, Seq2VecEncoder):

    def __init__(self, config: transformers.BertConfig):
        super(BertSeq2VecEncoderForPairs, self).__init__(config)
        self.bert = transformers.BertModel(config)
        self.dropout = torch.nn.Dropout(0.1)

    @classmethod
    def from_params(cls, params):
        return cls.from_pretrained(params["pretrained_bert_model_name"])

    def forward(self, stm: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]):
        s, t, m = stm
        _, pooled = self.bert(
            input_ids=s,
            token_type_ids=t,
            attention_mask=m
        )
        return self.dropout(pooled)

    def get_input_dim(self) -> int:
        pass

    def get_output_dim(self) -> int:
        pass


class BartSeq2VecEncoderForPairs(transformers.PreTrainedModel, Seq2VecEncoder):

    config_class = transformers.BartConfig

    def __init__(self, config: transformers.BartConfig = None , model_path: str = None):
        super(BartSeq2VecEncoderForPairs, self).__init__(config)
        self.bart = transformers.BartForConditionalGeneration(config)
        self.dropout = torch.nn.Dropout(0.1)

    @classmethod
    def from_params(cls, params):
        print(f"from params: {params}")
        return cls.from_pretrained(params["pretrained_bart_model_name"])

    def forward(self, stm: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]):


        s, t, m = stm

        outputs = self.bart(
            input_ids=s,
            attention_mask=m
        )

        pooled = outputs[0].mean(dim=1)  # Take the representation of the first token ([CLS] equivalent)


        return self.dropout(pooled)

    def get_input_dim(self) -> int:
        pass
        # return self.bart.config.d_model

    def get_output_dim(self) -> int:
        pass
        # return self.bart.config.d_model