from allennlp.data.tokenizers import Token, Tokenizer
from allennlp.data.token_indexers import SingleIdTokenIndexer

from typing import *
import transformers
import json
import torch
from allennlp.data.vocabulary import Vocabulary

class BertTokenizer(Tokenizer):

    def __init__(self,
                 pretrained_model_name: str = "bert-base-uncased",
                 prefix: str = "[CLS]",
                 suffix: str = "[SEP]"):
        self.underlying = transformers.BertTokenizer.from_pretrained(pretrained_model_name)
        self.prefix = prefix
        self.suffix = suffix

    def tokenize(self, text: str) -> List[Token]:
        tokens = self.underlying.tokenize(text)
        if self.prefix is not None:
            tokens = [self.prefix] + tokens
        if self.suffix is not None:
            tokens = tokens + [self.suffix]
        return [Token(t) for t in tokens]

    def batch_tokenize(self, texts: List[str]) -> List[List[Token]]:
        return [self.tokenize(s) for s in texts]


class BartTokenizerWrapper(Tokenizer):

    def __init__(self,
                 pretrained_model_name: str = "facebook/bart-base",
                 special_tokens_path: str = None):
        if special_tokens_path:
            # Load special tokens from JSON file
            with open(special_tokens_path, "r") as f:
                special_tokens = json.load(f)
            # Convert special tokens from dictionaries to strings
            special_tokens_str = {k: v['content'] for k, v in special_tokens.items()}
            # Initialize the tokenizer with the converted special tokens
            self.underlying = transformers.BartTokenizer.from_pretrained(
                pretrained_model_name,
                bos_token=special_tokens_str.get("bos_token"),
                eos_token=special_tokens_str.get("eos_token"),
                unk_token=special_tokens_str.get("unk_token"),
                sep_token=special_tokens_str.get("sep_token"),
                pad_token=special_tokens_str.get("pad_token"),
                cls_token=special_tokens_str.get("cls_token"),
                mask_token=special_tokens_str.get("mask_token")
            )
        else:
            self.underlying = transformers.BartTokenizer.from_pretrained(pretrained_model_name)

    # def tokenize(self, text: str) -> List[Token]:
    #     tokens = self.underlying.tokenize(text)
    #     return [Token(t) for t in tokens]
    #
    # def batch_tokenize(self, texts: List[str]) -> List[List[Token]]:
    #     return [self.tokenize(s) for s in texts]

class BartTokenIndexer(SingleIdTokenIndexer):


        def __init__(self, model_name: str, namespace: str = "wordpiece", lowercase_tokens: bool = False,
                     start_tokens: List[str] = None, end_tokens: List[str] = None, special_tokens_path: str = None):
            super().__init__(token_min_padding_length=0)
            # self.tokenizer = BartTokenizer.from_pretrained(model_name)

            # Load special tokens from JSON file
            with open(special_tokens_path, "r") as f:
                special_tokens = json.load(f)
            # Convert special tokens from dictionaries to strings
            special_tokens_str = {k: v['content'] for k, v in special_tokens.items()}
            # Initialize the tokenizer with the converted special tokens
            self.tokenizer = transformers.BartTokenizer.from_pretrained(
                model_name,
                bos_token=special_tokens_str.get("bos_token"),
                eos_token=special_tokens_str.get("eos_token"),
                unk_token=special_tokens_str.get("unk_token"),
                sep_token=special_tokens_str.get("sep_token"),
                pad_token=special_tokens_str.get("pad_token"),
                cls_token=special_tokens_str.get("cls_token"),
                mask_token=special_tokens_str.get("mask_token")
            )

            self.namespace = namespace
            self.lowercase_tokens = lowercase_tokens
            self.start_tokens = start_tokens or []
            self.end_tokens = end_tokens or []

        def count_vocab_items(self, token: Token, counter: Dict[str, Dict[str, int]]):
            text = token.text.lower() if self.lowercase_tokens else token.text
            counter[self.namespace][text] += 1

        def tokens_to_indices(self, tokens: List[Token], vocabulary: Vocabulary , index_name: str) -> Dict[
            str, List[int]]:
            # Convert tokens to text
            text_tokens = [token.text for token in tokens]

            # Add start and end tokens if specified
            text_tokens = self.start_tokens + text_tokens + self.end_tokens

            # Tokenize using BART tokenizer
            indices = self.tokenizer.convert_tokens_to_ids(text_tokens)

            return {index_name: indices}

        def get_padding_lengths(self, token: int) -> Dict[str, int]:
            return {}

        def as_padded_tensor(self, tokens: Dict[str, List[int]], desired_num_tokens: Dict[str, int],
                             padding_lengths: Dict[str, int]) -> Dict[str, torch.Tensor]:

            # print(f"tokens: {tokens}")
            padded_tokens = tokens["wordpiece"]
            desired_length = desired_num_tokens["wordpiece"]

            # Pad or truncate the list of token indices to the desired length
            padded_tokens = padded_tokens[:desired_length] + [self.tokenizer.pad_token_id] * (
                        desired_length - len(padded_tokens))

            print(f"new tokens: {padded_tokens}")

            # Convert token IDs to a single string
            decoded_string = self.tokenizer.decode(padded_tokens)

            # Convert token IDs to individual tokens
            tokens = self.tokenizer.convert_ids_to_tokens(padded_tokens)

            # print("Decoded string:", decoded_string)
            # print("Tokens:", tokens)
            # print(f"the size of padded_tokens is : {torch.tensor(padded_tokens).size()}")
            return {self.namespace: torch.tensor(padded_tokens)}

        def get_keys(self, index_name: str) -> List[str]:
            return [index_name]