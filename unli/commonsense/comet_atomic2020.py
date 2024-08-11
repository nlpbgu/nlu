import json
import torch
import argparse
from tqdm import tqdm
from pathlib import Path
from transformers import  AutoTokenizer , BartForConditionalGeneration , BartTokenizer # ,  AutoModelForSeq2SeqLM
import os

class Comet:
    def __init__(self, model_path , modify_special_token = True):

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"model_path {model_path}")
        self.model = BartForConditionalGeneration.from_pretrained(pretrained_model_name_or_path = model_path).to(self.device) # , config = "https://huggingface.co/mismayil/comet-bart-ai2/resolve/main/config.json"
        if modify_special_token :
            self.tokenizer = self.modify_special_token(model_path , os.path.join(model_path,"special_tokens_map.json"))
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        task = "summarization"
        self.use_task_specific_params(self.model, task)
        self.batch_size = 1
        self.decoder_start_token_id = None

    def use_task_specific_params(self,model, task):
      """Update config with summarization specific params."""
      task_specific_params = model.config.task_specific_params
      if task_specific_params is not None:
          pars = task_specific_params.get(task, {})
          model.config.update(pars)

    def trim_batch(self,input_ids, pad_token_id, attention_mask=None,):

      """Remove columns that are populated exclusively by pad_token_id"""
      keep_column_mask = input_ids.ne(pad_token_id).any(dim=0)

      # print(f"pad_token_id:  ", pad_token_id)
      # print(f"input_ids in tm: {input_ids}")

      if attention_mask is None:
          return input_ids[:, keep_column_mask]
      else:
          # print(f"else: {(input_ids[:, keep_column_mask], attention_mask[:, keep_column_mask])}")
          return (input_ids[:, keep_column_mask], attention_mask[:, keep_column_mask])


    def chunks(self,lst, n):

      """Yield successive n-sized chunks from lst."""
      for i in range(0, len(lst), n):
          yield lst[i : i + n]

    def generate(
            self,
            queries,
            decode_method="beam",
            num_generate=5,
            ):

        with torch.no_grad():
            examples = queries

            max_length = 128

            decs = []
            for batch in list(self.chunks(examples, self.batch_size)):

                # batch = self.tokenizer.tokenize(batch[0], return_tensors="pt", truncation=True, padding="max_length").to(self.device)

                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

                # print("input:  ",batch[0])
                tokens = self.tokenizer.tokenize(batch[0])
                tokens = [self.tokenizer.cls_token] + tokens + [self.tokenizer.sep_token]

                # Truncate tokens if they exceed max_length
                if len(tokens) > max_length:
                    tokens = tokens[:max_length]

                # Convert tokens to IDs
                ids = self.tokenizer.convert_tokens_to_ids(tokens)

                # Padding
                attention_mask = [1] * len(ids)  # Mask indicating real tokens
                padding_length = max_length - len(ids)

                ids += [self.tokenizer.pad_token_id] * padding_length  # Pad with pad_token_id
                attention_mask += [0] * padding_length  # Pad the attention mask

                batch = {'input_ids':torch.tensor([ids]).to(self.device) ,'attention_mask': torch.tensor([attention_mask]).to(self.device) }
                # print(f"batch:  {batch}")

                input_ids, attention_mask = self.trim_batch(**batch, pad_token_id=self.tokenizer.pad_token_id)

                # print(f"input_ids:  {input_ids}")
                # print(f"attention_mask:  {attention_mask}")
                summaries = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    decoder_start_token_id=self.decoder_start_token_id,
                    num_beams=num_generate,
                    num_return_sequences=num_generate,
                    # top_p=0.99,
                    # top_k=num_generate
                    )

                dec = self.tokenizer.batch_decode(summaries, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                # print("dec",dec)

                decs.append(dec)

            return decs

    def modify_special_token(self,pretrained_model_name , special_tokens_path):

        with open(special_tokens_path, "r") as f:
            special_tokens = json.load(f)
            # Convert special tokens from dictionaries to strings
        special_tokens_str = {k: v['content'] for k, v in special_tokens.items()}
        # Initialize the tokenizer with the converted special tokens

        tokenizer = BartTokenizer.from_pretrained(
            pretrained_model_name,
            bos_token=special_tokens_str.get("bos_token"),
            eos_token=special_tokens_str.get("eos_token"),
            unk_token=special_tokens_str.get("unk_token"),
            sep_token=special_tokens_str.get("sep_token"),
            pad_token=special_tokens_str.get("pad_token"),
            cls_token=special_tokens_str.get("cls_token"),
            mask_token=special_tokens_str.get("mask_token")
        )

        return tokenizer





