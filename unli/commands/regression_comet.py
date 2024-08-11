import os

import argparse
from unli.models import SentencePairModel
from unli.modules import CoupledSentencePairFeatureExtractor, BERTConcatenator, BertSeq2VecEncoderForPairs, MLP , BartSeq2VecEncoderForPairs , CoupledSentencePairFeatureExtractorBart
from unli.data.qrels import QRelsPointwiseReader
from unli.data.tokenizers import BertTokenizer , BartTokenizerWrapper , BartTokenIndexer
from allennlp.common import Params
from allennlp.data.token_indexers import PretrainedTransformerIndexer
from allennlp.data.iterators import BasicIterator
from allennlp.training import Trainer
from allennlp.data import Vocabulary
import torch
import logging
from torch.optim import Adam
import pandas as pd

parser = argparse.ArgumentParser(description="")
parser.add_argument("--data", type=str, default="", help="Path to QRels data")
parser.add_argument("--pretrained", type=str, default="", help="Pretrained model")
parser.add_argument("--out", type=str, default="", help="Output path")
parser.add_argument("--margin", type=float, default=0.3, help="")
parser.add_argument("--num_samples", type=int, default=1, help="")
parser.add_argument("--seed", type=int, default=0xCAFEBABE, help="")
parser.add_argument("--batch_size", type=int, default=16, help="")
parser.add_argument("--gpuid", type=int, default=0)
ARGS = parser.parse_args()


import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer, AutoConfig , BartTokenizer, BartForConditionalGeneration , AutoTokenizer , BartConfig
# from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Define the path to the COMET-ATOMIC-2020 model files
model_path = "/sise/home/orisim/projects/UNLI/comet-atomic_2020_BART_aaai"

# Load the configuration
config = BartConfig.from_pretrained(model_path)

# Load the model
comet_model = BartForConditionalGeneration.from_pretrained(model_path)  # , config=config
# comet_tokenizer = BartTokenizer.from_pretrained(model_path)


logging.basicConfig(level=logging.INFO)
torch.manual_seed(ARGS.seed)
vocab = Vocabulary()

model: torch.nn.Module = SentencePairModel(
    extractor=CoupledSentencePairFeatureExtractorBart(
        joiner=BERTConcatenator(),
        encoder=BartSeq2VecEncoderForPairs(config,model_path).from_pretrained(model_path) # BertSeq2VecEncoderForPairs.from_pretrained("bert-base-uncased")
    ),
    mlp=torch.nn.Sequential(
        torch.nn.Linear(768, 1),
        torch.nn.Sigmoid()
    ),
    loss_func=torch.nn.BCELoss(),
    mode="regression"
)
model.cuda()

if ARGS.pretrained != "":
    model.load_state_dict(torch.load(ARGS.pretrained))

reader = QRelsPointwiseReader(
    lazy=True,
    token_indexers={"wordpiece": BartTokenIndexer(model_path, special_tokens_path =os.path.join(model_path,"special_tokens_map.json"))  }, # PretrainedTransformerIndexer("bert-base-uncased", do_lowercase=True) BartTokenIndexer(model_path, special_tokens_path =os.path.join(model_path,"special_tokens_map.json"))
    left_tokenizer= BartTokenizerWrapper(model_path, special_tokens_path =os.path.join(model_path,"special_tokens_map.json")), # BertTokenizer(), BartTokenizerWrapper(model_path, special_tokens_path =os.path.join(model_path,"special_tokens_map.json"))
    right_tokenizer= BartTokenizerWrapper(model_path, special_tokens_path =os.path.join(model_path,"special_tokens_map.json")) # BertTokenizer() BartTokenizerWrapper(model_path , special_tokens_path = os.path.join(model_path,"special_tokens_map.json")
)
iterator = BasicIterator(batch_size=ARGS.batch_size)
iterator.index_with(vocab)


trainer = Trainer(
    model=model,
    optimizer=Adam(params=model.parameters(), lr=0.00001),
    grad_norm=1.0,
    train_dataset=reader.read(f"{ARGS.data}/train"),
    validation_dataset=reader.read(f"{ARGS.data}/dev"),
    iterator=iterator,
    validation_metric="+pearson",
    num_epochs=3,
    patience=3,
    serialization_dir=ARGS.out,
    cuda_device=ARGS.gpuid
)

trainer.train()

# output_path_dev = r"/sise/home/orisim/projects/UNLI/results_predictions_dev.csv"
# output_path_train = r"/sise/home/orisim/projects/UNLI/results_predictions_train.csv"
#
# # Save to a CSV file
# df = pd.DataFrame(model.results_predictions_dev)
# df.to_csv(output_path_dev, index=False)

# df = pd.DataFrame(model.results_predictions_t