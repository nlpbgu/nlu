import argparse
from unli.models import SentencePairModel
from unli.modules import CoupledSentencePairFeatureExtractor, BERTConcatenator, BertSeq2VecEncoderForPairs, MLP
from unli.data.qrels import QRelsPointwiseReader
from unli.data.tokenizers import BertTokenizer
from allennlp.common import Params
from allennlp.data.token_indexers import PretrainedTransformerIndexer
from allennlp.data.iterators import BasicIterator
from allennlp.training import Trainer
from allennlp.data import Vocabulary
import torch
import logging
from torch.optim import Adam
import pandas as pd
from scripts.util import copy_files_to_new_directory
import os
from unli.utils.augmentation_commonsense import Augmentation , BartAugmentation
from unli.data.storage import KeyValueStore


parser = argparse.ArgumentParser(description="")
parser.add_argument("--data", type=str, default="", help="Path to QRels data")
parser.add_argument("--pretrained", type=str, default="", help="Pretrained model")
parser.add_argument("--out", type=str, default="", help="Output path")
parser.add_argument("--margin", type=float, default=0.3, help="")
parser.add_argument("--num_samples", type=int, default=1, help="")
parser.add_argument("--seed", type=int, default=0xCAFEBABE, help="")
parser.add_argument("--batch_size", type=int, default=16, help="")
parser.add_argument("--gpuid", type=int, default=0)
parser.add_argument("--augmentation", type = str , help="Enable augmentation") # action='store_true'
parser.add_argument("--training_augmentation", action='store_true', help="trainig the augmentation you created")
parser.add_argument("--threshold", action='float', help="threshold we want to reference")


ARGS = parser.parse_args()

logging.basicConfig(level=logging.INFO)
torch.manual_seed(ARGS.seed)
vocab = Vocabulary()


dest_data_dir = os.path.join(ARGS.data, "augmentation")
augm_mode = None

if ARGS.augmentation :

    copy_files_to_new_directory(ARGS.data, dest_data_dir )
    kv_store =  KeyValueStore.open()
    data_dir_l = os.path.join(ARGS.data,"train.l")
    data_dir_r = os.path.join(ARGS.data,"train.r")
    kv_store.store_in_redis( data_dir_l)
    kv_store.store_in_redis( data_dir_r)

    modify_special_token = True
    if ARGS.augmentation == "comet":
        model_path = "/sise/home/orisim/projects/UNLI/comet-atomic_2020_BART_aaai"
        augm_mode = Augmentation(model_path, modify_special_token)

    if ARGS.augmentation == "bart":
        model_path = "stanford-oval/paraphraser-bart-large"
        model_path = "https://huggingface.co/stanford-oval/paraphraser-bart-large/blob/main/pytorch_model.bin"
        model_path = "/sise/home/orisim/projects/UNLI/bart_stanford/"
        modify_special_token = False
        augm_mode = BartAugmentation(model_path,modify_special_token)


model: torch.nn.Module = SentencePairModel(
    extractor=CoupledSentencePairFeatureExtractor(
        joiner=BERTConcatenator(),
        encoder=BertSeq2VecEncoderForPairs.from_pretrained("bert-base-uncased")
    ),
    mlp=torch.nn.Sequential(
        torch.nn.Linear(768, 1),
        torch.nn.Sigmoid()
    ),
    loss_func=torch.nn.BCELoss(),
    mode="regression",
    # reverse_vocab = vocab.get_index_to_token_vocabulary(namespace="tags"),
    augmentation = augm_mode ,  # Augmentation(model_path) if ARGS.augmentation else None
    kv_store = kv_store if ARGS.augmentation else None,
    data_dir = ARGS.data ,
    threshold = ARGS.threshold
)
model.cuda()

if ARGS.pretrained != "":
    model.load_state_dict(torch.load(ARGS.pretrained))

reader = QRelsPointwiseReader(
    lazy=True,
    token_indexers={"wordpiece": PretrainedTransformerIndexer("bert-base-uncased", do_lowercase=True)},
    left_tokenizer=BertTokenizer(),
    right_tokenizer=BertTokenizer()
)
iterator = BasicIterator(batch_size=ARGS.batch_size)
iterator.index_with(vocab)


trainer = Trainer(
    model=model,
    optimizer=Adam(params=model.parameters(), lr=0.00001),
    grad_norm=1.0,
    train_dataset=reader.read(f"{dest_data_dir if ARGS.training_augmentation  else ARGS.data}/train"), # dest_data_dir if ARGS.augmentation  else
    validation_dataset=reader.read(f"{dest_data_dir if ARGS.training_augmentation  else ARGS.data}/dev"),
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
#
# df = pd.DataFrame(model.results_predictions_train)
# df.to_csv(output_path_train, index=False)


