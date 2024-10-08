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
from scripts.util import copy_files_to_new_directory , download_files , remove_dir
import os
from unli.utils.augmentation_commonsense import Augmentation , BartAugmentation
from unli.data.storage import KeyValueStore
import re


def regression(rootdir,data,seed,pretrained,out,margin,num_samples,batch_size,gpuid,augmentation,training_augmentation,threshold,dir_augmentation):

    logging.basicConfig(level=logging.INFO)
    torch.manual_seed(seed)
    vocab = Vocabulary()

    dest_data_dir = os.path.join(data, "augmentation",dir_augmentation)
    augm_mode = None

    pattern = r"aug_([A-Za-z_]+)_threshold_([0-9.]+)_nli1_([0-9.]+|None)_nli2_([0-9.]+|None)_nli_([A-Za-z_]+|None)"

    match = re.search(pattern, dir_augmentation)
    augmentation_ , threshold , nli1 , nli2 , nli = match.group(1) , match.group(2) , match.group(3) , match.group(4) , match.group(5)



    if augmentation :

        copy_files_to_new_directory(data, dest_data_dir )
        remove_dir(out)

        kv_store =  KeyValueStore.open()
        data_dir_l = os.path.join(data,"train.l")
        data_dir_r = os.path.join(data,"train.r")
        kv_store.store_in_redis( data_dir_l)
        kv_store.store_in_redis( data_dir_r)

        modify_special_token = True

        if augmentation == "comet":
            model_path = os.path.join(rootdir,"pretrained_augm","comet-atomic_2020_BART_aaai")
            download_files("https://huggingface.co/mismayil/comet-bart-ai2/resolve/main/", model_path)
            augm_mode = Augmentation(model_path, modify_special_token)

        if augmentation == "bart":
            model_path = os.path.join(rootdir,"pretrained_augm","bart_stanford")
            download_files("https://huggingface.co/stanford-oval/paraphraser-bart-large/resolve/main", model_path)
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
        augmentation = augm_mode ,
        kv_store = kv_store if augmentation else None,
        data_dir = data ,
        threshold = float(threshold),
        dir_augmentation = dir_augmentation,
        nli = nli,
        nli1 = None if nli1 == 'None' else float(nli1),
        nli2 = None if nli2 == 'None' else float(nli2),

    )
    model.cuda()

    if pretrained != "":
        model.load_state_dict(torch.load(pretrained))

    reader = QRelsPointwiseReader(
        lazy=True,
        token_indexers={"wordpiece": PretrainedTransformerIndexer("bert-base-uncased", do_lowercase=True)},
        left_tokenizer=BertTokenizer(),
        right_tokenizer=BertTokenizer()
    )
    iterator = BasicIterator(batch_size=batch_size)
    iterator.index_with(vocab)


    print(f"{'Training your augmentation parameters... ' if training_augmentation else 'Creating augmentation to parameters... '} {augmentation_} model , threshold = { float(threshold)} , nli1 = {nli1} nli2 = {nli2} nli = {nli}")

    trainer = Trainer(
        model=model,
        optimizer=Adam(params=model.parameters(), lr=0.00001),
        grad_norm=1.0,
        train_dataset=reader.read(f"{dest_data_dir if training_augmentation  else data}/train"),
        validation_dataset=reader.read(f"{dest_data_dir if training_augmentation  else data}/dev"),
        iterator=iterator,
        validation_metric="+pearson",
        num_epochs=3,
        patience=3,
        serialization_dir=out,
        cuda_device=gpuid
    )

    trainer.train()

    print("\n",f"{'Training your augmentation parameters... ' if training_augmentation else 'Creating augmentation to parameters... '} {augmentation_} model , threshold = { float(threshold)} , nli1 = {nli1} nli2 = {nli2} nli = {nli} was Done")

    if augmentation:
        print("\n",f"The data augmentation saved at {dest_data_dir}")
        # print("\n",f"The amount of data was augmented is { kv_store._get_suffix(os.path.join(data,'train.r') , model.new_key_r) - model.last_org_hyp_key}")


    print("\n",f"The weights saved at {out}")


# output_path_dev = r"/sise/home/orisim/projects/UNLI/results_predictions_dev.csv"
# output_path_train = r"/sise/home/orisim/projects/UNLI/results_predictions_train.csv"
#
# # Save to a CSV file
# df = pd.DataFrame(model.results_predictions_dev)
# df.to_csv(output_path_dev, index=False)
#
# df = pd.DataFrame(model.results_predictions_train)
# df.to_csv(output_path_train, index=False)


