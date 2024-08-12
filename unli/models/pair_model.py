from typing import *

import numpy
import numpy as np
import torch
import shutil
import tempfile
import subprocess
import os.path
import pdb
import math

from allennlp.models import Model
from unli.modules import SentencePairFeatureExtractor, CoupledSentencePairFeatureExtractor, \
    DecoupledSentencePairFeatureExtractor
from unli.modules.mlp import *

import unli.modules.loss.hinge
import unli.modules.loss.hinge_ranking
from unli.utils.trec_eval import *
import scipy.stats
from unli.utils.augmentation_commonsense import Augmentation
from unli.data.storage import KeyValueStore
from scripts.util import *

   
class SentencePairModel(Model):

    def __init__(self,
                 extractor: SentencePairFeatureExtractor,
                 mlp: torch.nn.Module,
                 loss_func: torch.nn.Module,
                 mode: str,
                 num_r1_candidates: int = 2,
                 num_r0_candidates: int = 10,
                 mode_weights={},
                 reverse_vocab = None,
                 augmentation : Augmentation = None,
                 kv_store: KeyValueStore = None,
                 data_dir: str = None,
                 threshold: float = 0.8
                 ):

        super(SentencePairModel, self).__init__(vocab=None)
        self.extractor: SentencePairFeatureExtractor = extractor
        self.mlp: torch.nn.Module = mlp
        self.loss_func: torch.nn.Module = loss_func
        self.loss_value: float = float('inf')
        self.test_mode = mode
        self.mode_weights = mode_weights

        self.num_r1_candidates = num_r1_candidates
        self.num_r0_candidates = num_r0_candidates

        self.qrels = tempfile.NamedTemporaryFile(mode="w+", delete=False)
        self.qres = tempfile.NamedTemporaryFile(mode="w+", delete=False)

        self.last_qrels_filename: str = None
        self.last_qres_filename: str = None

        self.results_predictions_dev = []
        self.results_predictions_train = []
        self.epoch = 0
        self.flag = True
        self.threshold = threshold
        self.reverse_vocab = reverse_vocab,
        self.augmentation = augmentation
        self.ds_store = kv_store,
        self.data_dir = data_dir
        if self.ds_store[0] :
            self.new_key_r = self.ds_store[0]._get_highest_suffix(os.path.join(self.data_dir, "train.r"))
        # print(self.new_key_l,"  ", self.new_key_r)
        # print(self.ds_store[0].get_all_keys())
    @classmethod
    def from_params(cls, vocab, params):

        print("from_params of SentencePairModel been called")
        extractor: SentencePairFeatureExtractor = {
            "coupled": lambda: CoupledSentencePairFeatureExtractor.from_params(vocab, params["extractor"]),
            "decoupled": lambda: DecoupledSentencePairFeatureExtractor.from_params(vocab, params["extractor"])
        }[params["extractor_type"]]()

        mlp: MLP = MLP.from_params(params["mlp"])

        loss_func: torch.nn.Module = {
            "pointwise-l2": lambda: torch.nn.MSELoss(),
            "pointwise-log": lambda: torch.nn.BCELoss(),
            "pointwise-log-sigmoid": lambda: torch.nn.BCEWithLogitsLoss(),
            "pointwise-cross-entropy": lambda: torch.nn.CrossEntropyLoss(),
            "pointwise-hinge": lambda: unli.modules.loss.hinge.HingeLoss(params["loss"]["margin"]),
            "pairwise-hinge": lambda: unli.modules.loss.hinge_ranking.PairwiseHingeLoss(params["loss"]["margin"]),
            "listwise-hinge-mean": lambda: unli.modules.loss.hinge_ranking.MeanTripletLoss(params["loss"]["margin"]),
            "listwise-hinge-max": lambda: unli.modules.loss.hinge_ranking.MaxTripletLoss(params["loss"]["margin"])
        }[params["loss_type"]]()

        mode = params["mode"]

        mode_weights = params.get("mode_weights", {})

        model = cls(
            extractor=extractor,
            mlp=mlp,
            loss_func=loss_func,
            mode=mode,
            mode_weights=mode_weights
        )
        return model

    def compute_scores_pointwise(self,
                                 l: torch.Tensor,
                                 r: torch.Tensor
                                 ) -> torch.Tensor:
        """
        Use this model as a pointwise model.
        :param l: LongTensor[Batch, Words]  OR  LongTensor[Batch, Words, Chars=50] (elmo)
        :param r: LongTensor[Batch, Words]  OR  LongTensor[Batch, Words, Chars=50] (elmo)
        :return: y_pred: F[Batch, Output] for classification  OR  F[Batch] for regression
        """
        x = self.extractor(l, r)  # F[Batch, Feature]
        y_pred = self.mlp(x)  # F[Batch, Output]
        if y_pred.size(1) == 1:
            return y_pred.squeeze(dim=1)  # F[Batch]
        else:
            return y_pred  # F[Batch, Output]

    def compute_scores_pairwise(self,
                                l: torch.Tensor,
                                r1: torch.Tensor,
                                r0: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Use this model as a pairwise model.
        :param l: LongTensor[Batch, Words]  OR  LongTensor[Batch, Words, Char=50] (elmo)
        :param r1: LongTensor[Batch, Words]  OR  LongTensor[Batch, Words, Char=50] (elmo)
        :param r0: LongTensor[Batch, Words]  OR  LongTensor[Batch, Words, Char=50] (elmo)
        :return: y1_pred: F[Batch, Output]; y0_pred: F[Batch, Output]
        """
        x1, x0 = self.extractor.forward_2(l, r1, r0)
        y1_pred = self.mlp(x1)  # F[Batch, Output]
        y0_pred = self.mlp(x0)  # F[Batch, Output]
        return y1_pred, y0_pred

    def compute_scores_listwise(self,
                                l: torch.Tensor,
                                rs: torch.Tensor) -> torch.Tensor:
        """
        Use this model as a listwise model.
        :param l: LongTensor[Batch, Words]  OR  LongTensor[Batch, Words, Char=50] (elmo)
        :param rs: LongTensor[Batch, Cand, Words]  OR  LongTensor[Batch, Cand, Words, Char=50] (elmo)
        :return: y_pred: F[Batch, Cand, Output]
        """
        xs = self.extractor.forward_multi_candidate(l, rs)
        ys_pred = self.mlp(xs)
        return ys_pred

    def compute_scores_pair_listwise(self,
                                     l: torch.Tensor,
                                     r1s: torch.Tensor,
                                     r0s: torch.Tensor
                                     ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Use this model as a pair-listwise model.
        :param l: LongTensor[Batch, Words]  OR  LongTensor[Batch, Words, Char=50] (elmo)
        :param r1s: LongTensor[Batch, PosCand, Words]  OR  LongTensor[Batch, PosCand, Words, Char=50] (elmo)
        :param r0s: LongTensor[Batch, NegCand, Words]  OR  LongTensor[Batch, NegCand, Words, Char=50] (elmo)
        :return: y1_pred: F[Batch, PosCand, Output]; y0_pred: F[Batch, NegCand, Output]
        """
        x1s, x0s = self.extractor.forward_multi_candidate_2(l, r1s, r0s)
        y1s_pred = self.mlp(x1s)
        y0s_pred = self.mlp(x0s)
        return y1s_pred, y0s_pred

    def uniform_sample(self, w: torch.Tensor, x: torch.Tensor, sample_size: int) -> torch.Tensor:
        batch_size = w.size(0)
        u = torch.rand(batch_size, sample_size)  # F[Batch, Cand]
        n = (u * (x.float().unsqueeze(dim=1).expand_as(u))).long()  # F[Batch, Cand]

        i = torch.arange(0, batch_size, dtype=torch.int64).unsqueeze(dim=1).expand(batch_size, sample_size)

        if w.ndimension() == 3:
            w_sampled = w[i, n, :]  # F[Batch, Cand, Word]
        elif w.ndimension() == 4:
            w_sampled = w[i, n, :, :]  # F[Batch, Cand, Word, Char]
        else:
            raise Exception("Bad dimensionality in sampled tensor.")
        return w_sampled

    def forward(self, **source) -> Dict[str, torch.Tensor]:

        mode: str = source["mode"][0]
        weight = self.mode_weights.get(mode, 1.0)  # weight on loss w.r.t. to this batch mode

        loss: torch.Tensor = None
        pred_dict: Dict[str, torch.Tensor] = None

        # print(f"mode: {mode}") # to remove ori
        # print( f"self.training: {self.training}")

        if mode.startswith("pointwise"):
            lid: List[str] = source["lid"]
            l: torch.Tensor = source["l"][self.extractor.l_token_index_field]  # F[Batch, Word, ?Char]
            rid: List[str] = source["rid"]
            r: torch.Tensor = source["r"][self.extractor.r_token_index_field]  # F[Batch, Word, ?Char]
            y: torch.Tensor = source["y"]
            y_pred = self.compute_scores_pointwise(l, r)
            pred_dict = {"y_pred": y_pred}

            if self.training:

                loss = self.loss_func(y_pred, y) * weight
                self.loss_value = loss.item()

                self.flag = True
                queries = []
                new_qrels_data = []

                if self.epoch == 2 and self.augmentation:

                    for i in range(0, len(lid)):

                            # input = {
                            #     "epoch": self.epoch ,
                            #     "premises": lid[i],
                            #     "hypothesis": rid[i],
                            #     "y_unli": y[i].item(),
                            #     "y_predicted_unli": y_pred[i].item()
                            # }
                            # self.results_predictions_train.append(input)
                            # print(f"self.reverse_vocab: {self.reverse_vocab}")
                            # print(f"reverse_vocab.get_index_to_token_vocabulary():")
                            # print(self.reverse_vocab.get_index_to_token_vocabulary(namespace="tags"))
                            # print(f"reverse_vocab.get_vocab_size():" )
                            # print(self.reverse_vocab.get_vocab_size(namespace="tags"))
                            # decoded_tokens = [self.reverse_vocab[id].replace("#", "") for id in l[i].cpu().numpy() if id not in [101,102,]]
                            # decoded_sentence = " ".join(decoded_tokens)
                            # print(decoded_sentence)
                            # input["p_text"] =

                            if abs(y[i].item() - y_pred[i].item()) > self.threshold and ( y[i].item() <=0.15 ):  #y[i].item() >= 0.84 y[i].item() <=0.15 or

                                key_l = lid[i]
                                key_r = rid[i]
                                value_r = self.ds_store[0].get_value(os.path.join(self.data_dir,"train.r"),key_r)
                                value_l = self.ds_store[0].get_value(os.path.join(self.data_dir,"train.l"),key_l)
                                queries.append({"l": value_l, "r":value_r, "y":y[i].item() , "key_l": key_l })

                    if len(queries) > 0:

                        augmn_data = self.augmentation.augmentation_commonsense_data(queries)
                        # print("augmn_data is soze" , len(augmn_data))
                        for i,row in enumerate(augmn_data):

                            if len(row) > 0:

                                for  r in row:

                                    # self.new_key_l = lid[i] # self.ds_store[0].get_the_next_id(self.new_key_l,'l') #.add_new_entry(os.path.join(self.data_dir,"train.l"), queries[i]["l"])
                                    self.new_key_r = self.ds_store[0].get_the_next_id(self.new_key_r,'r') #.add_new_entry(os.path.join(self.data_dir,"train.r"), r)
                                    new_qrels_data.append({"key_l": queries[i]["key_l"] ,"new_key_r" : self.new_key_r, "y": queries[i]["y"] , "new_r": r })

                    if len(new_qrels_data) > 0:
                        add_row_to_qrels(os.path.join(self.data_dir,'augmentation','train.qrels')  , new_qrels_data )
                        # add_row_to_l(os.path.join(self.data_dir,'augmentation','train.l'), new_qrels_data)
                        add_row_to_r(os.path.join(self.data_dir,'augmentation','train.r'), new_qrels_data)

                return {"loss": loss, **pred_dict}

            else:  # dev

                print(f"epoch: {self.epoch}")
                for i in range(0, len(lid)):
                    if mode == "pointwise-regression":
                        print(TrecEvalRefItem(lid[i], rid[i], y[i].item()), file=self.qrels)
                        print(TrecEvalResItem(lid[i], rid[i], 0, y_pred[i].item()), file=self.qres)

                        if self.flag:
                            self.epoch +=1
                            self.flag = False

                        self.results_predictions_dev.append({
                                    "epoch": self.epoch-1,
                                    "premises": lid[i],
                                    "hypothesis": rid[i],
                                    "y_unli": y[i].item(),
                                    "y_predicted_unli": y_pred[i].item()
                                })

                    elif mode == "pointwise-classification":
                        y_pred_cls = y_pred.argmax(dim=1)
                        print(TrecEvalRefItem(lid[i], rid[i], y[i].item()), file=self.qrels)
                        print(TrecEvalResItem(lid[i], rid[i], 0, y_pred_cls[i].item()), file=self.qres)
                return pred_dict

        elif mode.startswith("pairwise"):
            lid: List[str] = source["lid"]
            l: torch.Tensor = source["l"][self.extractor.l_token_index_field]  # F[Batch, Word, ?Char]
            r1id: List[str] = source["r1id"]
            r0id: List[str] = source["r0id"]
            r1: torch.Tensor = source["r1"][self.extractor.r_token_index_field]  # F[Batch, Word, ?Char]
            r0: torch.Tensor = source["r0"][self.extractor.r_token_index_field]  # F[Batch, Word, ?Char]
            y1: torch.Tensor = source["y1"]
            y0: torch.Tensor = source["y0"]
            y1_pred, y0_pred = self.compute_scores_pairwise(l, r1, r0)
            pred_dict = {"y1_pred": y1_pred, "y0_pred": y0_pred}
            if self.training:
                loss = self.loss_func(y1_pred, y0_pred, y1, y0) * weight
                self.loss_value = loss.item()
                return {"loss": loss, **pred_dict}
            else:  # dev
                raise Exception("Pairwise mode should not be used in dev")

        elif mode.startswith("cross-pairwise"):
            l1id: List[str] = source["l1id"]
            l0id: List[str] = source["l0id"]
            r1id: List[str] = source["r1id"]
            r0id: List[str] = source["r0id"]
            l1: torch.Tensor = source["l1"][self.extractor.l_token_index_field]
            l0: torch.Tensor = source["l0"][self.extractor.l_token_index_field]
            r1: torch.Tensor = source["r1"][self.extractor.r_token_index_field]
            r0: torch.Tensor = source["r0"][self.extractor.r_token_index_field]
            y1: torch.Tensor = source["y1"]
            y0: torch.Tensor = source["y0"]
            y1_pred = self.compute_scores_pointwise(l1, r1)
            y0_pred = self.compute_scores_pointwise(l0, r0)
            pred_dict = {"y1_pred": y1_pred, "y0_pred": y0_pred}
            if self.training:
                loss = self.loss_func(y1_pred, y0_pred, y1, y0) * weight
                self.loss_value = loss.item()
                return {"loss": loss, **pred_dict}
            else:  # dev
                raise Exception("Cross-pairwise mode should not be used in dev")

        elif mode.startswith("listwise"):
            lid: List[str] = source["lid"]
            l: torch.Tensor = source["l"][self.extractor.l_token_index_field]  # F[Batch, Word, ?Char]
            rids: List[List[str]] = source["rids"]
            rs: torch.Tensor = source["rs"][self.extractor.r_token_index_field]  # F[Batch, Cand, Word, ?Char]
            ys: torch.Tensor = source["ys"]
            ys_pred = self.compute_scores_listwise(l, rs)
            pred_dict = {"ys_pred": ys_pred, "loss": loss}
            if self.training:
                loss = self.loss_func(ys_pred, ys) * weight
                self.loss_value = loss.item()
                return {"loss": loss, **pred_dict}
            else:
                for i in range(0, len(lid)):
                    for j in range(0, len(rids[i])):
                        if rids[i][j] is not None:  # sometimes it is padded with `None`s
                            print(TrecEvalRefItem(lid[i], rids[i][j], ys[i, j].item()), file=self.qrels)
                            print(TrecEvalResItem(lid[i], rids[i][j], 0, ys_pred[i, j].item()), file=self.qres)

        elif mode.startswith("pair-listwise"):
            lid: List[str] = source["lid"]
            l: torch.Tensor = source["l"][self.extractor.l_token_index_field]  # F[Batch, Word, ?Char]
            r1ids: List[List[str]] = source["r1ids"]
            r0ids: List[List[str]] = source["r0ids"]
            r1s: torch.Tensor = source["r1s"][self.extractor.r_token_index_field]  # F[Batch, PosCand, Word, ?Char]
            r0s: torch.Tensor = source["r0s"][self.extractor.r_token_index_field]  # F[Batch, NegCand, Word, ?Char]
            if self.training:
                num_pos_cands = torch.tensor([len([x for x in r1id if x is not None]) for r1id in r1ids])  # L[Batch]
                r1s = self.uniform_sample(r1s, num_pos_cands, self.num_r1_candidates)
                num_neg_cands = torch.tensor([len([x for x in r0id if x is not None]) for r0id in r0ids])  # L[Batch]
                r0s = self.uniform_sample(r0s, num_neg_cands, self.num_r0_candidates)

            y1s_pred, y0s_pred = self.compute_scores_pair_listwise(l, r1s, r0s)
            pred_dict = {"y1s_pred": y1s_pred, "y0s_pred": y0s_pred}
            if self.training:
                loss = self.loss_func(y1s_pred, y0s_pred) * weight
                self.loss_value = loss.item()
                return {"loss": loss, **pred_dict}
            else:
                raise Exception("Pairwise mode should not be used in dev")

        return pred_dict

    def forward_on_instance(self, instance) -> Dict[str, numpy.ndarray]:
        pass

    def dev_callback(self, epoch: int, serialization_dir: str):
        print(f"Epoch: {epoch}")
        print(f"serialization_dir/epoch-{epoch}-dev.qrels:  {serialization_dir}/epoch-{epoch}-dev.qrels")
        self.epoch = epoch
        shutil.copyfile(self.last_qrels_filename, f"{serialization_dir}/epoch-{epoch}-dev.qrels")
        shutil.copyfile(self.last_qres_filename, f"{serialization_dir}/epoch-{epoch}-dev.qres")

    def test_callback(self, serialization_dir: str):
        shutil.copyfile(self.last_qrels_filename, f"{serialization_dir}/test.qrels")
        shutil.copyfile(self.last_qres_filename, f"{serialization_dir}/test.qres")

    def get_metrics(self, reset: bool = False, mimick_test: bool = False) -> Dict[str, float]:

        loss_dict = {"loss": self.loss_value}
        if self.training:
            return loss_dict

        print(self.test_mode) # to remove ori
        if reset:
            self.qrels.flush()
            self.qres.flush()
            additional_metrics = {}

            if self.test_mode == "ranking":
                raw_trec_eval_output = subprocess.check_output([
                    "trec_eval",  # call external `trec_eval` process
                    self.qrels.name,
                    self.qres.name
                ]).decode()
                trec_eval_lines = raw_trec_eval_output.split("\n")[1:]  # remove TSV header row

                for l in trec_eval_lines:
                    tokens = l.split()
                    if len(tokens) == 3:  # the format of trec_eval output
                        k = tokens[0]
                        v = float(tokens[2])
                        additional_metrics[k] = v

                self.qrels.close()
                self.qres.close()

            elif self.test_mode == "regression":
                self.qrels.close()
                self.qres.close()
                ys = []
                ys_pred = []

                print(f"self.qrels.name {self.qrels.name}")
                print(f"self.qres.name {self.qres.name}")

                with open(self.qrels.name, mode='r') as qrels_f, open(self.qres.name, mode='r') as qres_f:
                    for l in qrels_f:
                        _, _, _, y = l.strip().split('\t')
                        ys.append(float(y))
                    for l in qres_f:
                        _, _, _, _, y, _ = l.strip().split('\t')
                        ys_pred.append(float(y))
                additional_metrics["pearson"] = scipy.stats.pearsonr(ys_pred, ys)[0]
                additional_metrics["spearman"] = scipy.stats.spearmanr(ys_pred, ys).correlation
                additional_metrics["mse"] = ((np.array(ys_pred) - np.array(ys)) ** 2).mean()

            elif self.test_mode == "classification":
                self.qrels.close()
                self.qres.close()
                ys = []
                ys_pred = []
                with open(self.qrels.name, mode='r') as qrels_f, open(self.qres.name, mode='r') as qres_f:
                    for l in qrels_f:
                        _, _, _, y = l.strip().split('\t')
                        ys.append(float(y))
                    for l in qres_f:
                        _, _, _, _, y, _ = l.strip().split('\t')
                        ys_pred.append(float(y))
                additional_metrics["accuracy"] = float(
                    sum(1 if y == y_pred else 0 for y, y_pred in zip(ys, ys_pred))) / float(len(ys))

            self.last_qrels_filename = self.qrels.name
            self.last_qres_filename = self.qres.name
            self.qrels = tempfile.NamedTemporaryFile(mode="w+", delete=False)
            self.qres = tempfile.NamedTemporaryFile(mode="w+", delete=False)

            return {**loss_dict, **additional_metrics}
        else:
            return loss_dict
