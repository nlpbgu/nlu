
from unli.data.storage import KeyValueStore
import os
from unli.commonsense.comet_atomic2020 import Comet
from collections import defaultdict
import re

class Augmentation():

        def __init__(self,model_path: str = None , modify_special_token = True ):

                print("model loading ...")
                self.augmet_model = Comet(model_path, modify_special_token)
                self.augmet_model.model.zero_grad()
                print("model loaded")


        def augmentation_commonsense_data(self, queries):

                queries_mod = []

                for query in queries:
                        query = f"Premise: {query['l']} Hypothesis: {query['r']} Paraphrase a Commonsense Hypothesis:"
                        queries_mod.append(query)


                results = self.augmet_model.generate(queries_mod, decode_method="beam", num_generate=5)

                results = process_sentences(results,  specific_sentences_to_remove=queries)
                # print("results after is size",len(results))

                return results



def process_sentences(list_of_lists, filter_words=('Generate', 'Explain'), specific_sentences_to_remove=(),
                      min_words=5):

        def tokenize(sentence):
                # Simple tokenization using regex to split on non-word characters
                return re.findall(r'\w+', sentence.lower())

        def has_high_overlap(tokens1, tokens2, threshold=0.75):
                # Calculate the overlap ratio between two sets of tokens
                set1, set2 = set(tokens1), set(tokens2)
                intersection = set1.intersection(set2)
                if len(set1) == 0 or len(set2) == 0:
                        return False
                return len(intersection) / max(len(set1), len(set2)) >= threshold # min

        processed_lists = []

        for inx,sentence_list in enumerate(list_of_lists):

                tokenized_sentences = [(sentence, tokenize(sentence)) for sentence in sentence_list]
                unique_sentences = []

                for i, (sentence, tokens) in enumerate(tokenized_sentences):

                        # Filter based on starting words, specific sentences, and minimum word count
                        if ( len(tokens) == 0 or any( word.lower() == tokens[0] for word in filter_words) or
                                len(tokens) < min_words or
                                sentence in [specific_sentences_to_remove[inx]]):
                                continue

                        # Check if this sentence has high overlap with any already added sentences
                        is_duplicate = False
                        for unique_sentence, unique_tokens in unique_sentences:
                                if has_high_overlap(tokens, unique_tokens):
                                        is_duplicate = True
                                        break

                        if not is_duplicate:
                                unique_sentences.append((sentence, tokens))

                # Replace "PersonX" and "PersonY" with "Person" and store results
                processed_sentences = [
                        sentence.replace("PersonX", "Person").replace("PersonY", "Person")
                        for sentence, _ in unique_sentences
                ]

                processed_lists.append(processed_sentences)

        return processed_lists


class BartAugmentation(Augmentation):

        def __init__(self, model_path: str = None, modify_special_token=True):
                super(BartAugmentation, self).__init__(model_path=model_path,modify_special_token=modify_special_token)


        def augmentation_commonsense_data(self, queries):
                queries_mod = []

                for query in queries:
                        query = f"Premise: {query['l']} Hypothesis: {query['r']} Paraphrase a Commonsense Hypothesis:"
                        queries_mod.append(query)


                results = self.augmet_model.generate(queries_mod, decode_method="beam", num_generate=5)


                results_bart_f =[]
                for result in results:

                        results_bart = []
                        for r in result:
                                r = self.extract_after_second_hypothesis(r)
                                if r:
                                        results_bart.append(r)

                        results_bart_f.append(results_bart)

                results = process_sentences(results_bart_f, specific_sentences_to_remove=queries)


                return results

        def extract_after_second_hypothesis(self,result):

                # Split the sentence by "Hypothesis:" and take everything after the second occurrence
                parts = result.split("hypothesis")
                if len(parts) > 1:
                        if len(parts[1].strip()) > 1:
                                return parts[1].strip()[1:]
                        return None
                return None  # Return None if there aren't two occurrences