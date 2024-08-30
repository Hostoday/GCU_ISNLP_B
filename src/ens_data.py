import json
import torch
from torch.utils.data import Dataset

from nltk.translate.bleu_score import corpus_bleu, sentence_bleu
from konlpy.tag import Mecab
from rouge import Rouge
from bert_score import score as bert_score_func

class CustomDataset(Dataset):
    def __init__(self, concat_file_list):
        IGNORE_INDEX = -100        
        self.output_list = [[] for _ in range(408)]
        for output_file in concat_file_list:
            with open(output_file,"r") as f:
                output = json.load(f)
            for i in range(len(output[:])):
                self.output_list[i].append(output[i]["output"])
        

    def __len__(self):
        return len(self.output_list)

    def __getitem__(self, idx):
        return {"input_ids": self.output_list[idx]}

class DataCollatorForInferenceDataset(object):
    def __init__(self):
        None
    def __call__(self, instances):
        input_ids = tuple([instance["input_ids"] for instance in instances])
        return {
            'input_ids':input_ids,
        }

#class DataCollatorForInferenceDataset(object):
#    def __init__(self, tokenizer):
#        self.tokenizer = tokenizer
#
#    def __call__(self, instances):
#        input_ids, labels, speaker_maps = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels", "speaker_map"))
#        input_ids = torch.nn.utils.rnn.pad_sequence(
#            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
#        )
#        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
#        return {
#            'input_ids': input_ids,
#            'labels': labels,
#            'attention_mask': input_ids.ne(self.tokenizer.pad_token_id),
#            'speaker_maps': speaker_maps
#        }