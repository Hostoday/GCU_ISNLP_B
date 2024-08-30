import json
import torch
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, rank_fname, tokenizer, max_length, stride, data_type="all"):
        self.inp = []
        self.rank_idx = []
        self.rank_inp = []
        self.output_list = []

        with open(rank_fname,"r") as rfn:
            rank_dataset = json.load(rfn)
            
        special_tokens = {'additional_special_tokens': ["[speaker1]", "[speaker2]"]}
        tokenizer.add_special_tokens(special_tokens)
        
        def make_chat_with_special_tokens_concat(inp, speaker_map):
            chat = []
            prev_speaker = None
            for cvt in inp['conversation']:
                speaker = speaker_map[cvt['speaker']]
                utterance = cvt['utterance']
                if prev_speaker == speaker:
                    chat[-1] += f"{utterance}"
                else:
                    chat.append(f"{speaker}:{utterance}")
                prev_speaker = speaker
            chat = "".join(chat)

            return chat
        

        for idx, example in enumerate(rank_dataset):
            conversation = example["input"]["conversation"]
            speakers = list(set([cvt['speaker'] for cvt in conversation]))
            
            if len(speakers) != 2:
                raise ValueError("Each conversation must have exactly two speakers.")

            speaker_map = {speakers[0]: "[speaker1]", speakers[1]: "[speaker2]"}
            chat = make_chat_with_special_tokens_concat(example["input"], speaker_map)

            select_data = [example["text"][0]]
            rank_score = [example["score"][0]]
            
            if data_type == "all":
                select_data = example["text"]
                rank_score= example["score"]
            else:
                for idx, da_type in enumerate(example["type"]):
                    if data_type in da_type:
                        select_data.append(example["text"][idx])
                        rank_score.append(example["score"][idx])

            rank_input_id = []
            rank_attention = []
            for rank_data in select_data:
                rank_input = chat + "</s>" + rank_data
                rank_input = tokenizer(rank_input,truncation=True,return_tensors="pt",max_length=1024)

                rank_input_id.append(rank_input["input_ids"])
                rank_attention.append(rank_input["attention_mask"])

            self.output_list.append(select_data)
            self.rank_idx.append(rank_score)
            self.rank_inp.append([rank_input_id,rank_attention])

    def __len__(self):
        return len(self.rank_inp)

    def __getitem__(self, idx):
        return {"input_ids": self.rank_inp[idx], "rank_score": self.rank_idx[idx], "output_list":self.output_list[idx]}

class DataCollatorForSupervisedDataset(object):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, instances):
        input_ids , rank_score = tuple([instance[key] for instance in instances] for key in ("input_ids", "rank_score"))
        return {
            'rank_id': input_ids[0][0],
            'rank_attention': input_ids[0][1],
            'rank_score': rank_score
        }

class DataCollatorForInferenceDataset(object):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, instances):
        input_ids,output_list  = tuple([instance[key] for instance in instances] for key in ("input_ids","output_list"))
        return {
            'rank_id': input_ids[0][0],
            'rank_attention': input_ids[0][1],
            'output_list' : output_list,
        }