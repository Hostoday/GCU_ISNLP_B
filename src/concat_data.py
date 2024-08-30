import json
import torch
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, fname,concat_file_list, tokenizer, prompt, model_type , mode='summary_concat'):
        IGNORE_INDEX = -100
        self.inp = []
        self.speaker_mappings = []
        
        PROMPT = prompt


        with open(fname, "r") as f:
            data = json.load(f)

        output_list = [[] for _ in range(408)]
        for output_file in concat_file_list:
            with open(output_file,"r") as f:
                output = json.load(f)
            for i in range(len(output[:])):
                output_list[i].append(output[i]["output"])
            
        special_tokens = {'additional_special_tokens': ["[speaker1]", "[speaker2]"]}
        tokenizer.add_special_tokens(special_tokens)

        def make_conversation(inp,speaker_map):
            subject_keywords = ', '.join(inp['subject_keyword'])
            conversation_summary = f"[대화 분석]\n주제: {subject_keywords}\n"
            
            # 대화를 순서대로 나열
            for cvt in inp['conversation']:
                speaker = speaker_map[cvt['speaker']]
                utterance = cvt['utterance']
                conversation_summary += f"{speaker}: {utterance}\n"

            return conversation_summary

        def summary_concat(output):
            summary = '[Summary]\n'+ '\n'.join(output)
            question = f"[Request]\n위 요약문들을 정보를 보존한 채로 하나로 합쳐주세요."
            input = summary + "\n" + question
            return input
        
        def summary_concat_mo(output):
            summary = '[Summary]\n'+ '\n'.join(output)
            question = f"[Request]\n위 요약문들을 읽고 하나의 요약문으로 정리해주세요. 요약문의 정보를 정리해서 하나의 요약문으로 만드는 것이 목표입니다."
            input = summary + "\n" + question
            return input
        
        def summary_concat_tg(output):
            summary = '[Summary]\n'+ '\n'.join(output)
            question = f"[Request]\n위 요약문들을 읽고 하나의 요약문으로 정리해주세요. 요약문의 정보를 정리해서 하나의 요약문으로 만드는 것이 목표입니다. 주어진 데이터들에서 공통된 내용을 중심으로 요약해주세요."
            input = summary + "\n" + question
            return input

        def summary_concat_2(inp,output):
            summary = f"[문제제기]\n 아래 요약문들은 {', '.join(inp['subject_keyword'])}에 대한 요약문들이다\n" + '\n'.join(output)
            question = f"[결론 도출하기]\n위 요약문들을 정보를 보존한 채로 하나로 합쳐주세요."
            input = summary + "\n" + question
            return input
        
        def summary_concat_3(output):
            summary = '[Summary]\n'+ '\n'.join(output)
            question = f"[Request]\n위 요약문들을 중복되지 않는 정보를 보존하면서 하나로 합쳐주세요."
            input = summary + "\n" + question
            return input
        
        def summary_concat_4(output):
            summary = '[문제 파악하기] 한 대화에 대해서 요약한 요약문들입니다.\n'+ '\n'.join(output) + "\n[정보 파악하기] 각 요약문들의 정보를 파악합니다.\n"
            question = f"[결론 도출하기]\n 이 모든 정보를 바탕으로 중복되지 않는 정보를 보존하면서 주어진 요약문들을 기반으로 하나로 합쳐주세요."
            input = summary + "\n" + question
            return input
        
        def summary_concat_5(inp, output, speaker_map):
            conversation = make_conversation(inp,speaker_map)
            candidates = "[후보1] "
            for idx, summary in enumerate(output):
                candidates += summary
                if idx == 1: candidates += "[후보2] "

            summary = '[문제 파악하기] 한 대화에 대해서 요약한 요약문들이고 \n'+ f"이 대화는 {', '.join(inp['subject_keyword'])}에 대한 것입니다.\n" + conversation + "\n [후보1]은 Bleurt score가 높은 요약문이지만 Rouge score가 낮은 요약문이고, [후보2]는 Rouge score가 높지만 Bleurt score가 낮은 요약문입니다. 당신은 두 가지 score가 높게 나오도록 요약문을 조합해야 합니다." + candidates + "\n[정보 파악하기] 각 요약문들의 정보를 파악합니다.\n"
            question = f"[결론 도출하기]\n 이 모든 정보를 바탕으로 중복되지 않는 정보를 보존하면서 주어진 요약문들을 기반으로 하나로 합쳐주세요."
            input = summary + "\n" + question
            return input
        
        def summary_concat_conversation(inp,output,speaker_map):
            conversation = make_conversation(inp,speaker_map)
            summary = f"[문제 파악하기]\n" + f"이 대화는 {', '.join(inp['subject_keyword'])}에 대한 것입니다.\n" + conversation
            summary = "[요약문 정보 파악하기]\n 위 대화를 기반으로 요약한 요약문들입니다."+'\n'.join(output)
            question = f"[결론 도출하기]\n위 요약문들을 정보를 보존한 채로 하나의 요약문으로 출력해주세요."
            input = summary + "\n" + question
            return input
        
        for idx, example in enumerate(data):
            conversation = example["input"]["conversation"]
            speakers = list(set([cvt['speaker'] for cvt in conversation]))

            if len(speakers) != 2:
                raise ValueError("Each conversation must have exactly two speakers.")

            speaker_map = {speakers[0]: "[speaker1]", speakers[1]: "[speaker2]"}

            if mode == 'summary_concat':
                message = summary_concat(output_list[idx])
            elif mode == 'summary_concat_2':
                message = summary_concat_2(example["input"], output_list[idx])
            elif mode == 'summary_concat_3':
                message = summary_concat_3(output_list[idx])
            elif mode == 'summary_concat_4':
                message = summary_concat_4(output_list[idx])
            elif mode == 'summary_concat_5':
                message = summary_concat_5(example["input"], output_list[idx], speaker_map)
            elif mode == 'summary_concat_mo':
                message = summary_concat_mo(output_list[idx])
            elif mode == 'summary_concat_tg':
                message = summary_concat_tg(output_list[idx])
            elif mode == 'summary_concat_conversation':
                message = summary_concat_conversation(example["input"], output_list[idx], speaker_map)

            if len(speakers) != 2:
                raise ValueError("Each conversation must have exactly two speakers.")

            speaker_map = {speakers[0]: "[speaker1]", speakers[1]: "[speaker2]"}
           
            # print(chat)
            if model_type == "gemma":
                message = [
                    {"role": "user", "content": message},
                    {"role": "assistant", "content": PROMPT}
                ]
            elif model_type == "default":
                message = [
                    {"role": "system", "content": PROMPT},
                    {"role": "user", "content": message}
                ]
            # elif model_type == "mistral":
            #     print("1")
            #     input()
            #     message = [f"[INST] {PROMPT}{chat}[/INST]"]
            
            
            if PROMPT == "None":
                message=[{"role": "user", "content": chat}]
                
                
            source = source = tokenizer.apply_chat_template(
                message,
                add_generation_prompt=True,
                return_tensors="pt",
            )

            input_ids = source[0]

            # target = example["output"]
            # if target != "":
            #     target += tokenizer.eos_token

            # target = process_target_func(target, speaker_map if mode != 'mode_original' else tokenizer, tokenizer)
            # target["input_ids"] = target["input_ids"].type(torch.int64)

            # input_ids = torch.concat((source[0], target["input_ids"][0]))
            # labels = torch.concat((torch.LongTensor([IGNORE_INDEX] * source[0].shape[0]), target["input_ids"][0]))
            self.inp.append(input_ids)
            self.speaker_mappings.append(speaker_map)

    def __len__(self):
        return len(self.inp)

    def __getitem__(self, idx):
        return {"input_ids": self.inp[idx],"speaker_map": self.speaker_mappings[idx]}

# class DataCollatorForSupervisedDataset(object):
#     def __init__(self, tokenizer):
#         self.tokenizer = tokenizer

#     def __call__(self, instances):
#         input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
#         input_ids = torch.nn.utils.rnn.pad_sequence(
#             [torch.tensor(ids) for ids in input_ids], batch_first=True, padding_value=self.tokenizer.pad_token_id
#         )
#         labels = torch.nn.utils.rnn.pad_sequence([torch.tensor(lbls) for lbls in labels], batch_first=True, padding_value=-100)
#         return {
#             'input_ids':input_ids,
#             'labels':labels,
#             'attention_mask':input_ids.ne(self.tokenizer.pad_token_id),
#         }

class DataCollatorForInferenceDataset(object):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, instances):
        input_ids,speaker_maps =  tuple([instance[key] for instance in instances] for key in ("input_ids", "speaker_map"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        return {
            'input_ids': input_ids,
            'attention_mask': input_ids.ne(self.tokenizer.pad_token_id),
            'speaker_maps': speaker_maps
        }