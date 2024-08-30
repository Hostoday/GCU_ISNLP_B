import json
from datasets import load_dataset

from torch.utils.data import Dataset

import json
import torch
from torch.utils.data import Dataset

class moCustomDataset(Dataset):
    def __init__(self, fname, file_list,tokenizer, prompt, model_type ,mode='moonmoon'):
        IGNORE_INDEX = -100
        self.inp = []
        self.label = []
        self.speaker_mappings = []
        
        PROMPT = prompt

        with open(fname, "r") as f:
            data = json.load(f)


        output_list = [[] for _ in range(408)]
        for output_file in file_list:
            with open(output_file,"r") as f:
                output = json.load(f)
            for i in range(len(output[:])):
                output_list[i].append("[요약문]" +output[i]["output"])
            
        special_tokens = {'additional_special_tokens': ["[speaker1]", "[speaker2]"]}
        tokenizer.add_special_tokens(special_tokens)


        def make_chat(inp, speaker_map, output):
          conversation = make_conversation(inp,speaker_map)
          
          question = f"\n[Question] 위 요약문들을 읽고 아래 {', '.join(inp['subject_keyword'])} 에 관한 대화를 제일 잘 요약한 요약문을 골라주세요\n 요약문과 대화내용의 유사도가 가장 높은 요약문을 고르는 것입니다.\n 요약문들중 대화의 주요 내용과 전반적인 맥락의 분위기를 잘 반영한 요약문을 골라주세요.\n ['주의사항'][speaker1]"

          summary = f"" + '\n'.join(output)

          chat = summary + "\n" + question + "\n" + conversation

          return chat
        
        def make_conversation(inp,speaker_map):
            subject_keywords = ', '.join(inp['subject_keyword'])
            conversation_summary = ""

            # 대화를 순서대로 나열
            for cvt in inp['conversation']:
                speaker = speaker_map[cvt['speaker']]
                utterance = cvt['utterance']
                conversation_summary += f"{speaker}: {utterance}\n"

            return conversation_summary

        def process_target_with_special_tokens(target, speaker_map, tokenizer):
            for speaker in speaker_map:
                target = target.replace(speaker, speaker_map[speaker])
            return tokenizer(target,
                             return_attention_mask=False,
                             add_special_tokens=False,
                             return_tensors="pt")

        def process_target_original(target, tokenizer):
            return tokenizer(target,
                             return_attention_mask=False,
                             add_special_tokens=False,
                             return_tensors="pt")
        
        for idx, example in enumerate(data):
            conversation = example["input"]["conversation"]
            speakers = list(set([cvt['speaker'] for cvt in conversation]))
            speaker_map = {speakers[0]: "[speaker1]", speakers[1]: "[speaker2]"}

            chat = make_chat(example["input"],speaker_map, output_list[idx])
            # print(chat)
            if model_type == "gemma":
                message = [
                    {"role": "user", "content": chat},
                    {"role": "assistant", "content": PROMPT}
                ]
            elif model_type == "default":
                message = [
                    {"role": "system", "content": PROMPT},
                    {"role": "user", "content": chat}
                ]
            # elif model_type == "mistral":
            #     print("1")
            #     input()
            #     message = [f"[INST] {PROMPT}{chat}[/INST]"]
            
            
            if PROMPT == "None":
                message=[{"role": "user", "content": chat}]
                
                
            source = tokenizer.apply_chat_template(
                message,
                add_generation_prompt=True,
                return_tensors="pt",
            )

            target = example["output"]
            if target != "":
                target += tokenizer.eos_token

            target = process_target_with_special_tokens(target, speaker_map if mode != 'mode_original' else tokenizer, tokenizer)
            target["input_ids"] = target["input_ids"].type(torch.int64)

            input_ids = torch.concat((source[0], target["input_ids"][0]))
            labels = torch.concat((torch.LongTensor([IGNORE_INDEX] * source[0].shape[0]), target["input_ids"][0]))
            self.inp.append(input_ids)
            self.label.append(labels)
            self.speaker_mappings.append(speaker_map)

    def __len__(self):
        return len(self.inp)

    def __getitem__(self, idx):
        return {"input_ids": self.inp[idx], "labels": self.label[idx], "speaker_map": self.speaker_mappings[idx]}



class Yo_dataset(Dataset):
    def __init__(self, fname, train_fname, tokenizer, prompt, model_type , mode='mode_with_special_tokens_concat'):
        IGNORE_INDEX = -100
        self.inp = []
        self.label = []
        self.speaker_mappings = []
        
        PROMPT = prompt


        with open(fname, "r") as f:
            data = json.load(f)
            
        with open(train_fname, "r") as f:
            train_data = json.load(f)

        special_tokens = {'additional_special_tokens': ["[speaker1]", "[speaker2]"]}
        tokenizer.add_special_tokens(special_tokens)

        def chat_moon_fewshot(inp, speaker_map, train_data):
            conversation, keyword_list = make_conversation(inp,speaker_map)

            matching_inputs = find_first_matching_input_with_output(keyword_list, train_data)

            question0 = "위 문장은 conversation 요약을 어떻게 하는지 보여주는 예시 입니다. 요약은 conversation의 문장 정보를 담고있어야 하며 의미적으로 유사해야 합니다.\n [conversation]안의 대화를 읽고 대화의 요약을 [output]에 작성한 것입니다."

            question1 = f"아래에 있는 두 speaker의 대화를 읽고. 주제 {', '.join(inp['subject_keyword'])} 에 집중하여 화자별로 정확하고 4줄이상으로 요약해주세요"


            if matching_inputs == None:
              output = ""
            else:
              output = '[output]' + matching_inputs['output']

            if matching_inputs == None:
              chat = ""
            else:
              chat = "[conversation]" + "\n"
              for cvt in matching_inputs['input']['conversation']:
                speaker = cvt['speaker']
                utterance = cvt['utterance']
                chat += (f"{speaker}: {utterance}") + " "
                
            
            yoyo = chat + "\n" + output + "\n" + question0 + question1 + "\n" +  conversation
    

            return yoyo
        def find_first_matching_input_with_output(test_keywords, train_data):
              for train_item in train_data:
                  train_keywords = train_item['input']['subject_keyword']                  
                  if any(keyword in train_keywords for keyword in test_keywords):                  
                      return {
                          'input': train_item['input'],
                          'output': train_item['output']
                      }
              if train_data:
                  first_item = train_data[0]
                  return {
                      'input': first_item['input'],
                      'output': first_item['output']
                  }
              return None
        
        def chat_moon(inp, speaker_map):#make question 
            chat = ["[Conversation]"]
            prev_speaker = None
            for cvt in inp['conversation']:
                speaker = speaker_map[cvt['speaker']]
                utterance = cvt['utterance']
                if prev_speaker == speaker:
                    chat[-1] += f" {utterance}"
                else:
                    chat.append(f"{speaker}: {utterance}")
                prev_speaker = speaker
            chat = "\n".join(chat)

            question = f"[Question]\n아래 대화를 읽고 화자별로 자세히 요약해주세요.\n 중요한 대화 토픽은{', '.join(inp['subject_keyword'])} 입니다.\n 집중해서 읽고 토픽에 기반한 대화 내용을 한국어로 요약해주세요."
            
            chat = chat + "\n\n" + question

            return chat
        
        def chat_moon_v2(inp, speaker_map):
            conversation = make_conversation(inp,speaker_map)
            
            question = f"[Question]\n아래 대화를 읽고 화자별로 자세히 요약해주세요.\n 중요한 대화 주제는 '{', '.join(inp['subject_keyword'])}' 입니다.\n 1.[speaker1] 이 말한 내용만 마지막까지 집중해서 요약해주세요. \n 2.[speaker2] 가 말한 내용만 마지막까지 집중해서 요약해주세요."

            chat = question + conversation

            return chat
        
        def chat_moon_en(inp, speaker_map):
            conversation = make_conversation(inp,speaker_map)
        
            question = f"[Question]\nRead the following conversation and provide a detailed summary for each speaker.\n The main topic of the conversation is '{', '.join(inp['subject_keyword'])}'.\n Summarize the conversation in Korean, focusing on the main points and opinions expressed by each speaker.\n Ensure the summary maintains the sentence structure and vocabulary choices similar to the original text. Use appropriate expressions to keep the summary natural and logical.\n You must provide the summary in Korean."

            chat = question + conversation

            return chat
        

        
        
        def make_chat_with_special_tokens_concat(inp, speaker_map):
            chat = ["[Conversation]"]
            prev_speaker = None
            for cvt in inp['conversation']:
                speaker = speaker_map[cvt['speaker']]
                utterance = cvt['utterance']
                if prev_speaker == speaker:
                    chat[-1] += f" {utterance}"
                else:
                    chat.append(f"{speaker}: {utterance}")
                prev_speaker = speaker
            chat = "\n".join(chat)

            question = f"[Question]\n위 {', '.join(inp['subject_keyword'])} 주제에 대한 대화를 화자별로 자세히 요약해주세요."
            chat = chat + "\n\n" + question

            return chat
        
        def make_chat_with_special_tokens(inp, speaker_map):
            chat = ["[Conversation]"]
            for cvt in inp['conversation']:
                speaker = speaker_map[cvt['speaker']]
                utterance = cvt['utterance']
                chat.append(f"{speaker}: {utterance}")
                prev_speaker = speaker
            chat = "\n".join(chat)

            question = f"[Question]\n위 {', '.join(inp['subject_keyword'])} 주제에 대한 대화를 화자별로 자세히 요약해주세요."
            chat = chat + "\n\n" + question

            return chat
        
        
        def make_chat_with_special_tokens_topic_intro(inp, speaker_map):
            topic_intro = f"이 대화는 {', '.join(inp['subject_keyword'])}에 대한 것입니다.\n"
            chat = [topic_intro + "[Conversation]"]
            prev_speaker = None
            for cvt in inp['conversation']:
                speaker = speaker_map[cvt['speaker']]
                utterance = cvt['utterance']
                if prev_speaker == speaker:
                    chat[-1] += f" {utterance}"
                else:
                    chat.append(f"{speaker} : {utterance}")
                prev_speaker = speaker
            chat = "\n".join(chat)

            question = f"[Question]\n위 {', '.join(inp['subject_keyword'])} 주제에 대한 대화를 화자별로 자세히 요약해주세요."
            chat = chat + "\n\n" + question

            return chat
        
        def make_chat_with_special_tokens_keywordinprompt(inp, speaker_map):
            topic_intro = f"이 대화는 {', '.join(inp['subject_keyword'])}에 대한 것입니다.\n"
            chat = [topic_intro + "[Conversation]"]
            prev_speaker = None
            for cvt in inp['conversation']:
                speaker = speaker_map[cvt['speaker']]
                utterance = cvt['utterance']
                if prev_speaker == speaker:
                    chat[-1] += f" {utterance}"
                else:
                    chat.append(f"{speaker} : {utterance}")
                prev_speaker = speaker
            chat = "\n".join(chat)

            question = f"[Question]\n위 대화를 키워드인 {', '.join(inp['subject_keyword'])}에 대해서 화자별로 상세하게 요약해주세요."
            chat = chat + "\n\n" + question

            return chat
        
        def make_chat_with_special_tokens_keywordinprompt_nointro(inp, speaker_map):
            topic_intro = f"이 대화는 {', '.join(inp['subject_keyword'])}에 대한 것입니다.\n"
            chat = [topic_intro + "[Conversation]"]
            prev_speaker = None
            for cvt in inp['conversation']:
                speaker = speaker_map[cvt['speaker']]
                utterance = cvt['utterance']
                if prev_speaker == speaker:
                    chat[-1] += f" {utterance}"
                else:
                    chat.append(f"{speaker} : {utterance}")
                prev_speaker = speaker
            chat = "\n".join(chat)

            question = f"[Question]\n위 대화를 키워드인 {', '.join(inp['subject_keyword'])}에 대해서 화자별로 상세하게 요약해주세요."
            chat = chat + "\n\n" + question

            return chat
        
        # def make_chat_with_special_tokens(inp, speaker_map):
        #     chat = ["[Conversation]"]
        #     prev_speaker = None
        #     for cvt in inp['conversation']:
        #         speaker = speaker_map[cvt['speaker']]
        #         utterance = cvt['utterance']
        #         if prev_speaker == speaker:
        #             chat[-1] += f" {utterance}"
        #         else:
        #             chat.append(f"{speaker} : {utterance}")   
        #         prev_speaker = speaker
        #     chat = "\n".join(chat)

        #     question = f"[Question]\n위 {', '.join(inp['subject_keyword'])} 주제에 대한 대화를 화자별로 자세히 요약해주세요."
        #     chat = chat + "\n\n" + question

        #     return chat
        
        def chat_other_prompt(inp, speaker_map):
            topic_intro = f"이 대화는 {', '.join(inp['subject_keyword'])}에 대한 것입니다.\n"
            chat = [topic_intro + "[Conversation]"]
            prev_speaker = None
            for cvt in inp['conversation']:
                speaker = speaker_map[cvt['speaker']]
                utterance = cvt['utterance']
                if prev_speaker == speaker:
                    chat[-1] += f" {utterance}"
                else:
                    chat.append(f"{speaker} : {utterance}")
                prev_speaker = speaker
            chat = "\n".join(chat)

            question = f"[Question]\n위 대화를 화자별로 {', '.join(inp['subject_keyword'])} 주제로 자세히 요약해주세요."
            chat = chat + "\n\n" + question

            return chat
        
        def chat_other_prompt_keyword(inp, speaker_map):
            topic_intro = f"이 대화는 {', '.join(inp['subject_keyword'])}에 대한 것입니다.\n"
            chat = [topic_intro + "[Conversation]"]
            prev_speaker = None
            for cvt in inp['conversation']:
                speaker = speaker_map[cvt['speaker']]
                utterance = cvt['utterance']
                if prev_speaker == speaker:
                    chat[-1] += f" {utterance}"
                else:
                    chat.append(f"{speaker} : {utterance}")
                prev_speaker = speaker
            chat = "\n".join(chat)

            question = f"[Question]\n위 대화내에서 {', '.join(inp['subject_keyword'])}에 있는 정보에 대해 화자별로 자세히 요약해주세요."
            chat = chat + "\n\n" + question

            return chat
        
        def chat_other_prompt_keyword_nointro(inp, speaker_map):
            chat = ["[Conversation]"]
            prev_speaker = None
            for cvt in inp['conversation']:
                speaker = speaker_map[cvt['speaker']]
                utterance = cvt['utterance']
                if prev_speaker == speaker:
                    chat[-1] += f" {utterance}"
                else:
                    chat.append(f"{speaker} : {utterance}")
                prev_speaker = speaker
            chat = "\n".join(chat)

            question = f"[Question]\n위 대화의 주제는 {', '.join(inp['subject_keyword'])}이며 이 것을 화자별로 길고 자세히 요약해주세요."
            chat = chat + "\n\n" + question

            return chat
        
        def chat_other_prompt_min(inp, speaker_map):
            chat = ["[Conversation]"]
            prev_speaker = None
            for cvt in inp['conversation']:
                speaker = speaker_map[cvt['speaker']]
                utterance = cvt['utterance']
                if prev_speaker == speaker:
                    chat[-1] += f" {utterance}"
                else:
                    chat.append(f"{speaker} : {utterance}")
                prev_speaker = speaker
            chat = "\n".join(chat)

            question = f"[Question]\n위 대화의 주제는 {', '.join(inp['subject_keyword'])}이며 내용을 화자별로 자세하게 최소 다섯 문장이 되도록 요약해주세요."
            chat = chat + "\n\n" + question

            return chat

        def chat_other_prompt_topic_summary(inp, speaker_map):
            chat = ["[Conversation]"]
            prev_speaker = None
            for cvt in inp['conversation']:
                speaker = speaker_map[cvt['speaker']]
                utterance = cvt['utterance']
                if prev_speaker == speaker:
                    chat[-1] += f" {utterance}"
                else:
                    chat.append(f"{speaker} : {utterance}")
                prev_speaker = speaker
            chat = "\n".join(chat)

            question = f"[Question]\n위 대화의 주제는 {', '.join(inp['subject_keyword'])}이며 주제와 관련된 정보로 요약해주세요."
            chat = chat + "\n\n" + question

            return chat
        
        # for item in data: # 
        #     chat = item.get("chat", None) #
            
        #     if chat is None: #
        #         chat= "Basic" #
            
        def chat_se(inp, speaker_map):
            topic_intro = f"이 대화는 {', '.join(inp['subject_keyword'])}에 대한 것입니다.\n"
            chat = [topic_intro + "[Conversation]"]
            prev_speaker = None # 이게 뭐에 대한건지 비교해보기(같은 화자가 연속으로 나올경우에 그 대화문을 그전에 말한 대화문에 이어 붙이는 역할임.)-확인했습니다! 감사합니다
            for cvt in inp['conversation']:
                speaker = speaker_map[cvt['speaker']]
                utterance = cvt['utterance']
                if prev_speaker == speaker:
                    chat[-1] += f"{utterance}"
                else:
                    chat.append(f"{speaker}: {utterance}")
                prev_speaker = speaker
            chat = "\n".join(chat)

            question = f"[Question]\n위 {', '.join(inp['subject_keyword'])} 주제에 대한 대화를 화자별로 자세히, 총 네 줄 이상으로 요약해주세요."
            chat = chat + "\n\n" + question

            return chat
        
        
        def no_intro_chat(inp, speaker_map):
            chat = ["[Conversation]"]
            prev_speaker = None
            for cvt in inp['conversation']:
                speaker = speaker_map[cvt['speaker']]
                utterance = cvt['utterance']
                if prev_speaker == speaker:
                    chat[-1] += f" {utterance}"
                else:
                    chat.append(f"{speaker} : {utterance}")
                prev_speaker = speaker
            chat = "\n".join(chat)

            question = f"[Question]\n위 대화를 화자별로 {', '.join(inp['subject_keyword'])} 주제로 자세히 요약해주세요."
            chat = chat + "\n\n" + question

            return chat
        
        def no_intro_document_change_prompt(inp, speaker_map):
            chat = ["[Conversation]"]
            prev_speaker = None
            for cvt in inp['conversation']:
                speaker = speaker_map[cvt['speaker']]
                utterance = cvt['utterance']
                if prev_speaker == speaker:
                    chat[-1] += f" {utterance}"
                else:
                    chat.append(f"{speaker}가 {utterance}라고 말했다.")
                prev_speaker = speaker
            chat = " ".join(chat)

            question = f"[Question]\n위 대화를 화자별로 {', '.join(inp['subject_keyword'])} 주제로 자세히 요약해주세요."
            chat = chat + "\n\n" + question

            return chat
        
        def no_intro_document_allandspeaker(inp, speaker_map):
            chat = ["[Conversation]"]
            prev_speaker = None
            for cvt in inp['conversation']:
                speaker = speaker_map[cvt['speaker']]
                utterance = cvt['utterance']
                if prev_speaker == speaker:
                    chat[-1] += f" {utterance}"
                else:
                    chat.append(f"{speaker}가 {utterance}라고 말했다.")
                prev_speaker = speaker
            chat = "\n".join(chat)

            question = f"[Question]\n위 대화를 화자별로 {', '.join(inp['subject_keyword'])} 주제로 자세히 요약해주세요."
            chat = chat + "\n\n" + question

            return chat
        
        def no_intro_document_docsection(inp, speaker_map):
            chat = ["[Document]"]
            prev_speaker = None
            for cvt in inp['conversation']:
                speaker = speaker_map[cvt['speaker']]
                utterance = cvt['utterance']
                if prev_speaker == speaker:
                    chat[-1] += f" {utterance}"
                else:
                    chat.append(f"{speaker}가 {utterance}라고 말했다.")
                prev_speaker = speaker
            chat = " ".join(chat)

            question = f"[Question]\n위 문서를 화자별로 {', '.join(inp['subject_keyword'])} 주제로 자세히 요약해주세요."
            chat = chat + "\n\n" + question

            return chat
        
        def conversation_convert_to_document(inp, speaker_map):
            topic_intro = f"이 대화는 {', '.join(inp['subject_keyword'])}에 대한 것입니다.\n"
            chat = [topic_intro + "[Conversation]"]
            prev_speaker = None
            for cvt in inp['conversation']:
                speaker = speaker_map[cvt['speaker']]
                utterance = cvt['utterance']
                if prev_speaker == speaker:
                    chat[-1] += f" {utterance}"
                else:
                    chat.append(f"{speaker}가 {utterance}라고 말했다.")
                prev_speaker = speaker
            chat = "\n".join(chat)

            question = f"[Question]\n위 {', '.join(inp['subject_keyword'])} 주제에 대한 대화를 화자별로 자세히 요약해주세요."
            chat = chat + "\n\n" + question

            return chat

        def make_chat_original(inp):
            chat = ["[Conversation]"]
            for cvt in inp['conversation']:
                speaker = cvt['speaker']
                utterance = cvt['utterance']
                chat.append(f"{speaker}: {utterance}")
            chat = "\n".join(chat)

            question = f"[Question]\n위 {', '.join(inp['subject_keyword'])} 주제에 대한 대화를 화자별로 요약해주세요."
            chat = chat + "\n\n" + question

            return chat
        
        def make_conversation(inp,speaker_map):
            subject_keywords = ', '.join(inp['subject_keyword'])
            conversation_summary = f"\n주제: {subject_keywords}\n"

            subject_keywords_list = []
            for keyword in inp['subject_keyword']:
                subject_keywords_list.append(keyword)

            # 대화를 순서대로 나열
            for cvt in inp['conversation']:
                speaker = speaker_map[cvt['speaker']]
                utterance = cvt['utterance']
                conversation_summary += f"{speaker}: {utterance}\n"

            return conversation_summary, subject_keywords_list
        
        def analysis_prompt(inp,speaker_map):
            conversation = make_conversation(inp,speaker_map)
            # question = f"[질문]\n위 주제에 대한 일상 대화를 요약해주세요."
            analysis_steps = [
                "[문제 파악하기]\n" + f"이 대화는 {', '.join(inp['subject_keyword'])}에 대한 것입니다.",
                conversation,
                "[결론 도출하기]\n" + f"이 모든 정보를 바탕으로, 두 [speaker1], [speaker2] 사이의 대화를 {', '.join(inp['subject_keyword'])}에 관한 내용을 요약해주세요."
            ]
            return '\n'.join(analysis_steps)
        
        def analysis_prompt_v2(inp,speaker_map):
            conversation = make_conversation(inp,speaker_map)
            # question = f"[질문]\n위 주제에 대한 일상 대화를 요약해주세요."
            analysis_steps = [
                "[문제 파악하기]\n" + f"이 대화는 {', '.join(inp['subject_keyword'])}에 대한 것입니다.\n",
                conversation,
                "[대화맥락 파악하기]\n" + f"위 대화에서 {', '.join(inp['subject_keyword'])}에 관해 화자별로 파악하세요.\n",
                "[결론 도출하기]\n" + f"이 모든 정보를 바탕으로, 두 [speaker1], [speaker2] 사이의 대화를 {', '.join(inp['subject_keyword'])}에 관한 내용을 요약해주세요."
            ]
            return '\n'.join(analysis_steps)
        
        def analysis_prompt_v3(inp,speaker_map):
            conversation = make_conversation(inp,speaker_map)
            # question = f"[질문]\n위 주제에 대한 일상 대화를 요약해주세요."
            analysis_steps = [
                "[문제 파악하기]\n" + f"이 대화는 {', '.join(inp['subject_keyword'])}에 대한 것입니다.\n",
                conversation,
                "[정보 추출하기]\n" + f"위 대화에서 {', '.join(inp['subject_keyword'])}에 관해 화자별로 정리하세요.\n",
                "[결론 도출하기]\n" + f"이 모든 정보를 바탕으로, 두 [speaker1], [speaker2] 사이의 대화를 {', '.join(inp['subject_keyword'])}에 관한 내용을 요약해주세요."
            ]
            return '\n'.join(analysis_steps)

        def process_target_with_special_tokens(target, speaker_map, tokenizer):
            for speaker in speaker_map:
                target = target.replace(speaker, speaker_map[speaker])
            return tokenizer(target,
                             return_attention_mask=False,
                             add_special_tokens=False,
                             return_tensors="pt")

        def process_target_original(target, tokenizer):
            return tokenizer(target,
                             return_attention_mask=False,
                             add_special_tokens=False,
                             return_tensors="pt")
        
        for idx, example in enumerate(data):
            conversation = example["input"]["conversation"]
            speakers = list(set([cvt['speaker'] for cvt in conversation]))
            
            if len(speakers) != 2:
                raise ValueError("Each conversation must have exactly two speakers.")

            speaker_map = {speakers[0]: "[speaker1]", speakers[1]: "[speaker2]"}
            if mode == 'mode_with_special_tokens_concat':
                chat = make_chat_with_special_tokens_concat(example["input"], speaker_map)
                process_target_func = process_target_with_special_tokens
            elif mode == 'make_chat_with_special_tokens':
                chat = make_chat_with_special_tokens(example["input"], speaker_map)
                process_target_func = process_target_with_special_tokens
            elif mode == 'mode_with_special_tokens_topic_intro':
                chat = make_chat_with_special_tokens_topic_intro(example["input"], speaker_map)
                process_target_func = process_target_with_special_tokens
            elif mode == 'no_intro_document_docsection':
                chat = no_intro_document_docsection(example["input"], speaker_map)
                process_target_func = process_target_with_special_tokens
            elif mode == 'mode_original':
                chat = make_chat_original(example["input"])
                process_target_func = process_target_original
            elif mode == 'mode_moon':
                chat = chat_moon(example["input"], speaker_map)
                process_target_func = process_target_with_special_tokens
            elif mode == 'mode_moon_v2':
                chat = chat_moon_v2(example["input"], speaker_map)
                process_target_func = process_target_with_special_tokens
            elif mode == 'mode_moon_en':
                chat =  chat_moon_en(example["input"], speaker_map)
                process_target_func = process_target_with_special_tokens  
            elif mode == 'mode_moon_fewshot':
                chat = chat_moon_fewshot(example["input"], speaker_map, train_data)
                process_target_func = process_target_with_special_tokens
                process_target_func = process_target_with_special_tokens
            elif mode == 'chat_other_prompt_min':
                chat = chat_other_prompt_min(example["input"], speaker_map)
                process_target_func = process_target_with_special_tokens
            elif mode == 'chat_other_prompt':
                chat = chat_other_prompt(example["input"],speaker_map)
                process_target_func = process_target_with_special_tokens
            elif mode == 'no_intro_chat':
                chat = no_intro_chat(example["input"],speaker_map)
                process_target_func = process_target_with_special_tokens
            elif mode == 'no_intro_document_change_prompt':
                chat = no_intro_document_change_prompt(example["input"],speaker_map)
                process_target_func = process_target_with_special_tokens
            elif mode == 'no_intro_document_allandspeaker':
                chat = no_intro_document_allandspeaker(example["input"],speaker_map)
                process_target_func = process_target_with_special_tokens
            elif mode == 'chat_other_prompt_keyword':
                chat = chat_other_prompt_keyword(example["input"],speaker_map)
                process_target_func = process_target_with_special_tokens
            elif mode == 'chat_other_prompt_keyword_nointro':
                chat = chat_other_prompt_keyword_nointro(example["input"],speaker_map)
                process_target_func = process_target_with_special_tokens
            elif mode == 'make_chat_with_special_tokens_keywordinprompt':
                chat = make_chat_with_special_tokens_keywordinprompt(example["input"],speaker_map)
                process_target_func = process_target_with_special_tokens
            elif mode == 'make_chat_with_special_tokens_keywordinprompt_nointro':
                chat = make_chat_with_special_tokens_keywordinprompt_nointro(example["input"],speaker_map)
                process_target_func = process_target_with_special_tokens
            elif mode == 'analysis_prompt':
                chat = analysis_prompt(example["input"],speaker_map)
                process_target_func = process_target_with_special_tokens
            elif mode == 'chat_se':
                chat = chat_se(example["input"], speaker_map)
                process_target_func = process_target_with_special_tokens   
            elif mode == 'analysis_prompt_v2':
                chat = analysis_prompt_v2(example["input"],speaker_map)
                process_target_func = process_target_with_special_tokens
            elif mode == 'analysis_prompt_v3':
                chat = analysis_prompt_v3(example["input"],speaker_map)
                process_target_func = process_target_with_special_tokens
            elif mode == 'chat_other_prompt_topic_summary':
                chat = chat_other_prompt_topic_summary(example["input"],speaker_map)
                process_target_func = process_target_with_special_tokens

            # print(chat)
            if model_type == "gemma":
                message = [
                    {"role": "user", "content": chat},
                    {"role": "assistant", "content": PROMPT}
                ]
            elif model_type == "default":
                message = [
                    {"role": "system", "content": PROMPT},
                    {"role": "user", "content": chat}
                ]
            # elif model_type == "mistral":
            #     print("1")
            #     input()
            #     message = [f"[INST] {PROMPT}{chat}[/INST]"]
            
            
            if PROMPT == "None":
                message=[{"role": "user", "content": chat}]
                
                
            source = tokenizer.apply_chat_template(
                message,
                add_generation_prompt=True,
                return_tensors="pt",
            )

            target = example["output"]
            if target != "":
                target += tokenizer.eos_token

            target = process_target_func(target, speaker_map if mode != 'mode_original' else tokenizer, tokenizer)
            target["input_ids"] = target["input_ids"].type(torch.int64)

            input_ids = torch.concat((source[0], target["input_ids"][0]))
            labels = torch.concat((torch.LongTensor([IGNORE_INDEX] * source[0].shape[0]), target["input_ids"][0]))
            self.inp.append(input_ids)
            self.label.append(labels)
            self.speaker_mappings.append(speaker_map)

    def __len__(self):
        return len(self.inp)

    def __getitem__(self, idx):
        return {"input_ids": self.inp[idx], "labels": self.label[idx], "speaker_map": self.speaker_mappings[idx]}

class DataCollatorForSupervisedDataset(object):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, instances):
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(ids) for ids in input_ids], batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence([torch.tensor(lbls) for lbls in labels], batch_first=True, padding_value=-100)
        return {
            'input_ids':input_ids,
            'labels':labels,
            'attention_mask':input_ids.ne(self.tokenizer.pad_token_id),
        }

class DataCollatorForInferenceDataset(object):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, instances):
        input_ids, labels, speaker_maps = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels", "speaker_map"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
        return {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': input_ids.ne(self.tokenizer.pad_token_id),
            'speaker_maps': speaker_maps
        }

class moonDataset(Dataset):
    def __init__(self, fname, rejectfname, tokenizer, model_type):
        IGNORE_INDEX = -100
        self.inp = []
        self.chosen = []
        self.rejected = []
        self.speaker_mappings = []
        
        PROMPT = "당신은 유능한 AI 어시스턴트입니다. 주어진 대화에서 화자별로 주제에 기반하여 상세하게 요약해주세요. Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request"

        with open(fname, "r") as f:
            data = json.load(f)
        
        with open(rejectfname, "r") as f:
            reject_data = json.load(f)

        special_tokens = {'additional_special_tokens': ["[speaker1]", "[speaker2]"]}
        tokenizer.add_special_tokens(special_tokens)

        def chat_moon(inp, speaker_map):#make question 
            chat = ["[Conversation]"]
            prev_speaker = None
            for cvt in inp['conversation']:
                speaker = speaker_map[cvt['speaker']]
                utterance = cvt['utterance']
                if prev_speaker == speaker:
                    chat[-1] += f" {utterance}"
                else:
                    chat.append(f"{speaker}: {utterance}")
                prev_speaker = speaker
            chat = "\n".join(chat)

            question = f"[Question]\n아래 대화를 읽고 화자별로 자세히 요약해주세요.\n 중요한 대화 토픽은{', '.join(inp['subject_keyword'])} 입니다.\n 집중해서 읽고 토픽에 기반한 대화 내용을 요약해주세요."
                        
            chat = chat + "\n\n" + question

            return chat

        # def process_target_with_special_tokens(target, speaker_map, tokenizer):#target tokenizing
        #     for speaker in speaker_map:
                
        #         target = target.replace(speaker, speaker_map[speaker])
        #     return tokenizer(target,
        #                      return_attention_mask=False,
        #                      add_special_tokens=False,
        #                      return_tensors="pt")
            
            
        for example in data:
            conversation = example["input"]["conversation"]
            speakers = list(set([cvt['speaker'] for cvt in conversation]))
            
            if len(speakers) != 2:
                raise ValueError("Each conversation must have exactly two speakers.")

            speaker_map = {speakers[0]: "[speaker1]", speakers[1]: "[speaker2]"}
            
            chat = chat_moon(example["input"], speaker_map)

            

            # print(chat)
            if model_type == "gemma":
                message = [
                    {"role": "user", "content": chat},
                    {"role": "assistant", "content": PROMPT}
                ]
            elif model_type == "default":
                message = [
                    {"role": "system", "content": PROMPT},
                    {"role": "user", "content": chat}
                ]

            source = tokenizer.apply_chat_template(#input_ids, attention_mask
                message,
                add_generation_prompt=True,
                tokenize = False
            )

            target = example["output"]
            if target != "":
                target += tokenizer.eos_token
                
            for speaker in speaker_map:
                target = target.replace(speaker, speaker_map[speaker])
            
            self.chosen.append(target)
            # print(target)
            #target = process_target_with_special_tokens(target, speaker_map, tokenizer)#target tokenizing
            #target["input_ids"] = target["input_ids"].type(torch.int64)#target tokenizing

            #reject_data = process_target_with_special_tokens(reject_data, speaker_map, tokenizer)
            #reject_data["input_ids"] = reject_data["input_ids"].type(torch.int64)
            
            
            input_ids = source
            #chosen = target["input_ids"][0]
            
            self.inp.append(input_ids)
            #self.chosen.append(chosen)
           
            self.speaker_mappings.append(speaker_map)
        for reject_data in reject_data:
          reject_data = reject_data["output"]
          if reject_data != "":
                reject_data += tokenizer.eos_token
          reject = reject_data

          self.rejected.append(reject)

    def __len__(self):
        return len(self.inp)

    def __getitem__(self, idx):
        return {"prompt": self.inp[idx],"chosen":self.chosen[idx],"rejected":self.rejected[idx],"speaker_map": self.speaker_mappings[idx]}
        #return {"input_ids": self.inp[idx], "chosen": self.chosen[idx], "rejected": self.rejected[idx], "speaker_map": self.speaker_mappings[idx]}

