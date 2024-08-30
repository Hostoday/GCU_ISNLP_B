import json
import torch
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, fname, tokenizer, prompt, model_type , mode='mode_with_special_tokens_concat'):
        IGNORE_INDEX = -100
        self.inp = []
        self.label = []
        self.speaker_mappings = []
        
        PROMPT = prompt


        with open(fname, "r") as f:
            data = json.load(f)
            
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

            question = f"[Question]\n아래 대화를 읽고 화자별로 자세히 요약해주세요.\n 중요한 대화 토픽은{', '.join(inp['subject_keyword'])} 입니다.\n 집중해서 읽고 토픽에 기반한 대화 내용을 한국어로 요약해주세요."
            
            chat = chat + "\n\n" + question

            return chat
        
        def chat_moon_v2(inp, speaker_map):
            conversation = make_conversation(inp,speaker_map)
            
            question = f"['질문']\n아래 대화를 읽고 화자별로 자세히 요약해주세요.\n 중요한 대화 주제는 '{', '.join(inp['subject_keyword'])}' 입니다. "

            yoyo = "['요청'] 대화 내용을 주제에 맞게 꼭 한국어로 요약해주세요.\n 각 화자의 말을 구분하여 꼭 한국어로 요약해주세요."    
            chat = question + conversation + yoyo

            return chat
        
        def chat_moon_co(inp, speaker_map):
            conversation = make_conversation(inp,speaker_map)

            question = f"['문제']:위 대화를 읽고 자세히 요약해주세요.\n 주제는 '{', '.join(inp['subject_keyword'])}' 입니다.\n"
            style = "저는 요약을 위해 대규모 언어 모델을 사용하는 소프트웨어 엔지니어입니다. 다음 텍스트를 요약합니다."
            step = "['1단계'] 대화 내용을 주제에 맞게 요약해주세요.\n"
            step2 = "['2단계'] 요약한 내용에 관한 각 화자의 대화를 전부 요약해주세요.\n"
            step3 = "['3단계'] 1단계와 2단계의 요약을 합쳐 분석하고 수정이 필요하면 수정해주세요.\n"

            chat = style + conversation + question + step + step2 + step3

            return chat
        
        def chat_moon_co2(inp, speaker_map):
            
            style = "\n저는 요약을 위해 대규모 언어 모델을 사용하는 소프트웨어 엔지니어입니다. 다음 텍스트를 요약합니다.\n"

            conversation = make_conversation(inp,speaker_map)
            
            question = f"\n['대화 정리'] 대화에서 주제 '{', '.join(inp['subject_keyword'])}'와 관련된 내용을 정리합니다.\n"

            question2 = "['대화 요약'] 정리한 내용으로 대화 내용을 요약합니다.\n"

            question3 = "['화자별 요약'] 정리한 내용에서 각 speaker의 대화와 생각을 요약합니다.\n"

            question4 = "['종합 요약'] 마지막으로 대화 요약과 화자별 요약을 종합하여 정리합니다.\n"

            chat = style + conversation + question + question2 + question3 + question4

            return chat
        
        def chat_moon_co3(inp, speaker_map):
            
            style = "\n저는 요약을 위해 대규모 언어 모델을 사용하는 소프트웨어 엔지니어입니다. 다음 텍스트를 요약합니다.\n"

            conversation = make_conversation(inp,speaker_map)

            question = f"\n['대화 요약'] 위 대화에서 주제 '{', '.join(inp['subject_keyword'])}'와 관련된 내용을 요약해야 합니다.\n"
            question = "첫번쨰로 주제에 관한 요약을 해주세요.\n"
            question2 = "두번째로 [speaker1] 가 말한 문장 에서 대화 요약 관련있는 문장과 생각을 요약해줘.\n"
            question3 = "마지막으로 [speaker2] 가 말한 문장 에서 대화 요약 관련있는 문장과 생각을 요약해줘.\n"

            chat = style + conversation + question + question2 + question3
            return chat

        def chat_moon_co4(inp, speaker_map):
            conversation = make_conversation(inp,speaker_map)
            
            question = f"['대화 주제 분석'] 대화에서 주제 '{', '.join(inp['subject_keyword'])}'와 관련된 내용을 분석합니다.\n"

            question2 = "['화자별 대화 분석'] [speaker1]과 [speaker2] 가 한 말을 분석합니다.\n" 

            question3 = "['최종 요약'] 분석한 대화 주제와 각 화자별 대화를 요약합니다. 요약문과 대화내용은 의미적으로 비슷하고 3줄이상여야 합니다.\n "
            
            chat = conversation + question + question2 + question3

            return chat

        def chat_moon_co5(inp, speaker_map):
            conversation = make_conversation(inp,speaker_map)
            b = "대화문을 읽고 아래의 형식에 따라 요약문을 만들어주세요.\n"
            q = f"['주제']: '{', '.join(inp['subject_keyword'])}'.\n "
            o = "['전체 대화 내용 요약문']: 대화를 읽고 내용을 길고 상세하게 요약합니다.\n"
            q1 = "['speaker1 대화 요약문']: [speaker1]이 한 말을 생각을 길고 상세하게 요약합니다.\n"
            q2 = "['speaker2 대화 요약문']: [speaker2]이 한 말과 생각을 길고 상세하게 요약합니다.\n"
            q3 = "['최종 요약문']: ['전체 대화 내용 요약문'], ['speaker1 대화 요약문'], ['speaker2 대화 요약문']을 순서대로 합쳐 최종 요약문을 만듭니다.\n"
            
            chat = conversation + b + q + o + q1 + q2 + q3
            return chat

        def chat_moon_co6(inp, speaker_map):
            conversation = make_conversation(inp,speaker_map)
            b = "대화문을 읽고 아래의 형식에 따라 요약문을 만들어주세요.\n"
            q = f"['전체 내용 요약']: 주제'{', '.join(inp['subject_keyword'])}' 에 대한 내용을 요약합니다.\n "
            q1 = "['speaker1 대화 요약문']: [speaker1]이 한 말을 생각을 길고 상세하게 요약합니다.\n"
            q2 = "['speaker1 대화 요약문']: [speaker2]이 한 말과 생각을 길고 상세하게 요약합니다.\n"
            q3 = "['최종 요약문']: ['전체 대화 내용 요약문'], ['speaker1 대화 요약문'], ['speaker2 대화 요약문']을 순서대로 합쳐 최종 요약문을 만듭니다.\n"
            
            chat = conversation + b + q + q1 + q2 + q3
            return chat
        
        def chat_moon_co7(inp, speaker_map):
            conversation = make_conversation(inp,speaker_map)
            b = "대화문을 읽고 아래의 형식에 따라 요약문을 만들어주세요.\n"
            q = f"['대화 주제']: 대화 주제를 찾습니다. 키워드는 {', '.join(inp['subject_keyword'])} 입니다.\n"
            q1 = "['speaker1 대화 요약문']: [speaker1]이 한 말을 생각을 길고 상세하게 전부 다 요약합니다.\n"
            q2 = "['speaker2 대화 요약문']: [speaker2]이 한 말과 생각을 길고 상세하게 전부 다 요약합니다.\n"
            q3 = "[최종 요약문]: ['대화 주제'], ['speaker1 대화 요약문'], ['speaker2 대화 요약문']을 순서대로 합쳐 최종 요약문을 만듭니다.\n"

            chat = conversation + b + q + q1 + q2 + q3
            return chat
        
        ## 전부다
        def chat_moon_co8(inp, speaker_map):
            conversation = make_conversation(inp,speaker_map)
            b = "대화문을 읽고 아래의 형식에 따라 요약문을 만들어주세요.\n"
            q1 = "['speaker1 대화 요약문']: [speaker1]이 한 말을 생각을 길고 상세하게 전부 다 요약합니다.\n"
            q2 = "['speaker2 대화 요약문']: [speaker2]이 한 말과 생각을 길고 상세하게 전부 다 요약합니다.\n"
            q4 = "['대화 주제'] : ['speaker1 대화 요약문'], ['speaker2 대화 요약문'] 이 두개의 요약문의 공통적인 주제를 정리합니다.\n"
            q3 = "[최종 요약문]: ['대화 주제'], ['speaker1 대화 요약문'], ['speaker2 대화 요약문']을 순서대로 합쳐 최종 요약문을 만듭니다.\n"

            chat = conversation + b + q1 + q2 + q4 + q3
            return chat
        
        ## 핵심만
        def chat_tg1(inp, speaker_map):
            conversation = make_conversation(inp,speaker_map)
            b = "대화문을 읽고 아래의 형식에 따라 요약문을 만들어주세요.\n"
            q1 = "['speaker1 대화 요약문']: [speaker1]이 한 말을 핵심적인 부분만을 포함해서 요약합니다.\n"
            q2 = "['speaker2 대화 요약문']: [speaker2]이 한 말을 핵심적인 부분만을 포함해서 요약합니다.\n"
            q4 = "['대화 주제'] : ['speaker1 대화 요약문'], ['speaker2 대화 요약문'] 이 두개의 요약문의 공통적인 주제를 정리합니다.\n"
            q3 = "[최종 요약문]: ['대화 주제'], ['speaker1 대화 요약문'], ['speaker2 대화 요약문']을 순서대로 합쳐 최종 요약문을 만듭니다.\n"

            chat = conversation + b + q1 + q2 + q4 + q3
            return chat
        
        ## 감정만
        def chat_tg2(inp, speaker_map):
            conversation = make_conversation(inp,speaker_map)
            b = "대화문을 읽고 아래의 형식에 따라 요약문을 만들어주세요.\n"
            q1 = "['speaker1 대화 요약문']: [speaker1]의 감정을 중심으로 [speaker1]이 한 말을 구체적으로 요약합니다.\n"
            q2 = "['speaker2 대화 요약문']: [speaker2]의 감정을 중심으로 [speaker2]이 한 말을 구체적으로 요약합니다.\n"
            q4 = "['대화 주제'] : ['speaker1 대화 요약문'], ['speaker2 대화 요약문'] 이 두개의 요약문의 공통적인 주제를 정리합니다.\n"
            q3 = "[최종 요약문]: ['대화 주제'], ['speaker1 대화 요약문'], ['speaker2 대화 요약문']을 순서대로 합쳐 최종 요약문을 만듭니다.\n"

            chat = conversation + b + q1 + q2 + q4 + q3
            return chat
        
        def chat_tg3(inp, speaker_map):
            conversation = make_conversation(inp,speaker_map)
            b = "대화문을 읽고 아래의 형식에 따라 요약문을 만들어주세요.\n"
            q1 = "['speaker1 대화 요약문']: [speaker1]에게 일어난 사건을 중심으로 [speaker1]이 한 말을 구체적으로 요약합니다.\n"
            q2 = "['speaker2 대화 요약문']: [speaker2]에게 일어난 사건을 중심으로 [speaker2]이 한 말을 구체적으로 요약합니다.\n"
            q4 = "['대화 주제'] : ['speaker1 대화 요약문'], ['speaker2 대화 요약문'] 이 두개의 요약문의 공통적인 주제를 정리합니다.\n"
            q3 = "[최종 요약문]: ['대화 주제'], ['speaker1 대화 요약문'], ['speaker2 대화 요약문']을 순서대로 합쳐 최종 요약문을 만듭니다.\n"

            chat = conversation + b + q1 + q2 + q4 + q3
            return chat
        
        def cat_mod(output, inp, speaker_map):
            conversation = make_conversation(inp,speaker_map)

            a = "[요약문]: " + output
            b = "대화문을 읽고 지침을 따라 주어진 요약문에서 틀린 내용을 고쳐주세요. [지침]: 반복된 내용이 없도록 고치세요. [speaker1], [speaker2]의 요약내용 중에 오류가 있다면 고쳐주세요. [speaker1], [speaker2]가 말한 내용이 섞여있다면 명확하게 구분이 되도록 나눠주세요. [중요]: 요약 내용은 똑같이 해주세요! 요약문에 없는 내용 넣지 말아주세요. "
           
            chat = conversation + a + b
            return chat


        def chat_moon_spanish(inp, speaker_map):
            conversation = make_conversation(inp,speaker_map)
        
            question = f"[Pregunta]\nLee la siguiente conversación y proporciona un resumen detallado para cada hablante.\n El tema principal de la conversación es '{', '.join(inp['subject_keyword'])}'.\n Resume la conversación en coreano, enfocándote en los puntos principales y opiniones expresadas por cada hablante.\n Asegúrate de que el resumen mantenga una estructura de oración y elecciones de vocabulario similares al texto original. Utiliza expresiones apropiadas para que el resumen sea natural y lógico.\n Debes proporcionar el resumen en coreano."

            chat = question + conversation

            return chat

        def chat_moon_fewshot(inp, speaker_map):
            conversation = make_conversation(inp,speaker_map)
            few = f"[Conversation] \n [speaker2] : 저는 여행 다니는 것을 굉장히 좋아하는데요. 그래가지고 스페인이나 뭐 영국 유럽 아니면 국내에서도 뭐 강릉이나 전주 같은 데를 많이 다녔는데 혹시 여행 다니는 거 좋아하시나요? \n[speaker1] : 저 여행 다니는 거 되게 좋아해서 대학교 내내 여행을 엄청 많이 다녔었는데요. 제가 고등학교 때는 여행에 대해 흥미가 없었는데 그게 좀 아버지가 짠대로 패키지처럼 여행을 다녀서 그런 것 같아요. 그래서 대학교 간 이후로는 해외여행을 되게 많이 갔었는데 그중에서 제일 기 좋았던 거는 스페인이랑 포르투갈이었거든요. 어~ 혹시 포르투갈이나 스페인 유럽 쪽 다녀오신 적 있으신가요? \n[speaker2] : 어~ 네. 저도 우연히 스페인과 포르투갈을 다녀왔었었습니다. 어~ 저는 스페인 중에서도 마드리드에 근교에 있었던 톨레도라는 지역이 굉장히 좋았는데요. 그 톨레도에서 특히 기억에 남았던 거는 거기에 대성당이 있는데 그 성당이 엄청 화려하더라고요. 그래서 거기를 꾸며논 거를 보면은 금을 엄청 많이 사용해가지고 되게 빤짝빤짝하고 좀 성당은 보통 좀 소박하다라는 인식이 있었는데 아~ 이렇게 화려한 성당도 있구나라는 거를 새롭게 알게 됐었습니다. 어~ 또 톨레도에 지역 음식도 같이 먹었었는데 아~ 이름은 지금 잘 생각이 나지는 않지만 굉장히 달달했던 그런 디저트 종류였는데 그~ 디저트도 먹고 그다음에 천천히 걸어 다니면서 주변 풍경도 보고 근교 여행만의 약간 소박한 맛이 있었다고 생각을 합니다. 어~ 또 물론 마드리드도 굉장히 좋았는데 유럽 여행을 많이 가셨다고 해서 혹시 톨레도도 가본 적이 있나요?\n[speaker1] : 아~ 제가 톨레도도 다녀왔는데 저는 이제 여행 일정을 길게 잡아서 톨레도는 하루를 봤는데 도 그렇게 너무 더웠기 때문에 많이 보진 못한 것 같아요. 그때는 버스 관광버스를 타고 계속 돌아다니면서 이제 내리는 데마다 관광을 할 수 있는 버스를 탔는데요. 그 버스를 타고 전체를 다 내려서 보려고 했지만 날씨가 너무 더워서 금방 금방 이제 xx 장소로 넘어갔던 것 같 같습니다. 거기는 이제 고대 도시라고 해서 사람들이 많이 추천한 거에 비해서는 저는 하루를 잡기에는 조금 부족한 여행지라는 생각이 들었고 오히려 광장에서 쇼핑을 했던 게 더 기억에 남습니다.\n[Keyword]\n해외여행 \n[output] 이 대화에서 화자들은 좋았던 여행지와 기억나는 주요 명소에 대해 이야기했습니다. SD2000001은 여행을 좋아하여 국내, 해외 여행을 많이 다녔다고 말했습니다. 특히 기억에 남는 여행지로 스페인 마드리드의 톨레도를 소개했습니다. 그 중 화려하게 꾸며진 대성당과 디저트가 인상적이었다고 이야기했습니다. SD2000002는 대학교에 진학한 후 해외여행을 자주 다녔고, 스페인과 포루투갈이 가장 기억에 남는 여행지라고 말했습니다. 그리고 톨레도도 다녀왔지만 날씨가 더워서 제대로 구경하지 못했다는 경험을 이야기했습니다"
            question = f"위의 대화 요약인 [output]처럼 [keyword] {', '.join(inp['subject_keyword'])} 에 대한 대화를 자세히 요약해주세요. 아래에 있는 두 speaker의 대화를 읽고 비슷하게 요약해주세요."
            
            chat = few + question + conversation

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
        
        def make_chat_with_special_tokens(inp, speaker_map):
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

            question = f"[Question]\n위 {', '.join(inp['subject_keyword'])} 주제에 대한 대화를 화자별로 자세히 요약해주세요."
            chat = chat + "\n\n" + question

            return chat
        
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
        
        def chat_se2(inp, speaker_map):
            # topic_intro = f"이 대화는 {', '.join(inp['subject_keyword'])}에 대한 것입니다.\n"
            # chat = [topic_intro + "[Conversation]"]
            # prev_speaker = None # 이게 뭐에 대한건지 비교해보기(같은 화자가 연속으로 나올경우에 그 대화문을 그전에 말한 대화문에 이어 붙이는 역할임.)-확인했습니다! 감사합니다
            # for cvt in inp['conversation']:
            #     speaker = speaker_map[cvt['speaker']]
            #     utterance = cvt['utterance']
            #     if prev_speaker == speaker:
            #         chat[-1] += f"{utterance}"
            #     else:
            #         chat.append(f"{speaker}: {utterance}")
            #     prev_speaker = speaker
            # chat = "\n".join(chat)
            conversation = make_conversation(inp,speaker_map)

            question_list = [
            f"[Question]\n위 {', '.join(inp['subject_keyword'])} 주제에 대한 대화를 화자별로 자세히 파악해주세요.",
            f"[Final Statement]\n 앞의 내용들을 통합하여 두 화자의 {', '.join(inp['subject_keyword'])}에 대한 대화를 네 줄 이상으로 상세히 정리해주세요."
            ]

            return '\n'.join(question_list)
        
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
            conversation_summary = f"[대화 분석]\n주제: {subject_keywords}\n"
            
            # 대화를 순서대로 나열
            for cvt in inp['conversation']:
                speaker = speaker_map[cvt['speaker']]
                utterance = cvt['utterance']
                conversation_summary += f"{speaker}: {utterance}\n"

            return conversation_summary
        
        def analysis_prompt(inp,speaker_map):
            conversation = make_conversation(inp,speaker_map)
            question = f"위 주제에 대한 일상 대화를 요약해주세요."
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
        
        def analysis_prompt_v2_se(inp, speaker_map):
            conversation = make_conversation(inp, speaker_map)
            
            analysis_steps = [
                "[문제 파악하기]\n" + f"이 대화는 {', '.join(inp['subject_keyword'])}에 대한 것입니다.\n",
                conversation,
                "[대화맥락 파악하기]\n" + f"위 대화에서 {', '.join(inp['subject_keyword'])}에 대해 각 화자의 입장과 주된 발언을 정리하세요.\n",
                "[핵심 내용 요약하기]\n" + "위에서 파악한 내용을 바탕으로, 이 대화의 주요 포인트를 추려내세요. 중요한 정보와 의견의 핵심이 무엇인지 명확히 기술하세요.\n",
                "[결론 도출하기]\n" + f"마지막으로, 두 화자의 대화를 {', '.join(inp['subject_keyword'])}에 관한 내용을 바탕으로 요약하세요. 이 요약은 모든 중요한 정보가 포함되도록 간결하게 작성해야 합니다."
            ]
            
            return '\n'.join(analysis_steps)

        def analysis_prompt_v2_se2(inp, speaker_map):
            conversation = make_conversation(inp, speaker_map)
            
            analysis_steps = [
                "[문제 파악하기]\n" + f"이 대화는 주로 {', '.join(inp['subject_keyword'])}에 대한 것입니다. 각 화자의 의견이 다루고 있는 주요 주제를 식별하세요.\n",
                conversation,
                "[대화맥락 파악하기]\n" + f"위 대화에서 각 화자가 {', '.join(inp['subject_keyword'])}에 대해 제시한 입장과 그들이 중점을 둔 핵심 논점을 요약하세요. 각 화자가 강조한 감정이나 우려 사항을 명확하게 드러내세요.\n",
                "[핵심 내용 요약하기]\n" + "위에서 파악한 내용을 바탕으로, 대화의 주요 포인트를 구체적인 키워드와 구문을 사용하여 요약하세요. 중요한 정보와 의견의 핵심이 무엇인지 명확히 기술하십세요.\n",
                "[결론 도출하기]\n" + f"마지막으로, 두 화자의 대화를 {', '.join(inp['subject_keyword'])}에 관한 내용을 바탕으로 요약하세요. 이 요약은 모든 중요한 정보가 포함되도록 하며, 원문과 최대한 일치하는 표현을 사용해 간결하게 작성해야 해요."
            ]
            return '\n'.join(analysis_steps)


        def analysis_prompt_v2_se3(inp, speaker_map):
            conversation = make_conversation(inp, speaker_map)
            
            analysis_steps = [
                "[문제 파악하기]\n" + f"이 대화는 주로 {', '.join(inp['subject_keyword'])}에 대한 것입니다. 각 화자가 언급한 주요 주제를 명확히 식별하세요. 이를 통해 각 화자가 대화에서 중요하게 다룬 포인트를 파악해야 합니다.\n",
                conversation,
                "[대화맥락 파악하기]\n" + f"위 대화에서 각 화자가 {', '.join(inp['subject_keyword'])}에 대해 제시한 입장과 중점적으로 다룬 논점을 명확히 요약하세요. 각 화자가 강조한 감정이나 우려 사항을 명확히 드러내세요.\n",
                "[핵심 내용 요약하기]\n" + "위에서 파악한 내용을 바탕으로, 대화의 주요 포인트를 구체적인 키워드와 구문을 사용하여 명확하게 요약하세요. 이때 각 화자의 입장이 어떻게 대조되었는지, 주요 차이점과 공통점을 구체적으로 기술해야 합니다.\n",
                "[결론 도출하기]\n" + f"마지막으로, 두 화자의 대화를 {', '.join(inp['subject_keyword'])}에 관한 내용으로 요약하세요. 모든 중요한 정보가 포함되도록 하며, 원문에서 사용된 주요 단어와 구문을 최대한 유지하면서 간결하게 작성해야 합니다. 예를 들어, 원문에서 사용된 표현을 재사용하거나 중요한 문장을 축약해 사용해도 좋습니다."
            ]
            
            return '\n'.join(analysis_steps)
        
        def analysis_prompt_v2_se4(inp, speaker_map):
            conversation = make_conversation(inp, speaker_map)
            
            analysis_steps = [
                  "[주요 주제 파악]\n" + f"이 대화는 주로 {', '.join(inp['subject_keyword'])}에 관한 것입니다. 대화의 핵심 주제를 간단히 요약하세요.\n",
                  conversation,
                  "[대화 요약]\n" + f"위 대화에서 각 화자가 강조한 내용을 간결하게 요약하고, 대화에서 가장 중요한 논점을 간단한 문장으로 정리하세요.\n",
                  "[최종 요약]\n" + "대화의 주요 포인트를 통합하여, 간결하고 명확하게 최종 요약문을 작성하세요. 원문에서 사용된 주요 단어를 유지하되, 불필요한 세부사항을 제외하고 요약문을 작성하세요."
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

        def analysis_prompt_v4(inp,speaker_map):
            conversation = make_conversation(inp,speaker_map)
            # question = f"[질문]\n위 주제에 대한 일상 대화를 요약해주세요."
            analysis_steps = [
                "[문제 파악하기]\n" + f"이 대화는 {', '.join(inp['subject_keyword'])}에 관한 내용입니다.\n",
                conversation,
                "[화자별 정보 파악하기]\n" + f"위 대화에서 화자별로 {', '.join(inp['subject_keyword'])}에 관해 설명하세요.\n",
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
        
        for example in data:
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
            elif mode == 'mode_moon_co':
                chat =  chat_moon_co(example["input"], speaker_map)
                process_target_func = process_target_with_special_tokens
            elif mode == 'mode_moon_co2':
                chat = chat_moon_co2(example["input"], speaker_map)
                process_target_func = process_target_with_special_tokens
            elif mode == 'mode_moon_co3':
                chat = chat_moon_co3(example["input"], speaker_map)
                process_target_func = process_target_with_special_tokens
            elif mode == 'mode_moon_co4':
                chat = chat_moon_co4(example["input"], speaker_map)
                process_target_func = process_target_with_special_tokens
            elif mode == 'mode_moon_co5':
                chat = chat_moon_co5(example["input"], speaker_map)
                process_target_func = process_target_with_special_tokens
            elif mode == 'mode_moon_co6':
                chat = chat_moon_co6(example["input"], speaker_map)
                process_target_func = process_target_with_special_tokens
            elif mode == 'mode_moon_co7':
                chat = chat_moon_co7(example["input"], speaker_map)
                process_target_func = process_target_with_special_tokens
            elif mode == 'mode_moon_co8':
                chat = chat_moon_co8(example["input"], speaker_map)
                process_target_func = process_target_with_special_tokens
            elif mode == 'mode_moon_fewshot':
                chat = chat_moon_fewshot(example["input"], speaker_map)
                process_target_func = process_target_with_special_tokens
            elif mode == 'mode_moon_spanish':
                chat = chat_moon_spanish(example["input"], speaker_map)
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
            elif mode == 'chat_se2':
                chat = chat_se2(example["input"], speaker_map)
                process_target_func = process_target_with_special_tokens
            elif mode == 'analysis_tg1':
                chat = chat_tg1(example["input"],speaker_map)
                process_target_func = process_target_with_special_tokens 
            elif mode == 'analysis_tg2':
                chat = chat_tg2(example["input"],speaker_map)
                process_target_func = process_target_with_special_tokens
            elif mode == 'analysis_tg3':
                chat = chat_tg3(example["input"],speaker_map)
                process_target_func = process_target_with_special_tokens
            elif mode == 'mod_cat':
                chat = cat_mod(example['output'],example["input"],speaker_map)
                process_target_func = process_target_with_special_tokens    
            elif mode == 'analysis_prompt_v2':
                chat = analysis_prompt_v2(example["input"],speaker_map)
                process_target_func = process_target_with_special_tokens
            elif mode == 'analysis_prompt_v2_se':
                chat = analysis_prompt_v2_se(example["input"],speaker_map)
                process_target_func = process_target_with_special_tokens
            elif mode == 'analysis_prompt_v2_se2':
                chat = analysis_prompt_v2_se(example["input"],speaker_map)
                process_target_func = process_target_with_special_tokens
            elif mode == 'analysis_prompt_v2_se3':
                chat = analysis_prompt_v2_se(example["input"],speaker_map)
                process_target_func = process_target_with_special_tokens
            elif mode == 'analysis_prompt_v3':
                chat = analysis_prompt_v3(example["input"],speaker_map)
                process_target_func = process_target_with_special_tokens
            elif mode == 'analysis_prompt_v4':
                chat = analysis_prompt_v3(example["input"],speaker_map)
                process_target_func = process_target_with_special_tokens
            elif mode == 'chat_other_prompt_topic_summary':
                chat = chat_other_prompt_topic_summary(example["input"],speaker_map)
                process_target_func = process_target_with_special_tokens
            elif mode == 'analysis_prompt_v4':
                chat = analysis_prompt_v4(example["input"],speaker_map)
                process_target_func = process_target_with_special_tokens

            # print(chat)
            if model_type == "gemma":
                message = [
                    {"role": "user", "content": chat},
                    {"role": "assistant", "content": PROMPT}
                ]
            elif model_type == "default":
                message = [
                    {"role": "user", "content": chat},
                    {"role": "system", "content": PROMPT}
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