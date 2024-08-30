python -m run.test \
   --model_id maywell/EXAONE-3.0-7.8B-Instruct-Llamafied\
   --device 'cuda:0' \
   --batch_size 1\
   --model_ckpt_path 'hojunssss/2024-08-18-10-38-630' \
   --prompt "대화에서 다루어진 주요 주제들을 중심으로 각 화자의 의견과 선호를 구체적인 키워드를 사용하여 요약하십시오. 대화의 맥락을 반영하여, 각 화자가 강조한 감정이나 중요하게 생각한 요소들을 포함시키세요. **원문에서 사용된 핵심 단어와 구문을 최대한 유지**하면서, 간결하고 명확하게 요약하십시오. 모든 중요한 정보를 빠짐없이 포함하도록 주의하고, 요약은 원문의 내용을 충실히 반영해야 합니다." \
   --model_type gemma\
   --prompt_type analysis_prompt_v2_se2 \
   --output ./힘쎈고양이야4-4.json\

python -m run.test \
   --model_id maywell/EXAONE-3.0-7.8B-Instruct-Llamafied\
   --device 'cuda:0' \
   --batch_size 1\
   --model_ckpt_path 'hojunssss/2024-08-18-10-38-630' \
   --prompt "요약문에서 등장하는 주요 주제들을 중심으로 각 화자의 의견과 선호를 *구체적인 키워드*를 사용하여 요약하세요. 대화의 맥락을 반영하여 각 화자가 표현한 감정이나 중요하게 생각한 요소들을 포함하도록 하세요. **원문에서 사용된 주요 단어와 구문을 최대한 유지**하면서, 간결하고 명확하게 요약하십시오. 모든 주요 정보를 놓치지 않도록 주의하십시오." \
   --model_type gemma\
   --prompt_type analysis_prompt_v2_se2\
   --output ./힘쎈고양이야4-2.json\

python -m run.test \
   --model_id maywell/EXAONE-3.0-7.8B-Instruct-Llamafied\
   --device 'cuda:0' \
   --batch_size 1\
   --model_ckpt_path 'hojunssss/2024-08-18-10-38-630' \
   --prompt "요약문에서 등장하는 주요 주제들을 중심으로 각 화자의 의견과 선호를 *구체적인 키워드*를 사용하여 요약하세요. 대화의 맥락을 반영하여 각 화자가 표현한 감정이나 중요하게 생각한 요소들을 포함하도록 하세요. **원문에서 사용된 주요 단어와 구문을 최대한 유지**하면서, 간결하고 명확하게 요약하십시오. 모든 주요 정보를 놓치지 않도록 주의하십시오." \
   --model_type gemma\
   --prompt_type analysis_prompt_v2_se\
   --output ./힘쎈고양이야3-2.json\

python -m run.test \
   --model_id maywell/EXAONE-3.0-7.8B-Instruct-Llamafied\
   --device 'cuda:0' \
   --batch_size 1\
   --model_ckpt_path 'hojunssss/2024-08-18-10-38-630' \
   --prompt "요약문들에서 등장하는 주요 주제들을 중심으로 각 화자의 의견과 선호를 요약하세요. 대화의 맥락을 반영하여 각 화자가 표현한 감정이나 중요하게 생각한 요소들을 포함하도록 하세요. 가능한 한 원문과 일치하는 표현을 사용하되, 간결하고 명확하게 요약하십시오." \
   --model_type gemma\
   --prompt_type analysis_prompt_v2_se\
   --output ./힘쎈고양이야3.json\
   # 새로운 prompt_type, 태규님 prompt
   # 레즈언드~

python -m run.test \
    --model_id beomi/Llama-3-Open-Ko-8B \
    --device 'cuda:0' \
    --batch_size 1\
    --model_ckpt_path 'hojunssss/2024-08-07-12-22-580' \
    --prompt "주어진 대화를 [speaker1], [speaker2]로 화자를 구분하여 간결하고 정확하게 요약하세요. 요약 시 다음의 평가 지표에 집중하십시오:<Rouge-1 점수>: 중요한 단어와 구문을 잘 포착. <BERT 점수>: 의미적 유사성 유지.<BLEURT 점수>: 전반적 맥락과 분위기 반영. 반복을 피하고, 각 화자의 의견을 정확히 구분하여 세 줄 이상으로 요약하세요." \
    --prompt_type analysis_prompt_v3 \
    --file_type inf\
    --output './기대되는고양이2-3.json'\
    --model_type gemma
    # 굳!! 개굳!!

python -m run.test \
   --model_id maywell/EXAONE-3.0-7.8B-Instruct-Llamafied\
   --device 'cuda:0' \
   --batch_size 1\
   --model_ckpt_path 'hojunssss/2024-08-18-10-38-630' \
   --prompt "대화에서 등장하는 주요 주제들을 중심으로 각 화자의 의견과 선호를 요약하세요. 대화의 맥락을 반영하여 각 화자가 표현한 감정이나 중요하게 생각한 요소들을 포함하도록 하세요. 가능한 한 원문과 일치하는 표현을 사용하되, 간결하고 명확하게 요약하십시오. 프롬프트 설계 요소: 주제 중심: 대화의 주요 주제(여행 스타일, 숙소, 음식 등)를 강조합니다. 의견과 선호: 각 화자의 의견과 선호를 명확하게 반영하도록 유도합니다. 감정과 중요 요소 반영: 대화에서 감지되는 감정이나 강조된 부분을 포함합니다. 원문 일치도: 원문과의 일치도를 높이기 위해 유사한 표현을 사용하되, 간결하고 명확하게 작성합니다. " \
   --model_type gemma\
   --prompt_type analysis_tg3\
   --output ./L두리안1.json\

python -m run.test \
   --model_id maywell/EXAONE-3.0-7.8B-Instruct-Llamafied\
   --device 'cuda:0' \
   --batch_size 1\
   --model_ckpt_path 'hojunssss/2024-08-18-10-38-630' \
   --prompt "이번 요약 업무에서, 당신은 주어진 대화를 간결하고 정확하게 요약하는 역할을 필히 잘 수행해야됩니다. 주어진 대화를 [speaker1], [speaker2]로 화자를 구분하여 간결하고 정확하게 요약하세요. 요약 시 다음의 평가 지표에 집중하십시오:<Rouge-1 점수>: 중요한 단어와 구문을 잘 포착. <BERT 점수>: 의미적 유사성 유지.<BLEURT 점수>: 전반적 맥락과 분위기 반영. 반복을 피하고, 각 화자의 의견을 정확히 구분하여 두 줄 이상으로 요약하세요." \
   --model_type gemma\
   --prompt_type analysis_prompt_v2\
   --output ./힘쎈고양이야1-2.json\
   # 진짜 개.쎔

python -m run.test \
    --model_id beomi/Llama-3-Open-Ko-8B \
    --device 'cuda:0'\
    --batch_size 1\
    --model_ckpt_path 'hojunssss/2024-08-07-12-22-580' \
    --prompt "안녕하세요! 이번 요약 업무에서, 당신은 주어진 대화를 간결하고 정확하게 요약하는 역할을 맡습니다. 다음의 주요한 평가 지표과 특히 '높은 BLEURT 지수'를 위하여 작업을 수행해주세요. 평가 지표는 다음과 같습니다. [Rouge-1 점수]: 요약문이 원본 대화에서 중요한 단어와 구문을 얼마나 잘 포착했는지를 평가. 핵심적인 정보를 잘 요약할수록 높은 점수를 받음. [BERT 점수]: 요약문이 원본 대화와 의미적으로 얼마나 유사한지를 측정. 단어 선택뿐만 아니라 전체적 의미와 맥락을 잘 유지하는 것이 중요. [BLEURT 점수]: 원본 대화와 요약문 간의 유사성을 평가. 높은 BLEURT 점수를 얻기 위해서는 원본 대화의 주요 내용과 전반적인 맥락의 분위기를 잘 반영. [주의사항]: [speaker1], [speaker2]로 화자를 구분하며, 해당 대화 스크립트를 먼저 시작하는 화자를 [speaker1]로 간주합니다. 대화 순서에 특히나 유의하여, 각 화자의 발화를 정확히 구분하고, 각 화자가 제시한 의견을 별도로 정리해주세요. 동일한 내용을 반복하지 않도록 특히나 주의하십시오. 위의 고려해야될 세 가지의 평가지표와 주의사항에 집중하여, 주어진 대화의 핵심을 잘 파악하여 정확하면서도 세 줄 이상으로 요약해 주세요. 이 업무에 최선을 다해 주세요. 당신의 능력을 기대합니다!" \
    --prompt_type analysis_prompt_v2\
    --file_type inf\
    --output './하하.json'\
    --model_type gemma 

python -m run.test \
    --model_id beomi/Llama-3-Open-Ko-8B \
    --device 'cuda:0' \
    --batch_size 1\
    --model_ckpt_path 'hojunssss/2024-08-07-12-22-580' \
    --prompt "안녕하세요! 이번 요약 업무에서, 당신은 주어진 대화를 간결하고 정확하게 요약하는 역할을 맡습니다. 다음의 주요한 평가 지표과 특히 '높은 BLEURT 지수'를 위하여 작업을 수행해주세요. 평가 지표는 다음과 같습니다. [Rouge-1 점수]: 이 점수는 요약문이 원본 대화에서 중요한 단어와 구문을 얼마나 잘 포착했는지를 평가합니다. 핵심적인 정보를 잘 요약할수록 높은 점수를 받을 수 있습니다. [BERT 점수]: 이 점수는 요약문이 원본 대화와 의미적으로 얼마나 유사한지를 측정합니다. 단어 선택뿐만 아니라 전체적인 의미와 맥락을 잘 유지하는 것이 중요합니다. [BLEURT 점수]: 이 점수는 원본 대화와 요약문 간의 유사성을 평가합니다. 높은 BLEURT 점수를 얻기 위해서는 원본 대화의 주요 내용과 전반적인 맥락의 분위기를 잘 반영해야 합니다. [주의사항]: [speaker1], [speaker2]로 화자를 구분하며, 해당 대화 스크립트를 먼저 시작하는 화자를 [speaker1]로 간주합니다. 대화 순서에 특히나 유의하여, 각 화자의 발화를 정확히 구분하고, 각 화자가 제시한 의견을 별도로 정리해주세요. 동일한 내용을 반복하지 않도록 주의하십시오. 각 화자의 이름과 그에 따른 내용을 명확히 매칭하여 요약해 주세요. 위의 고려해야될 세 가지의 평가지표와 주의사항에 집중하여, 주어진 대화의 핵심을 잘 파악하여 정확하면서도 세 줄 이상으로 요약해 주세요. 이 업무에 최선을 다해 주세요. 당신의 능력을 기대합니다!" \
    --prompt_type analysis_prompt_v2 \
    --file_type inf\
    --output './상받을고양이again.json'\
    --model_type gemma

python -m run.test \
    --model_id beomi/Llama-3-Open-Ko-8B\
    --device 'cuda:0' \
    --batch_size 1\
    --model_ckpt_path 'hojunssss/2024-08-07-12-22-580' \
    --prompt "당신은 유능한 AI 어시스턴트입니다. 주어진 대화를 집중해서 읽고 토픽에 기반한 대화 내용을 요약해주세요. [speaker1], [speaker2]로 화자를 구분합니다. 각 화자의 발화를 정확히 구분하고, 각 화자가 제시한 의견을 별도로 정리하며, 동일한 내용을 반복하지 않도록 주의하십시오. 각 화자의 이름과 그에 따른 내용을 명확히 매칭하여 요약해 주세요. 꼭 한국어로만 답변해주세요" \
    --prompt_type mode_moon_co6 \
    --model_type gemma\
    --file_type inf\
    --output './pasad3_v2.json'

python -m run.test \
    --model_id beomi/Llama-3-Open-Ko-8B\
    --device 'cuda:0' \
    --batch_size 1\
    --model_ckpt_path 'hojunssss/2024-08-07-12-22-580' \
    --prompt "당신은 유능한 AI 어시스턴트입니다. 주어진 대화를 집중해서 읽고 토픽에 기반한 대화 내용을 요약해주세요. [speaker1], [speaker2]로 화자를 구분합니다. 각 화자의 발화를 정확히 구분하고, 각 화자가 제시한 의견을 별도로 정리하며, 동일한 내용을 반복하지 않도록 주의하십시오. 각 화자의 이름과 그에 따른 내용을 명확히 매칭하여 요약해 주세요. 꼭 한국어로만 답변해주세요" \
    --prompt_type mode_moon_co5 \
    --model_type gemma\
    --file_type inf\
    --output './pasad33_v2.json'

python -m run.test \
   --model_id maywell/EXAONE-3.0-7.8B-Instruct-Llamafied\
   --device 'cuda:0' \
   --batch_size 1\
   --model_ckpt_path 'hojunssss/2024-08-18-10-38-630' \
   --prompt "대화에서 등장하는 주요 주제들을 중심으로 각 화자의 의견과 선호를 요약하세요. 대화의 맥락을 반영하여 각 화자가 표현한 감정이나 중요하게 생각한 요소들을 포함하도록 하세요. 가능한 한 원문과 일치하는 표현을 사용하되, 간결하고 명확하게 요약하십시오. 프롬프트 설계 요소: 주제 중심: 대화의 주요 주제(여행 스타일, 숙소, 음식 등)를 강조합니다. 의견과 선호: 각 화자의 의견과 선호를 명확하게 반영하도록 유도합니다. 감정과 중요 요소 반영: 대화에서 감지되는 감정이나 강조된 부분을 포함합니다. 원문 일치도: 원문과의 일치도를 높이기 위해 유사한 표현을 사용하되, 간결하고 명확하게 작성합니다. " \
   --model_type gemma\
   --prompt_type mode_moon_co5\
   --output ./L문1.json\

python -m run.test \
   --model_id maywell/EXAONE-3.0-7.8B-Instruct-Llamafied\
   --device 'cuda:0' \
   --batch_size 1\
   --model_ckpt_path 'hojunssss/2024-08-18-10-38-630' \
   --prompt "대화에서 등장하는 주요 주제들을 중심으로 각 화자의 의견과 선호를 요약하세요. 대화의 맥락을 반영하여 각 화자가 표현한 감정이나 중요하게 생각한 요소들을 포함하도록 하세요. 가능한 한 원문과 일치하는 표현을 사용하되, 간결하고 명확하게 요약하십시오. 프롬프트 설계 요소: 주제 중심: 대화의 주요 주제(여행 스타일, 숙소, 음식 등)를 강조합니다. 의견과 선호: 각 화자의 의견과 선호를 명확하게 반영하도록 유도합니다. 감정과 중요 요소 반영: 대화에서 감지되는 감정이나 강조된 부분을 포함합니다. 원문 일치도: 원문과의 일치도를 높이기 위해 유사한 표현을 사용하되, 간결하고 명확하게 작성합니다. " \
   --model_type gemma\
   --prompt_type mode_moon_co3\
   --output ./L문2.json\

python -m run.test \
    --model_id mistralai/Mistral-Nemo-Instruct-2407\
    --device 'cuda:0' \
    --batch_size 1\
    --model_ckpt_path 'hojunssss/2024-08-08-15-15-580' \
    --prompt "당신은 유능한 AI 어시스턴트입니다. 주어진 대화를 집중해서 읽고 토픽에 기반한 대화 내용을 요약해주세요. [speaker1], [speaker2]로 화자를 구분합니다. 각 화자의 발화를 정확히 구분하고, 각 화자가 제시한 의견을 별도로 정리하며, 동일한 내용을 반복하지 않도록 주의하십시오. 각 화자의 이름과 그에 따른 내용을 명확히 매칭하여 요약해 주세요. 화자를 헷갈려서 요약하지 말아주세요" \
    --prompt_type mode_moon_v2 \
    --model_type gemma\
    --file_type inf\
    --output './yooyoyoyo.json'

 python -m run.test \
     --model_id maywell/EXAONE-3.0-7.8B-Instruct-Llamafied\
     --device 'cuda:0' \
     --batch_size 1\
     --model_ckpt_path 'hojunssss/2024-08-09-05-51-600' \
     --prompt "당신은 유능한 AI 어시스턴트입니다. 주어진 대화를 집중해서 읽고 토픽에 기반한 대화 내용을 요약해주세요. [speaker1], [speaker2]로 화자를 구분합니다. 각 화자의 발화를 정확히 구분하고, 각 화자가 제시한 의견을 별도로 정리하며, 동일한 내용을 반복하지 않도록 주의하십시오. 각 화자의 이름과 그에 따른 내용을 명확히 매칭하여 요약해 주세요." \
     --prompt_type mode_with_special_tokens_concat  \
     --file_type inf\
     --output './만년2인자.json'

python -m run.test \
   --model_id maywell/EXAONE-3.0-7.8B-Instruct-Llamafied\
   --device 'cuda:0' \
   --batch_size 1\
   --model_ckpt_path 'hojunssss/2024-08-18-10-38-630' \
   --prompt "사용자의 대화에서 발생한 주요 사건을 요약하세요. 사건의 진행 과정과 중요한 결과를 포함하여 간결하게 요약하십시오 " \
   --model_type gemma\
   --prompt_type analysis_tg3\
   --output ./L메로나3.json\

python -m run.test \
    --model_id beomi/Llama-3-Open-Ko-8B \
    --device 'cuda:0' \
    --batch_size 1\
    --model_ckpt_path 'hojunssss/2024-08-07-12-22-580' \
    --prompt "주어진 대화를 두 화자를 정확히 구분하여 간결하고 정확하게 요약하세요. 요약 시 다음의 평가 지표에 집중하십시오:<Rouge-1 점수>: 중요한 단어와 구문을 잘 포착. <BERT 점수>: 의미적 유사성 유지.<BLEURT 점수>: 전반적 맥락과 분위기 반영. 반복을 피하고, 각 화자의 의견을 정확히 구분하여 세 줄 이상으로 요약하세요." \
    --prompt_type analysis_prompt_v2 \
    --file_type inf\
    --output './기대되는고양이2.json'\
    --model_type gemma
    # 레전드 우 >^< 으으으

python -m run.test \
    --model_id beomi/Llama-3-Open-Ko-8B \
    --device 'cuda:0'\
    --batch_size 1\
    --model_ckpt_path 'hojunssss/2024-08-07-12-22-580' \
    --prompt "주어진 대화를 [speaker1], [speaker2]로 화자를 구분하여 간결하고 정확하게 요약하세요. 요약 시 다음의 평가 지표에 집중하십시오:<Rouge-1 점수>: 중요한 단어와 구문을 잘 포착. <BERT 점수>: 의미적 유사성 유지.<BLEURT 점수>: 전반적 맥락과 분위기 반영. 반복을 피하고, 각 화자의 의견을 정확히 구분하여 세 줄 이상으로 요약하세요." \
    --prompt_type analysis_prompt_v2\
    --file_type inf\
    --output './웃는고양.json'\
    --model_type gemma 

python -m run.test \
    --model_id beomi/Llama-3-Open-Ko-8B \
    --device 'cuda:0'\
    --batch_size 1\
    --model_ckpt_path 'hojunssss/2024-08-07-12-22-580' \
    --prompt "주어진 대화에 대해 화자를 구분하여 간결하고 정확하게 요약하세요. 요약 시 다음의 평가 지표에 집중하십시오:<Rouge-1 점수>: 중요한 단어와 구문을 잘 포착. <BERT 점수>: 의미적 유사성 유지.<BLEURT 점수>: 전반적 맥락과 분위기 반영. 반복을 피하고, 각 화자의 의견을 정확히 구분하여 세 줄 이상으로 요약하세요." \
    --prompt_type analysis_prompt_v2\
    --file_type inf\
    --output './웃는고양2.json'\
    --model_type gemma 


python -m run.test \
   --model_id maywell/EXAONE-3.0-7.8B-Instruct-Llamafied\
   --device 'cuda:0' \
   --batch_size 1\
   --model_ckpt_path 'hojunssss/2024-08-18-10-38-630' \
   --prompt "대화에서 다루어진 주요 주제들을 중심으로 각 화자의 의견과 선호를 구체적인 키워드를 사용하여 요약하십시오. 대화의 맥락을 반영하여, 각 화자가 강조한 감정이나 중요하게 생각한 요소들을 포함시키세요. **원문에서 사용된 핵심 단어와 구문을 최대한 유지**하면서, 간결하고 명확하게 요약하십시오. 모든 중요한 정보를 빠짐없이 포함하도록 주의하고, 요약은 원문의 내용을 충실히 반영해야 합니다." \
   --model_type gemma\
   --prompt_type analysis_prompt_v2_se4 \
   --output ./복딩이.json\

python -m run.concat \
    --model_id maywell/EXAONE-3.0-7.8B-Instruct-Llamafied\
    --device 'cuda:0' \
    --batch_size 1\
    --model_ckpt_path 'hojunssss/2024-08-18-10-38-630' \
    --prompt "이번 요약 업무에서, 당신은 주어진 대화를 간결하고 정확하게 요약하는 역할을 필히 잘 수행해야됩니다. 주어진 대화를 [speaker1], [speaker2]로 화자를 구분하여 간결하고 정확하게 요약하세요. 요약 시 다음의 평가 지표에 집중하십시오:<Rouge-1 점수>: 중요한 단어와 구문을 잘 포착. <BERT 점수>: 의미적 유사성 유지.<BLEURT 점수>: 전반적 맥락과 분위기 반영. 반복을 피하고, 각 화자의 의견을 정확히 구분하여 두 줄 이상으로 요약하세요." \
    --model_type gemma\
    --prompt_type analysis_prompt_v3 \
    --output ./웃쨔고양아.json\
    --concat_file_list "/inference/기대되는고양이2-3.json" "/inference/하하.json"

python -m run.concat \
    --model_id maywell/EXAONE-3.0-7.8B-Instruct-Llamafied\
    --device 'cuda:0' \
    --batch_size 1\
    --model_ckpt_path 'hojunssss/2024-08-18-10-38-630' \
    --prompt "요약문에서 등장하는 주요 주제들을 중심으로 각 화자의 의견과 선호를 *구체적인 키워드*를 사용하여 요약하세요. 대화의 맥락을 반영하여 각 화자가 표현한 감정이나 중요하게 생각한 요소들을 포함하도록 하세요. **원문에서 사용된 주요 단어와 구문을 최대한 유지**하면서, 간결하고 명확하게 요약하십시오. 모든 주요 정보를 놓치지 않도록 주의하십시오." \
    --model_type gemma\
    --prompt_type analysis_prompt_v2_se2 \
    --output ./a츄르먹자2.json\
    --concat_file_list "/inference/힘쎈고양이야4-4.json" "/inference/힘쎈고양이야4-2.json" "/inference/하하.json" "/inference/고양이야.json" "/inference/상받을고양이again2.json" "/inference/힘쎈고양이야.json" "/inference/힘쎈고양이야1-2.json" "/inference/복딩이.json"

python -m run.concat \
    --model_id maywell/EXAONE-3.0-7.8B-Instruct-Llamafied\
    --device 'cuda:0' \
    --batch_size 1\
    --model_ckpt_path 'hojunssss/2024-08-18-10-38-630' \
    --prompt "요약문에서 등장하는 주요 주제들을 중심으로 각 화자의 의견과 선호를 *구체적인 키워드*를 사용하여 요약하세요. 대화의 맥락을 반영하여 각 화자가 표현한 감정이나 중요하게 생각한 요소들을 포함하도록 하세요. **원문에서 사용된 주요 단어와 구문을 최대한 유지**하면서, 간결하고 명확하게 요약하십시오. 모든 주요 정보를 놓치지 않도록 주의하십시오." \
    --model_type gemma\
    --prompt_type analysis_prompt_v2_se2 \
    --output ./굴먹는양푸니들.json\
    --concat_file_list "/inference/힘쎈고양이야4-2.json" "/inference/힘쎈고양이야1-2.json" "/inference/하하.json" "/inference/힘쎈고양이야4-4.json" "/inference/상받을고양이again.json"


#애옹 -->옹박사 예정
python -m run.concat \
    --model_id beomi/Llama-3-Open-Ko-8B\
    --device 'cuda:0' \
    --batch_size 1\
    --model_ckpt_path 'hojunssss/2024-08-07-12-22-580' \
    --prompt "당신은 유능한 AI 어시스턴트입니다. 사용자가 요청하는 부분을 정확하게 답변해주세요." \
    --model_type gemma\
    --prompt_type summary_concat_tg\
    --output ./애옹이랑라마랑.json\
    --concat_file_list "/inference/웃쨔고양아.json" "/inference/pasad3_v2.json" "/inference/pasad33_v2.json" "/inference/기대되는고양이2-3.json" "/inference/웃는고양.json" "/inference/L문1.json"

#고양이 모음 라마로돌림 -- 옹석사
python -m run.concat \
    --model_id beomi/Llama-3-Open-Ko-8B\
    --device 'cuda:0' \
    --batch_size 1\
    --model_ckpt_path 'hojunssss/2024-08-07-12-22-580' \
    --prompt "당신은 유능한 AI 어시스턴트입니다. 사용자가 요청하는 부분을 정확하게 답변해주세요." \
    --model_type gemma\
    --output ./애옹이랑라마랑2.json\
    --concat_file_list "/inference/pasad3_v2.json" "/inference/pasad33_v2.json" "/inference/기대되는고양이2.json" "/inference/기대되는고양이2-3.json" "/inference/웃는고양.json"

#애옹이 코로나제거 --> SI
python -m run.concat \
    --model_id maywell/EXAONE-3.0-7.8B-Instruct-Llamafied\
    --device 'cuda:0' \
    --batch_size 1\
    --model_ckpt_path 'hojunssss/2024-08-18-10-38-630' \
    --prompt "요약문들에서 등장하는 주요 주제들을 중심으로 각 화자의 의견과 선호를 요약하세요. 대화의 맥락을 반영하여 각 화자가 표현한 감정이나 중요하게 생각한 요소들을 포함하도록 하세요. 가능한 한 원문과 일치하는 표현을 사용하되, 간결하고 명확하게 요약하십시오." \
    --model_type gemma\
    --prompt_type summary_concat_tg\
    --output ./애옹이코로나제거에스까르.json\
    --concat_file_list "/inference/힘쎈고양이야4-2.json" "/inference/웃쨔고양아.json" "/inference/pasad3_v2.json" "/inference/웃는고양2.json" "/inference/L문1.json"


python -m run.concat \
    --model_id maywell/EXAONE-3.0-7.8B-Instruct-Llamafied\
    --device 'cuda:0' \
    --batch_size 1\
    --model_ckpt_path 'hojunssss/2024-08-18-10-38-630' \
    --prompt "요약문들에서 등장하는 주요 주제들을 중심으로 각 화자의 의견과 선호를 요약하세요. 대화의 맥락을 반영하여 각 화자가 표현한 감정이나 중요하게 생각한 요소들을 포함하도록 하세요. 가능한 한 원문과 일치하는 표현을 사용하되, 간결하고 명확하게 요약하십시오." \
    --model_type gemma\
    --prompt_type summary_concat_tg\
    --output ./애옹이.json\
    --concat_file_list "/inference/웃쨔고양아.json" "/inference/pasad3_v2.json" "/inference/pasad33_v2.json" "/inference/기대되는고양이2-3.json" "/inference/웃는고양.json" "/inference/L문1.json"

#애옹이 업그레이드 --> 애옹애옹
python -m run.concat \
    --model_id maywell/EXAONE-3.0-7.8B-Instruct-Llamafied\
    --device 'cuda:0' \
    --batch_size 1\
    --model_ckpt_path 'hojunssss/2024-08-18-10-38-630' \
    --prompt "요약문들에서 등장하는 주요 주제들을 중심으로 각 화자의 의견과 선호를 요약하세요. 대화의 맥락을 반영하여 각 화자가 표현한 감정이나 중요하게 생각한 요소들을 포함하도록 하세요. 가능한 한 원문과 일치하는 표현을 사용하되, 간결하고 명확하게 요약하십시오." \
    --model_type gemma\
    --prompt_type summary_concat_tg\
    --output ./애옹쓰.json\
    --concat_file_list "/inference/애옹이.json" "/inference/L문2.json" "/inference/L메로나3.json" "/inference/yooyoyoyo.json" "/inference/만년2인자.json"

#고양이 합친거 점수올리기 --> 옹옹옹
python -m run.concat \
    --model_id beomi/Llama-3-Open-Ko-8B\
    --device 'cuda:0' \
    --batch_size 1\
    --model_ckpt_path 'hojunssss/2024-08-07-12-22-580' \
    --prompt "당신은 유능한 AI 어시스턴트입니다. 사용자가 요청하는 부분을 정확하게 답변해주세요." \
    --model_type gemma\
    --output ./고양이집합.json\
    --concat_file_list  "/inference/애옹쓰.json" "/inference/애옹이.json" "/inference/애옹이랑라마랑.json" "/inference/애옹이랑라마랑2.json" "/inference/기대되는고양이2.json"

# 고양이 엘지로 합치기 --> ㅇㅅㅇ
python -m run.concat \
    --model_id maywell/EXAONE-3.0-7.8B-Instruct-Llamafied\
    --device 'cuda:0' \
    --batch_size 1\
    --model_ckpt_path 'hojunssss/2024-08-18-10-38-630' \
    --prompt "요약문들에서 등장하는 주요 주제들을 중심으로 각 화자의 의견과 선호를 요약하세요. 대화의 맥락을 반영하여 각 화자가 표현한 감정이나 중요하게 생각한 요소들을 포함하도록 하세요. 가능한 한 원문과 일치하는 표현을 사용하되, 간결하고 명확하게 요약하십시오." \
    --model_type gemma\
    --prompt_type summary_concat_tg\
    --output ./고양이손들고서있어.json\
    --concat_file_list "/inference/애옹쓰.json" "/inference/애옹이.json" "/inference/애옹이랑라마랑.json" "/inference/애옹이랑라마랑2.json" "/inference/기대되는고양이2.json"

#고양이로 1등 --> 혼나자
python -m run.concat \
    --model_id maywell/EXAONE-3.0-7.8B-Instruct-Llamafied\
    --device 'cuda:0' \
    --batch_size 1\
    --model_ckpt_path 'hojunssss/2024-08-18-10-38-630' \
    --prompt "요약문들에서 등장하는 주요 주제들을 중심으로 각 화자의 의견과 선호를 요약하세요. 대화의 맥락을 반영하여 각 화자가 표현한 감정이나 중요하게 생각한 요소들을 포함하도록 하세요. 가능한 한 원문과 일치하는 표현을 사용하되, 간결하고 명확하게 요약하십시오." \
    --model_type gemma\
    --prompt_type summary_concat_tg\
    --output ./고양이혼나자.json\
    --concat_file_list "/inference/애옹쓰.json" "/inference/고양이집합.json" "/inference/애옹이.json" "/inference/pasad3_v2.json" "/inference/고양이손들고서있어.json"

python -m run.concat \
    --model_id maywell/EXAONE-3.0-7.8B-Instruct-Llamafied\
    --device 'cuda:0' \
    --batch_size 1\
    --model_ckpt_path 'hojunssss/2024-08-18-10-38-630' \
    --prompt "요약문에서 등장하는 주요 주제들을 중심으로 각 화자의 의견과 선호를 *구체적인 키워드*를 사용하여 요약하세요. 대화의 맥락을 반영하여 각 화자가 표현한 감정이나 중요하게 생각한 요소들을 포함하도록 하세요. **원문에서 사용된 주요 단어와 구문을 최대한 유지**하면서, 간결하고 명확하게 요약하십시오. 모든 주요 정보를 놓치지 않도록 주의하십시오." \
    --model_type gemma\
    --prompt_type summary_concat_tg\
    --output ./성심당_1단망고_케이크.json\
    --concat_file_list "/inference/힘쎈고양이야4-2.json" "/inference/힘쎈고양이야3-2.json" "/inference/힘쎈고양이야3.json" "/inference/기대되는고양이2-3.json" "/inference/L두리안1.json" 

python -m run.concat \
    --model_id maywell/EXAONE-3.0-7.8B-Instruct-Llamafied\
    --device 'cuda:0' \
    --batch_size 1\
    --model_ckpt_path 'hojunssss/2024-08-18-10-38-630' \
    --prompt "요약문에서 등장하는 주요 주제들을 중심으로 각 화자의 의견과 선호를 *구체적인 키워드*를 사용하여 요약하세요. 대화의 맥락을 반영하여 각 화자가 표현한 감정이나 중요하게 생각한 요소들을 포함하도록 하세요. **원문에서 사용된 주요 단어와 구문을 최대한 유지**하면서, 간결하고 명확하게 요약하십시오. 모든 주요 정보를 놓치지 않도록 주의하십시오." \
    --model_type gemma\
    --prompt_type summary_concat_tg\
    --output ./성심당_2단두리안_케이크3.json\
    --concat_file_list "/inference/힘쎈고양이야4-4.json" "/inference/힘쎈고양이야4-2.json" "/inference/성심당_1단망고_케이크.json" 

python -m run.concat \
    --model_id maywell/EXAONE-3.0-7.8B-Instruct-Llamafied\
    --device 'cuda:0' \
    --batch_size 1\
    --model_ckpt_path 'hojunssss/2024-08-18-10-38-630' \
    --prompt "요약문에서 등장하는 주요 주제들을 중심으로 각 화자의 의견과 선호를 *구체적인 키워드*를 사용하여 요약하세요. 대화의 맥락을 반영하여 각 화자가 표현한 감정이나 중요하게 생각한 요소들을 포함하도록 하세요. **원문에서 사용된 주요 단어와 구문을 최대한 유지**하면서, 간결하고 명확하게 요약하십시오. 모든 주요 정보를 놓치지 않도록 주의하십시오." \
    --model_type gemma\
    --prompt_type analysis_prompt_v2_se2 \
    --output ./양푸니들.json\
    --concat_file_list "/inference/힘쎈고양이야4-2.json" "/inference/힘쎈고양이야1-2.json" "/inference/하하.json" "/inference/상받을고양이again.json"

python -m run.concat \
    --model_id maywell/EXAONE-3.0-7.8B-Instruct-Llamafied\
    --device 'cuda:0' \
    --batch_size 1\
    --model_ckpt_path 'hojunssss/2024-08-18-10-38-630' \
    --prompt "요약문에서 등장하는 주요 주제들을 중심으로 각 화자의 의견과 선호를 *구체적인 키워드*를 사용하여 요약하세요. 대화의 맥락을 반영하여 각 화자가 표현한 감정이나 중요하게 생각한 요소들을 포함하도록 하세요. **원문에서 사용된 주요 단어와 구문을 최대한 유지**하면서, 간결하고 명확하게 요약하십시오. 모든 주요 정보를 놓치지 않도록 주의하십시오." \
    --model_type gemma\
    --prompt_type summary_concat_tg\
    --output ./성심당_2단두리안_케이크4.json\
    --concat_file_list "/inference/양푸니들.json" "/inference/성심당_1단망고_케이크.json" "/inference/힘쎈고양이야4-2.json" "/inference/힘쎈고양이야4-4.json" 

#최강고양이 변경 돌려야햐 --> 족장 제출
python -m run.concat \
    --model_id maywell/EXAONE-3.0-7.8B-Instruct-Llamafied\
    --device 'cuda:0' \
    --batch_size 1\
    --model_ckpt_path 'hojunssss/2024-08-18-10-38-630' \
    --prompt "요약문에서 등장하는 주요 주제들을 중심으로 각 화자의 의견과 선호를 *구체적인 키워드*를 사용하여 요약하세요. 대화의 맥락을 반영하여 각 화자가 표현한 감정이나 중요하게 생각한 요소들을 포함하도록 하세요. **원문에서 사용된 주요 단어와 구문을 최대한 유지**하면서, 간결하고 명확하게 요약하십시오. 모든 주요 정보를 놓치지 않도록 주의하십시오." \
    --model_type gemma\
    --prompt_type summary_concat_tg\
    --output ./고양이족장.json\
    --concat_file_list "/inference/양푸니들.json" "/inference/성심당_1단망고_케이크.json" "/inference/애옹이코로나제거에스까르.json" "/inference/힘쎈고양이야4-4.json"

#최강고양이 -- 지켜
python -m run.concat \
    --model_id maywell/EXAONE-3.0-7.8B-Instruct-Llamafied\
    --device 'cuda:0' \
    --batch_size 1\
    --model_ckpt_path 'hojunssss/2024-08-18-10-38-630' \
    --prompt "요약문에서 등장하는 주요 주제들을 중심으로 각 화자의 의견과 선호를 *구체적인 키워드*를 사용하여 요약하세요. 대화의 맥락을 반영하여 각 화자가 표현한 감정이나 중요하게 생각한 요소들을 포함하도록 하세요. **원문에서 사용된 주요 단어와 구문을 최대한 유지**하면서, 간결하고 명확하게 요약하십시오. 모든 주요 정보를 놓치지 않도록 주의하십시오." \
    --model_type gemma\
    --prompt_type summary_concat_tg\
    --output ./고양이대장.json\
    --concat_file_list "/inference/양푸니들.json" "/inference/성심당_1단망고_케이크.json" "/inference/고양이혼나자.json" "/inference/힘쎈고양이야4-2.json"

python -m run.concat \
    --model_id beomi/Llama-3-Open-Ko-8B\
    --device 'cuda:0' \
    --batch_size 1\
    --model_ckpt_path 'hojunssss/2024-08-07-12-22-580' \
    --prompt "요약문에서 등장하는 주요 주제들을 중심으로 각 화자의 의견과 선호를 *구체적인 키워드*를 사용하여 요약하세요. 대화의 맥락을 반영하여 각 화자가 표현한 감정이나 중요하게 생각한 요소들을 포함하도록 하세요. **원문에서 사용된 주요 단어와 구문을 최대한 유지**하면서, 간결하고 명확하게 요약하십시오. 모든 주요 정보를 놓치지 않도록 주의하십시오." \
    --model_type gemma\
    --prompt_type analysis_prompt_v2_se2\
    --output ./파퀴아오_3단케이크2.json\
    --concat_file_list "/inference/굴먹는양푸니들.json" "/inference/힘쎈고양이야4-4.json" "/inference/성심당_2단두리안_케이크3.json" "/inference/애옹이코로나제거에스까르.json"

#높은애들 합쳐 --> 미오치치
python -m run.concat \
    --model_id maywell/EXAONE-3.0-7.8B-Instruct-Llamafied\
    --device 'cuda:0' \
    --batch_size 1\
    --model_ckpt_path 'hojunssss/2024-08-18-10-38-630' \
    --prompt "요약문에서 등장하는 주요 주제들을 중심으로 각 화자의 의견과 선호를 *구체적인 키워드*를 사용하여 요약하세요. 대화의 맥락을 반영하여 각 화자가 표현한 감정이나 중요하게 생각한 요소들을 포함하도록 하세요. **원문에서 사용된 주요 단어와 구문을 최대한 유지**하면서, 간결하고 명확하게 요약하십시오. 모든 주요 정보를 놓치지 않도록 주의하십시오." \
    --model_type gemma\
    --prompt_type summary_concat_tg\
    --output ./미오치치고양이.json\
    --concat_file_list "/inference/고양이대장.json" "/inference/고양이족장.json" "/inference/힘쎈고양이야4-4.json" "/inference/성심당_2단두리안_케이크3.json" "/inference/애옹이코로나제거에스까르.json"

#지켜 업그레이드 은가누고양이 돌려야함 --> 은가누
python -m run.concat \
    --model_id maywell/EXAONE-3.0-7.8B-Instruct-Llamafied\
    --device 'cuda:0' \
    --batch_size 1\
    --model_ckpt_path 'hojunssss/2024-08-18-10-38-630' \
    --prompt "요약문에서 등장하는 주요 주제들을 중심으로 각 화자의 의견과 선호를 *구체적인 키워드*를 사용하여 요약하세요. 대화의 맥락을 반영하여 각 화자가 표현한 감정이나 중요하게 생각한 요소들을 포함하도록 하세요. **원문에서 사용된 주요 단어와 구문을 최대한 유지**하면서, 간결하고 명확하게 요약하십시오. 모든 주요 정보를 놓치지 않도록 주의하십시오." \
    --model_type gemma\
    --prompt_type summary_concat_tg\
    --output ./은가누고양이.json\
    --concat_file_list "/inference/양푸니들.json" "/inference/성심당_1단망고_케이크.json" "/inference/힘쎈고양이야4-4.json" "/inference/힘쎈고양이야4-2.json" "/inference/애옹이코로나제거에스까르.json"

#이야 --> 야이
python -m run.ensemble_inf \
   --device 'cuda:0' \
   --output /이야.json\
   --concat_file_list "/inference/미오치치고양이.json" "/inference/고양이족장.json" "/inference/은가누고양이.json" "/inference/양푸니들.json" "/inference/힘쎈고양이야4-4.json"

#150 깔끔한 emsemble
python -m run.ensemble_inf \
   --device 'cuda:0' \
   --output /깔끔.json\
   --concat_file_list "/inference/굴먹는양푸니들.json" "/inference/힘쎈고양이야4-2.json" "/inference/L메로나3.json"

#ensemble 모여라 --> 옹교수
python -m run.concat \
    --model_id maywell/EXAONE-3.0-7.8B-Instruct-Llamafied\
    --device 'cuda:0' \
    --batch_size 1\
    --model_ckpt_path 'hojunssss/2024-08-18-10-38-630' \
    --prompt "요약문에서 등장하는 주요 주제들을 중심으로 각 화자의 의견과 선호를 *구체적인 키워드*를 사용하여 요약하세요. 대화의 맥락을 반영하여 각 화자가 표현한 감정이나 중요하게 생각한 요소들을 포함하도록 하세요. **원문에서 사용된 주요 단어와 구문을 최대한 유지**하면서, 간결하고 명확하게 요약하십시오. 모든 주요 정보를 놓치지 않도록 주의하십시오." \
    --model_type gemma\
    --prompt_type summary_concat_tg\
    --output ./옹교수.json\
    --concat_file_list "/inference/미오치치고양이.json" "/inference/이야.json" "/inference/깔끔.json"

python -m run.concat \
    --model_id maywell/EXAONE-3.0-7.8B-Instruct-Llamafied\
    --device 'cuda:0' \
    --batch_size 1\
    --model_ckpt_path 'hojunssss/2024-08-18-10-38-630' \
    --prompt "요약문에서 등장하는 주요 주제들을 중심으로 각 화자의 의견과 선호를 *구체적인 키워드*를 사용하여 요약하세요. 대화의 맥락을 반영하여 각 화자가 표현한 감정이나 중요하게 생각한 요소들을 포함하도록 하세요. **원문에서 사용된 주요 단어와 구문을 최대한 유지**하면서, 간결하고 명확하게 요약하십시오. 모든 주요 정보를 놓치지 않도록 주의하십시오." \
    --model_type gemma\
    --prompt_type summary_concat_tg\
    --output ./z살려.json\
    --concat_file_list "/inference/옹교수.json" "/inference/굴먹는양푸니들.json" "/inference/복딩이.json"

# 옹교수 + 굴먹는양푸니들 + 깔끔 (essemble한것들) -> 옹근 0 제출
python -m run.concat \
    --model_id maywell/EXAONE-3.0-7.8B-Instruct-Llamafied\
    --device 'cuda:0' \
    --batch_size 1\
    --model_ckpt_path 'hojunssss/2024-08-18-10-38-630' \
    --prompt "요약문에서 등장하는 주요 주제들을 중심으로 각 화자의 의견과 선호를 *구체적인 키워드*를 사용하여 요약하세요. 대화의 맥락을 반영하여 각 화자가 표현한 감정이나 중요하게 생각한 요소들을 포함하도록 하세요. **원문에서 사용된 주요 단어와 구문을 최대한 유지**하면서, 간결하고 명확하게 요약하십시오. 모든 주요 정보를 놓치지 않도록 주의하십시오." \
    --model_type gemma\
    --prompt_type summary_concat_tg\
    --output ./옹근0.json\
    --concat_file_list "/inference/옹교수.json" "/inference/굴먹는양푸니들.json" "/inference/깔끔.json"

# 옹근0 llama버전 -> 옹옹옹옹옹 -> bleurt good 46.26
python -m run.concat \
    --model_id beomi/Llama-3-Open-Ko-8B\
    --device 'cuda:0' \
    --batch_size 1\
    --model_ckpt_path 'hojunssss/2024-08-07-12-22-580' \
    --prompt "당신은 유능한 AI 어시스턴트입니다. 사용자가 요청하는 부분을 친절하고 정확하게 답변해주세요." \
    --model_type gemma\
    --output ./옹옹옹옹옹.json\
    --concat_file_list "/inference/옹교수.json" "/inference/굴먹는양푸니들.json" "/inference/깔끔.json"

#진짜 끝내자 옹근0 + 옹옹옹옹옹 + 옹옹이들 + 성심당_2단두리안_케이크4 --> 짬뽕 -->루지 높음
python -m run.concat \
    --model_id maywell/EXAONE-3.0-7.8B-Instruct-Llamafied\
    --device 'cuda:0' \
    --batch_size 1\
    --model_ckpt_path 'hojunssss/2024-08-18-10-38-630' \
    --prompt "요약문에서 등장하는 주요 주제들을 중심으로 각 화자의 의견과 선호를 *구체적인 키워드*를 사용하여 요약하세요. 대화의 맥락을 반영하여 각 화자가 표현한 감정이나 중요하게 생각한 요소들을 포함하도록 하세요. **원문에서 사용된 주요 단어와 구문을 최대한 유지**하면서, 간결하고 명확하게 요약하십시오. 모든 주요 정보를 놓치지 않도록 주의하십시오." \
    --model_type gemma\
    --prompt_type summary_concat_tg\
    --output ./z크으아.json\
    --concat_file_list "/inference/옹근0.json" "/inference/옹옹옹옹옹.json" "/inference/성심당_2단두리안_케이크4.json"

#doo 로 제출
python -m run.ensemble_inf \
   --device 'cuda:0' \
   --output /z도와줘.json\
   --concat_file_list "/inference/z살려.json" "/inference/복딩이.json" "/inference/힘쎈고양이야4-4.json"

# 라마로 짬뽕, 미오치치, 굴먹양푸니 합.. -> 하나로마트
python -m run.concat \
    --model_id beomi/Llama-3-Open-Ko-8B\
    --device 'cuda:0' \
    --batch_size 1\
    --model_ckpt_path 'hojunssss/2024-08-07-12-22-580' \
    --prompt "당신은 유능한 AI 어시스턴트입니다. 사용자가 요청하는 부분을 친절하고 정확하게 답변해주세요." \
    --model_type gemma\
    --output ./라마로마무리.json\
    --concat_file_list "/inference/z크으아.json" "/inference/미오치치고양이.json" "/inference/굴먹는양푸니들.json"

#총출동 prompt 변경 -> 별
python -m run.concat \
    --model_id maywell/EXAONE-3.0-7.8B-Instruct-Llamafied\
    --device 'cuda:0' \
    --batch_size 1\
    --model_ckpt_path 'hojunssss/2024-08-18-10-38-630' \
    --prompt "요약문에서 등장하는 주요 주제들을 중심으로 각 화자의 의견과 선호를 *구체적인 키워드*를 사용하여 요약하세요. 대화의 맥락을 반영하여 각 화자가 표현한 감정이나 중요하게 생각한 요소들을 포함하도록 하세요. **원문에서 사용된 주요 단어와 구문을 최대한 유지**하면서, 간결하고 명확하게 요약하십시오. 모든 주요 정보를 놓치지 않도록 주의하십시오." \
    --model_type gemma\
    --prompt_type summary_concat_tg\
    --output ./z총출동별.json\
    --concat_file_list "/inference/라마로마무리.json" "/inference/z도와줘.json" "/inference/굴먹는양푸니들.json"

python -m run.concat \
    --model_id maywell/EXAONE-3.0-7.8B-Instruct-Llamafied\
    --device 'cuda:0' \
    --batch_size 1\
    --model_ckpt_path 'hojunssss/2024-08-18-10-38-630' \
    --prompt "당신은 유능한 AI 어시스턴트입니다. 사용자가 요청하는 부분을 친절하고 정확하게 답변해주세요." \
    --model_type gemma\
    --prompt_type summary_concat_tg\
    --output ./z총출동.json\
    --concat_file_list "/inference/z도와줘.json" "/inference/라마로마무리.json" "/inference/굴먹는양푸니들.json"

# 총출동 앙상블 -> 지구
python -m run.ensemble_inf \
   --device 'cuda:0' \
   --output /총총출동.json\
   --concat_file_list "/inference/z총출동.json" "/inference/z총출동별.json"

python -m run.ensemble_inf \
   --device 'cuda:0' \
   --output /직화장인.json\
   --concat_file_list "/inference/총총출동.json" "/inference/굴먹는양푸니들.json" "/inference/고양이대장.json"

# 제출완
python -m run.ensemble_inf \
   --device 'cuda:0' \
   --output /방이돈가4.json\
   --concat_file_list "/inference/z총출동.json" "/inference/굴먹는양푸니들.json" "/inference/직화장인.json" #"/inference/양푸니들.json" "/inference/a츄르먹자2.json" 
# # 

# 제출완
python -m run.ensemble_inf \
   --device 'cuda:0' \
   --output /방이돈가8.json\
   --concat_file_list "/inference/z총출동.json" "/inference/직화장인.json" # "/inference/직화장인.json" #"/inference/양푸니들.json" "/inference/a츄르먹자2.json" 

# 제출완
python -m run.ensemble_inf \
   --device 'cuda:0' \
   --output /방이돈가2.json\
   --concat_file_list "/inference/총총출동.json" "/inference/굴먹는양푸니들.json" "/inference/직화장인.json" #"/inference/양푸니들.json"
# 60.65

# 제출완
python -m run.ensemble_inf \
   --device 'cuda:0' \
   --output /방이돈가3.json\
   --concat_file_list "/inference/총총출동.json" "/inference/굴먹는양푸니들.json" "/inference/직화장인.json" #"/inference/양푸니들.json" "/inference/a츄르먹자2.json" 
# # 

python -m run.ensemble_inf \
   --device 'cuda:0' \
   --output /트럼프합체.json\
   --concat_file_list "/inference/z총출동.json" "/inference/방이돈가8.json" "/inference/굴먹는양푸니들.json"

python -m run.concat \
    --model_id beomi/Llama-3-Open-Ko-8B\
    --device 'cuda:0' \
    --batch_size 1\
    --model_ckpt_path 'hojunssss/2024-08-07-12-22-580' \
    --prompt "당신은 유능한 AI 어시스턴트입니다. 사용자가 요청하는 부분을 친절하고 정확하게 답변해주세요." \
    --model_type gemma\
    --prompt_type summary_concat_tg\
    --output ./볼카노프스키_4단케이크.json\
    --concat_file_list "/inference/방이돈가2.json" "/inference/직화장인.json" "/inference/방이돈가3.json" "/inference/방이돈가4.json"

python -m run.concat \
    --model_id maywell/EXAONE-3.0-7.8B-Instruct-Llamafied\
    --device 'cuda:0' \
    --batch_size 1\
    --model_ckpt_path 'hojunssss/2024-08-18-10-38-630' \
    --prompt "당신은 유능한 AI 어시스턴트입니다. 사용자가 요청하는 부분을 친절하고 정확하게 답변해주세요." \
    --model_type gemma\
    --prompt_type summary_concat_tg\
    --output ./트럼프저격수.json\
    --concat_file_list "/inference/파퀴아오_3단케이크2.json" "/inference/트럼프합체.json" "/inference/볼카노프스키_4단케이크.json"



