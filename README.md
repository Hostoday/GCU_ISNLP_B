# GCU_ISNLP_B

2024 국립국어원 인공지능 경진대회 모두의 말뭉치 일상대화요약 (가)유형 'GCU_ISNLP_E'팀

```
├── resource/
│   ├── data/
|   |   ├── sample.json
│   |   ├── 일상대화요약_dev.json
|   |   ├── 일상대화요약_test.json
|   |   ├── 일상대화요약_train.json
│   |   └── 일상대화요약_train+dev.json
│   └── model/
│   
├── run/
│   ├── big_concat.py
│   ├── concat.py
│   ├── ensemble_inf.py
│   ├── solar_test.py
|   ├── solar_train.py
|   ├── test.py
│   └── train.py
├── src/
│   ├── arg_parser.py
│   ├── concat_data.py
│   ├── data.py
│   ├── ens_data.py
│   ├── monn_data.py
│   ├── rerank_data.py
│   └── utils.py
├── environment.yaml
├── total_sh_file.sh
├── train.sh
└── README.md
```
### Backbone model(candidate summary generator)
- **maywell/EXAONE-3.0-7.8B-Instruct-Llamafied** : EXAONE 은 -3.0-7.8B-Instruct 78억개의 매개변
수를 가진 모델로 기존 모델의 학습 데이터보다 4배 이상 많은 양의 데이터를 사용하여 사전학습되고 명
령어 조정된 생성 모델이다 또한 . inference 기존 모델보다 시간이나 메모리 사용량이 56%, 35% 감소하
였으며 한국어와 영어만 학습시켜 한국어에서 뛰어난 성능을 보인다 이런 . 모델을 보다 쉽게 사용하기 위
해서 Llama의 모델 형식으로 변경하여서 사용한 것이다.

  https://huggingface.co/maywell/EXAONE-3.0-7.8B-Instruct-Llamafied

- **beomi/Llama-3-Open-Ko-8B** : Llama-3-Open-Ko-8B Llama-3-8B 모델은 를 기반으로 사전 학습된
언어 모델에 60GB 이상의 중복 제거된 텍스트가 포함된 공개적으로 이용 가능한 리소스를 사용하여 한
국어에 대해 훈련된 모델이다.

  https://huggingface.co/beomi/Llama-3-Open-Ko-8B

- ****mistralai/Mistral-Nemo-Instruct-2407****: Mistral AI NVIDIA 와 가 협력하여 구축한 12B의 매개변수를
가진 다국어 모델로써 기존의 Mistral-Nemo-Base-2407의 과는 다른 tokenizer Llama 를 사용하였으며
3 tokenizer와 비교했을 때 모든언어릐 약 85% text 에 대한 를 압축하여 더 성능이 뛰어남을 입증하였다.
  
  https://huggingface.co/mistralai/Mistral-Nemo-Instruct-2407

### Backbone model(re-ranking model)
- ****ddobokki/electra-small-sts-cross-encoder****: electra-small cross encoder 해당 모델은 모델을 로 사용
하여 KorSTS KLUE STS 와 를 학습시킨 모델임

  https://huggingface.co/ddobokki/electra-small-sts-cross-encoder

### Library
- python 3.9.19
- transformers 4.43.2
- trl 0.8.6
- xformers 0.0.26.post1
- torch 2.3.0
- PEFT 0.12.0
