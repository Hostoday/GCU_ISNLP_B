export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

# heegyu/llama-2-ko-7b-chat, MLP-KTLim/llama-3-Korean-Bllossom-8B, beomi/Solar-Ko-Recovery-11B, beomi/Llama-3-Open-Ko-8B

python -m run.train \
  --model_id beomi/Llama-3-Open-Ko-8B \
  --device cuda:0\
  --epoch 12\
  --lr 2e-5\
  --batch_size 1\
  --wandb_project DCS_wandb \
  --wandb_entity DCS_2024 \
  --wandb_run_name llama_3_model_chat_other_prompt \
  --prompt_type mode_with_special_tokens_concat \
  --prompt "당신은 유능한 AI 어시스턴트입니다. 주어진 대화를 집중해서 읽고 토픽에 기반한 대화 내용을 요약해주세요. [speaker1], [speaker2]로 화자를 구분합니다. 각 화자의 발화를 정확히 구분하고, 각 화자가 제시한 의견을 별도로 정리하며, 동일한 내용을 반복하지 않도록 주의하십시오. 각 화자의 이름과 그에 따른 내용을 명확히 매칭하여 요약해 주세요." \
  --trainer True \
  --model_type gemma \
  --gradient_accumulation_steps 10

python -m run.train \
  --model_id mistralai/Mistral-Nemo-Instruct-2407\
  --device cuda:0 \
  --epoch 12 \
  --lr 4e-5 \
  --batch_size 1 \
  --wandb_project DCS_wandb \
  --wandb_entity DCS_2024 \
  --wandb_run_name llama_3_model_chat_other_prompt \
  --prompt_type analysis_prompt_v3 \
  --prompt "당신은 유능한 AI 어시스턴트입니다. 주어진 대화에서 화자별로 주제에 기반하여 상세하게 요약해주세요." \
  --trainer True \
  --model_type gemma \
  --gradient_accumulation_steps 10

python -m run.solar_train \
  --model_id maywell/EXAONE-3.0-7.8B-Instruct-Llamafied \
  --device cuda:0 \
  --epoch 12 \
  --lr 3e-5 \
  --batch_size 1 \
  --wandb_project DCS_wandb \
  --wandb_entity DCS_2024 \
  --wandb_run_name llama_3_model_chat_other_prompt \
  --prompt_type mode_moon_co7 \
  --prompt "당신은 유능한 AI 어시스턴트입니다. 주어진 대화를 자세히 집중해서 읽고 토픽에 기반한 대화 내용을 요약해주세요. [speaker1], [speaker2]로 화자를 구분합니다. 각 화자의 발화를 정확히 구분하고, 각 화자가 제시한 의견을 각자 정리하며, 동일한 내용을 반복하지 않도록 주의하십시오. 각 화자의 이름과 그에 따른 내용을 명확히 매칭하여 요약해 주세요." \
  --trainer True \
  --gradient_accumulation_steps 10

python -m run.solar_train \
  --model_id maywell/EXAONE-3.0-7.8B-Instruct-Llamafied \
  --device cuda:0 \
  --epoch 12 \
  --lr 4e-5 --batch_size 1 \
  --wandb_project DCS_wandb \
  --wandb_entity DCS_2024 \
  --wandb_run_name llama_3_model_chat_other_prompt \
  --prompt_type make_chat_with_special_tokens \
  --prompt "당신은 유능한 AI 어시스턴트입니다. 주어진 대화에서 화자별로 주제에 기반하여 자세히 요약해주세요." \
  --trainer True \
  --gradient_accumulation_steps 10