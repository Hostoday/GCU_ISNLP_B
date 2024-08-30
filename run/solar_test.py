
import argparse
import json
from tqdm import tqdm

import torch
import numpy
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from src.utils import set_random_seed

from src.data import CustomDataset, DataCollatorForSupervisedDataset, DataCollatorForInferenceDataset
from peft import PeftModel, PeftConfig
import os


# fmt: off
parser = argparse.ArgumentParser(prog="test", description="Testing about Conversational Context Inference.")

g = parser.add_argument_group("Common Parameter")
g.add_argument("--output", type=str, required=True, help="output filename")
g.add_argument("--model_id", type=str, required=True, help="huggingface model id")
g.add_argument("--tokenizer", type=str, help="huggingface tokenizer")
g.add_argument("--device", type=str, required=True, help="device to load the model")
g.add_argument("--model_ckpt_path", type=str, required=True, help="model checkpoint path")
g.add_argument("--prompt_type", type=str, default='mode_with_special_tokens', help="prompt type")
g.add_argument("--batch_size", type=int, default=2, help="batch size")
g.add_argument("--model_type", type=str, default="default", help="model is gemma?")
g.add_argument("--file_type",type=str,default="inf",help="file type")
g.add_argument("--prompt",type=str,default = "You are a helpful AI assistant. Please summarize the main topics of the user's conversation. 당신은 유능한 AI 어시스턴트입니다. 사용자의 대화를 통해 주요 주제에 대해 요약해주세요.")
g.add_argument("--num_beams",type=int,default=1,help="beam-search")
g.add_argument("--do_sample",type=bool,default=False,help="do_sample")
g.add_argument("--no_repeat_ngram_size",type=int,default=15,help="repeat ngram size")
g.add_argument("--top_p", type = float, default = 0.94, help = "top-p inference")
g.add_argument("--top_k", type = int, default = 50, help = "top-k inference")
g.add_argument("--seed", type = int, default = 42, help = "random seed") #3209512016602887751 % (2**32)
# fmt: on
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def main(args):
    current_seed = torch.initial_seed()
    print(f"Current seed: {current_seed}")
    set_random_seed(args.seed)

    if args.tokenizer == None:
        args.tokenizer = args.model_id
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    # Set padding_side to left
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>"),
        #tokenizer.convert_tokens_to_ids("</s>")
        #tokenizer.convert_tokens_to_ids("<|eos|>")
        tokenizer.convert_tokens_to_ids("<|end_of_text|>")
    ]

    dataset = CustomDataset("resource/data/일상대화요약_test.json", tokenizer,  args.prompt, args.model_type, args.prompt_type)

    test_dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=DataCollatorForInferenceDataset(tokenizer),
    )
    
    config = PeftConfig.from_pretrained(args.model_ckpt_path)
    quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                llm_int8_threshold=6.0,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type='nf4'
                )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map={'': args.device},  # Use explicit device map
        quantization_config=quantization_config,
    )
    model.resize_token_embeddings(len(tokenizer))
    model = PeftModel.from_pretrained(model, args.model_ckpt_path)
    model.eval()
    model.to(args.device)
    torch.set_grad_enabled(False)

    with open("resource/data/일상대화요약_test.json", "r") as f:
        result = json.load(f)

    batch_start_idx = 0
    for batch in tqdm(test_dataloader, desc="Test"):
        inp = batch["input_ids"].to(args.device)
        attention_mask = batch["attention_mask"].to(args.device)
        speakermap = batch["speaker_maps"]
        
        outputs = model.generate(
            inp,
            attention_mask=attention_mask,
            max_new_tokens=2048,
            eos_token_id=terminators,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=False,
            # top_k = 200,
            # top_p = 0.90,
            num_beams=2,
            early_stopping=True,
            # num_return_sequences = 3
            #repetition_penalty=1.2,  # 반복 방지
            #no_repeat_ngram_size=3   # 반복 방지
        )

        generated_texts = []
        for output in outputs:
            text = tokenizer.decode(output[inp.shape[-1]:], skip_special_tokens=False)
            generated_texts.append(text)

        # Replace special tokens with speaker IDs
        for i, text in enumerate(generated_texts):
            speaker_map = speakermap[i]
            for token, speaker in speaker_map.items():
                text = text.replace(speaker, token)
            text = text.replace("<|end_of_text|>", "")
            text = text.replace("<|begin_of_text|>", "")
            text = text.replace("</s>", "")
            text = text.replace("[|endofturn|]", "")
            result[batch_start_idx + i]["output"] = text
            print(result[batch_start_idx + i]["output"])

        batch_start_idx += len(generated_texts)
    
    with open(f"inference/{args.output}", "w", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False, indent=4))


if __name__ == "__main__":
    exit(main(parser.parse_args()))