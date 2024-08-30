import argparse
import json
from tqdm import tqdm

import torch
import numpy
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import random
from src.concat_data import CustomDataset, DataCollatorForInferenceDataset
from peft import PeftModel, PeftConfig
from src.utils import set_random_seed

import os
import numpy as np
from unsloth import FastLanguageModel 

os.environ["TOKENIZERS_PARALLELISM"] = "false"


# fmt: off
parser = argparse.ArgumentParser(prog="test", description="Testing about Conversational Context Inference.")

g = parser.add_argument_group("Common Parameter")
g.add_argument("--output", type=str, required=True, help="output filename")
g.add_argument("--model_id", type=str, required=True, help="huggingface model id")
g.add_argument("--tokenizer", type=str, help="huggingface tokenizer")
g.add_argument("--device", type=str, required=True, help="device to load the model")
g.add_argument("--model_ckpt_path", type=str, required=True, help="model checkpoint path")
g.add_argument("--batch_size", type=int, default=2, help="batch size")
g.add_argument("--model_type", type=str, default="default", help="model is gemma?")
g.add_argument("--concat_file_list", type = str, nargs='+', required=True , help = "concat file others")
g.add_argument("--prompt_type", type=str, default="summary_concat",help="prompt type")
g.add_argument("--prompt",type=str,default = "You are a helpful AI assistant. Please summarize the main topics of the user's conversation. 당신은 유능한 AI 어시스턴트입니다. 사용자의 대화를 통해 주요 주제에 대해 요약해주세요.")
g.add_argument("--seed", type = int, default = 42, help = "top-k inference")
# fmt: on



def main(args):
    set_random_seed(args.seed)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_id,
        max_seq_length=1024,
        dtype=torch.bfloat16,
        load_in_4bit=True,
    )
     
    if args.tokenizer == None:
        args.tokenizer = args.model_id
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    # Set padding_side to left
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]
    if args.model_id == "rtzr/ko-gemma-2-9b-it":
        terminators.append(tokenizer.convert_tokens_to_ids("<eos>"))
        
    dataset = CustomDataset("resource/data/일상대화요약_test.json",args.concat_file_list, tokenizer,  args.prompt, args.model_type)
    

    test_dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=DataCollatorForInferenceDataset(tokenizer),
    )
    
    # config = PeftConfig.from_pretrained(args.model_ckpt_path)
    # quantization_config = BitsAndBytesConfig(
    #             load_in_4bit=True,
    #             llm_int8_threshold=6.0,
    #             bnb_4bit_compute_dtype=torch.float16,
    #             bnb_4bit_use_double_quant=True,
    #             bnb_4bit_quant_type='nf4'
    #             )

    # model = AutoModelForCausalLM.from_pretrained(
    #     args.model_id,
    #     torch_dtype=torch.bfloat16,
    #     device_map=args.device,
    #     quantization_config=quantization_config,
    # )

    model.resize_token_embeddings(len(tokenizer))
    model = PeftModel.from_pretrained(model, args.model_ckpt_path)
    FastLanguageModel.for_inference(model)
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
        # eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
        # pad_token_id=tokenizer.pad_token_id,
        do_sample=False,
        early_stopping=True
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
            text = text.replace("<|eot_id|>", "")
            text = text.replace("<|start_header_id|>", "")
            text = text.replace("assistant", "")            
            text = text.replace("[|endofturn|]","")
            text = text.replace("<eos>","")
            text = text.replace("<bos>","")
            text = text.replace("</h1>","")
            text = text.replace("<s>","")
            text = text.replace("</s>","")
            text = text.replace("###","")
            text = text.replace("Instruction","")
            
            result[batch_start_idx + i]["output"] = text
            print(result[batch_start_idx + i]["output"])

        batch_start_idx += len(generated_texts)

    with open(f"inference/{args.output}", "w", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False, indent=4))


if __name__ == "__main__":
    exit(main(parser.parse_args()))