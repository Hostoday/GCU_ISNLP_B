import os

from unsloth import FastLanguageModel 
from unsloth import is_bfloat16_supported
import argparse
import json
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import (AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, 
                          Trainer, TrainingArguments, get_cosine_schedule_with_warmup, 
                          get_linear_schedule_with_warmup, EarlyStoppingCallback, EvalPrediction)
from trl import SFTTrainer
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model, TaskType
from src.data import CustomDataset, DataCollatorForSupervisedDataset
from src.utils import set_random_seed
from src.arg_parser import get_args
from accelerate import Accelerator
import wandb
from transformers import TrainerCallback, TrainerState, TrainerControl
import logging
from datetime import datetime, timezone, timedelta
from datasets import Dataset, load_metric

from nltk.translate.bleu_score import corpus_bleu, sentence_bleu
from konlpy.tag import Mecab
from rouge import Rouge
from bert_score import score as bert_score_func

import numpy as np

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class CustomCallback(TrainerCallback):
    def on_epoch_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        logging.info(f"Starting epoch {state.epoch}...")
        print(f"Starting epoch {state.epoch}...")

    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        logging.info(f"Log: {state.log_history[-1]}")
        print(f"Log: {state.log_history[-1]}")

def init_model(args):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_id,
        max_seq_length=2048,
        dtype=torch.bfloat16,
        load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        use_gradient_checkpointing=True,
        random_state=3407,
    )
    
    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    return model, tokenizer

def init_tokenizer(args):
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer if args.tokenizer else args.model_id)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

def train_model(args):
    set_random_seed(args.seed)
    model, tokenizer = init_model(args)

    dataset = CustomDataset("resource/data/일상대화요약_train.json", tokenizer, args.prompt, args.model_type, args.prompt_type)
    dataset = Dataset.from_dict({'input_ids': dataset.inp, 'labels': dataset.label})

    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    model.resize_token_embeddings(len(tokenizer))    

    training_args = TrainingArguments(
        save_strategy="steps",
        warmup_steps=10,
        weight_decay=args.weight_decay,
        logging_steps=10,
        do_train=True,
        do_eval=False,
        auto_find_batch_size=True,
        optim="adamw_bnb_8bit",
        lr_scheduler_type="cosine",
        neftune_noise_alpha=5,
        per_device_train_batch_size=args.batch_size,
        num_train_epochs=args.epoch,
        learning_rate=args.lr,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        output_dir=f'{args.save_dir}/{args.model_id}/GCU_ISNLP_B',
        save_steps=10,
        save_total_limit=10,
    )

    
    trainer = SFTTrainer(
        model=model,
        dataset_text_field='text',
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
        callbacks=[CustomCallback],
    )
    
    trainer.train()

if __name__ == "__main__":
    args = get_args()

    if args.wandb=='True':
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            name=args.wandb_run_name,
            config=args,
        )
        wandb.config.update(args)
    
    if args.trainer == "True":
        exit(train_model(args)) 