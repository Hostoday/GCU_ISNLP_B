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

def loss_fn(target, outputs, tokenizer):
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    logits = outputs.logits.view(-1, outputs.logits.size(-1))
    target = target.view(-1)
    loss = criterion(logits, target)
    return loss

def calc_BLEU(true, pred, apply_avg=True, apply_best=False, use_mecab=True):
    stacked_bleu = []

    if type(true[0]) is str:
        true = list(map(lambda x: [x], true))

    mecab_tokenizer = Mecab()

    for i in range(len(true)):
        best_bleu = 0
        sum_bleu = 0
        for j in range(len(true[i])):
            if use_mecab:
                ref = mecab_tokenizer.morphs(true[i][j])
                candi = mecab_tokenizer.morphs(pred[i])
            else:
                ref = true[i][j].split()
                candi = pred[i].split()

            score = sentence_bleu([ref], candi, weights=(1, 0, 0, 0))

            sum_bleu += score
            if score > best_bleu:
                best_bleu = score

        avg_bleu = sum_bleu / len(true[i])
        if apply_best:
            stacked_bleu.append(best_bleu)
        if apply_avg:
            stacked_bleu.append(avg_bleu)

    return sum(stacked_bleu) / len(stacked_bleu)

def calc_ROUGE_1(true, pred):
    rouge_evaluator = Rouge()
    scores = rouge_evaluator.get_scores(pred, true, avg=True)
    return scores['rouge-1']['f']

def calc_bertscore(true, pred):
    P, R, F1 = bert_score_func(cands=pred, refs=true, lang="ko", model_type='bert-base-multilingual-cased', rescale_with_baseline=True)
    return F1.mean().item()

def compute_metrics(eval_pred, tokenizer):
    preds, labels = eval_pred.predictions, eval_pred.label_ids
    
    if isinstance(preds, tuple):
        preds = preds[0]
    if isinstance(labels, tuple):
        labels = labels[0]

    labels = labels[:, 1:]
    preds = preds[:, :-1]
    
    mask = labels == -100
    labels[mask] = tokenizer.pad_token_id
    preds[mask] = tokenizer.pad_token_id
    
    decoded_labels = [tokenizer.decode(label, skip_special_tokens=True) for label in labels]
    decoded_preds = [tokenizer.decode(pred, skip_special_tokens=True) for pred in preds]
    
    avg_rouge1 = calc_ROUGE_1(decoded_labels, decoded_preds)
    avg_bertscore = calc_bertscore(decoded_labels, decoded_preds)
    avg_bleu = calc_BLEU(decoded_labels, decoded_preds)

    return {
        'rouge1': avg_rouge1,
        'bertscore': avg_bertscore,
        'bleu': avg_bleu,
        'average_score': (avg_rouge1 + avg_bertscore + avg_bleu) / 3
    }

def preprocess_logits_for_metrics(logits, labels):
    """
    Original Trainer may have a memory leak. 
    This is a workaround to avoid storing too many tensors that are not needed.
    """
    pred_ids = torch.argmax(logits, dim=-1)
    return pred_ids, labels

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
    # set_random_seed(3209512016602887751 % (2**32))
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
        # evaluation_strategy="steps",
        # eval_steps=10,
        do_train=True,
        do_eval=False,
        auto_find_batch_size=True,
        optim="adamw_bnb_8bit",
        lr_scheduler_type="cosine",
        neftune_noise_alpha=5,
        per_device_train_batch_size=args.batch_size,
        # per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epoch,
        learning_rate=args.lr,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        output_dir=f'{args.save_dir}/{args.model_id}/{datetime.now(timezone(timedelta(hours=9))).strftime("%Y-%m-%d-%H-%M")}',
        save_steps=10,
        save_total_limit=10,
    )

    # Define a wrapper function to pass the tokenizer to compute_metrics
    def compute_metrics_wrapper(pred):
        return compute_metrics(pred, tokenizer)
    
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=dataset,
        # eval_dataset=valid_dataset,
        data_collator=data_collator,
        # preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        # compute_metrics=compute_metrics_wrapper,
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