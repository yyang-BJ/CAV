from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import Trainer, TrainingArguments, set_seed

from cav_rl.config import load_sft_config
from cav_rl.data import CAVSFTDataset, load_gsm8k_examples, make_sft_collator
from cav_rl.modeling import load_actor, load_tokenizer, maybe_apply_lora


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/sft_gsm8k.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_sft_config(args.config)
    set_seed(config.seed)
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)

    dtype = torch.bfloat16 if config.bf16 else None
    tokenizer = load_tokenizer(config.model_name_or_path)
    model = load_actor(config.model_name_or_path, torch_dtype=dtype)
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    model = maybe_apply_lora(model, config.lora)

    train_examples = load_gsm8k_examples(
        config.dataset_name,
        config.dataset_config,
        split="train",
        max_samples=config.max_train_samples,
    )
    eval_examples = load_gsm8k_examples(
        config.dataset_name,
        config.dataset_config,
        split="test",
        max_samples=config.max_eval_samples,
    )
    max_length = config.max_prompt_length + config.max_completion_length
    train_dataset = CAVSFTDataset(
        train_examples,
        tokenizer,
        max_length=max_length,
        budget_actions=config.budget_actions,
        reason_budget=config.sft_reason_budget,
    )
    eval_dataset = CAVSFTDataset(
        eval_examples,
        tokenizer,
        max_length=max_length,
        budget_actions=config.budget_actions,
        reason_budget=config.sft_reason_budget,
    )

    training_args = TrainingArguments(
        output_dir=config.output_dir,
        per_device_train_batch_size=config.train_batch_size,
        per_device_eval_batch_size=config.train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        num_train_epochs=config.num_train_epochs,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        bf16=config.bf16,
        eval_strategy="steps",
        eval_steps=config.save_steps,
        save_total_limit=2,
        report_to="none",
        remove_unused_columns=False,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=make_sft_collator(tokenizer),
    )
    trainer.train()
    trainer.save_model(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)


if __name__ == "__main__":
    main()

