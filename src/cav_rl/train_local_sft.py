from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch
from transformers import Trainer, TrainingArguments, set_seed

from cav_rl.config import load_sft_config
from cav_rl.data import (
    BaselineSFTDataset,
    CAVSFTDataset,
    build_validated_sft_completion,
    choose_sft_mode,
    load_gsm8k_examples,
    make_sft_collator,
)
from cav_rl.modeling import load_actor, load_tokenizer, maybe_apply_lora
from cav_rl.parsing import parse_completion
from cav_rl.prompts import build_sft_completion
from cav_rl.sft_trainer import FormatWeightedSFTTrainer


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="CAV format-alignment SFT")
    parser.add_argument("--config", default="configs/sft_gsm8k.yaml")
    return parser.parse_args(argv)


def _dataset_kwargs(config) -> dict:
    return dict(
        budget_actions=config.budget_actions,
        reason_budget=config.sft_reason_budget,
        seed=config.seed,
        fit_budget=bool(getattr(config, "sft_fit_budget", True)),
        direct_answer_prob=float(getattr(config, "sft_direct_answer_prob", 0.0)),
        multi_macro_prob=float(getattr(config, "sft_multi_macro_prob", 0.0)),
        format_token_weight=float(getattr(config, "sft_format_token_weight", 1.0)),
    )


def _spot_check_format(config, tokenizer, train_examples, ds_kwargs: dict) -> None:
    allowed = set(config.budget_actions)
    n_check = min(64, len(train_examples))
    invalid = 0
    for i in range(n_check):
        ex = train_examples[i]
        mode = choose_sft_mode(
            random.Random(config.seed + i),
            ex.rationale,
            direct_prob=ds_kwargs["direct_answer_prob"],
            multi_prob=ds_kwargs["multi_macro_prob"],
        )
        if ds_kwargs["fit_budget"]:
            completion = build_validated_sft_completion(
                ex,
                tokenizer,
                config.budget_actions,
                reason_budget=config.sft_reason_budget,
                mode=mode,
            )
        else:
            completion = build_sft_completion(ex.rationale, ex.answer, config.sft_reason_budget or 64)
        if not parse_completion(completion, allowed, tokenizer=tokenizer).valid_format:
            invalid += 1
    print(
        f"[CAV-SFT] format spot-check: {n_check - invalid}/{n_check} valid "
        f"(fit_budget={ds_kwargs['fit_budget']}, direct={ds_kwargs['direct_answer_prob']}, "
        f"multi={ds_kwargs['multi_macro_prob']}, tag_weight={ds_kwargs['format_token_weight']})",
        flush=True,
    )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = load_sft_config(args.config)
    set_seed(config.seed)
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)

    dtype = torch.bfloat16 if config.bf16 else None
    tokenizer = load_tokenizer(config.model_name_or_path)
    # Causal LM SFT requires right padding; left padding is only for generation.
    tokenizer.padding_side = "right"
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
    sft_style = str(getattr(config, "sft_style", "cav")).lower()
    if sft_style in {"baseline", "cot", "gsm8k"}:
        train_dataset = BaselineSFTDataset(train_examples, tokenizer, max_length=max_length)
        eval_dataset = BaselineSFTDataset(eval_examples, tokenizer, max_length=max_length)
        print(f"[CAV-SFT] style=baseline CoT | train={len(train_dataset)} eval={len(eval_dataset)}", flush=True)
        trainer_cls = Trainer
        format_token_weight = 1.0
    else:
        ds_kwargs = _dataset_kwargs(config)
        train_dataset = CAVSFTDataset(train_examples, tokenizer, max_length=max_length, **ds_kwargs)
        eval_dataset = CAVSFTDataset(eval_examples, tokenizer, max_length=max_length, **ds_kwargs)
        _spot_check_format(config, tokenizer, train_examples, ds_kwargs)
        format_token_weight = ds_kwargs["format_token_weight"]
        trainer_cls = FormatWeightedSFTTrainer if format_token_weight > 1.0 else Trainer

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

    trainer = trainer_cls(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=make_sft_collator(tokenizer),
    )
    trainer.train()
    trainer.save_model(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)
    print(f"[CAV-SFT] saved adapter/model to {config.output_dir}", flush=True)


if __name__ == "__main__":
    main()
