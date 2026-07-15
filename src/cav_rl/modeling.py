from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class CAVCriticOutput:
    v_high: torch.Tensor
    v_low: torch.Tensor


def load_tokenizer(model_name_or_path: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer


def load_actor(model_name_or_path: str, torch_dtype=None):
    return AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch_dtype, trust_remote_code=True)


class CAVCritic(nn.Module):
    """Shared Qwen-style critic with high-level and low-level value heads.

    V_high(H_k) estimates the value before choosing a computation budget.
    V_low(H_k, b_k) estimates the value after a budget is committed and before the
    budget-conditioned reason/answer payload is generated.
    """

    def __init__(self, model_name_or_path: str, torch_dtype=None, pooling: str = "last_token"):
        super().__init__()
        self.backbone = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
        )
        hidden_size = self.backbone.config.hidden_size
        self.pooling = pooling
        self.v_high_head = nn.Linear(hidden_size, 1)
        self.v_low_head = nn.Linear(hidden_size, 1)

    def _pool(self, hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if self.pooling != "last_token":
            masked = hidden * attention_mask.unsqueeze(-1)
            denom = attention_mask.sum(dim=1, keepdim=True).clamp_min(1)
            return masked.sum(dim=1) / denom
        lengths = attention_mask.shape[1] - 1 - attention_mask.flip(dims=[1]).argmax(dim=1)
        batch_idx = torch.arange(hidden.shape[0], device=hidden.device)
        return hidden[batch_idx, lengths]

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> CAVCriticOutput:
        output = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )
        pooled = self._pool(output.hidden_states[-1], attention_mask)
        return CAVCriticOutput(
            v_high=self.v_high_head(pooled).squeeze(-1),
            v_low=self.v_low_head(pooled).squeeze(-1),
        )


def maybe_apply_lora(model, lora_config_data):
    if not lora_config_data.enabled:
        return model
    from peft import LoraConfig, get_peft_model

    config = LoraConfig(
        r=lora_config_data.r,
        lora_alpha=lora_config_data.alpha,
        lora_dropout=lora_config_data.dropout,
        target_modules=lora_config_data.target_modules,
        task_type="CAUSAL_LM",
    )
    return get_peft_model(model, config)

