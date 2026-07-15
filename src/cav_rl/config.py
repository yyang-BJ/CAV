from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class LoraConfigData:
    enabled: bool = True
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: list[str] = field(default_factory=list)


@dataclass
class GenerationConfigData:
    temperature: float = 1.0
    top_p: float = 0.95
    do_sample: bool = True


@dataclass
class RewardConfig:
    correct_reward: float = 1.0
    wrong_reward: float = 0.0
    computation_price: float = 0.0005
    actual_token_price: float = 0.0001
    format_penalty: float = 0.1
    missing_stop_penalty: float = 0.2
    invalid_budget_penalty: float = 0.1
    overflow_budget_penalty: float = 0.05
    target_expected_budget: float = 128.0
    dual_lr: float = 0.01
    initial_lambda_c: float = 0.0005
    min_lambda_c: float = 0.0
    max_lambda_c: float = 0.02


@dataclass
class CriticConfig:
    pooling: str = "last_token"
    use_bootstrap_targets: bool = True


@dataclass
class SFTConfig:
    model_name_or_path: str
    dataset_name: str
    dataset_config: str
    output_dir: str
    seed: int = 42
    max_prompt_length: int = 1024
    max_completion_length: int = 768
    max_train_samples: int | None = None
    max_eval_samples: int | None = 256
    train_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    num_train_epochs: int = 1
    learning_rate: float = 2e-5
    weight_decay: float = 0.0
    warmup_ratio: float = 0.03
    logging_steps: int = 10
    save_steps: int = 500
    bf16: bool = True
    gradient_checkpointing: bool = True
    budget_actions: list[int] = field(default_factory=lambda: [0, 16, 32, 64, 128])
    sft_reason_budget: int = 64
    lora: LoraConfigData = field(default_factory=LoraConfigData)


@dataclass
class CAVPPOConfig:
    model_name_or_path: str
    dataset_name: str
    dataset_config: str
    output_dir: str
    sft_adapter_or_model: str | None = None
    critic_model_name_or_path: str | None = None
    seed: int = 42
    max_prompt_length: int = 1024
    max_completion_length: int = 768
    max_train_samples: int | None = None
    max_eval_samples: int | None = 256
    budget_actions: list[int] = field(default_factory=lambda: [0, 16, 32, 64, 128])
    max_macro_steps: int = 6
    rollout_batch_size: int = 4
    ppo_epochs: int = 2
    mini_batch_size: int = 1
    num_iterations: int = 1000
    actor_learning_rate: float = 1e-6
    critic_learning_rate: float = 1e-5
    weight_decay: float = 0.0
    clip_range: float = 0.2
    value_coef: float = 1.0
    kl_coef: float = 0.01
    entropy_coef: float = 0.0
    max_grad_norm: float = 1.0
    gamma: float = 1.0
    gae_lambda: float = 0.95
    generation: GenerationConfigData = field(default_factory=GenerationConfigData)
    reward: RewardConfig = field(default_factory=RewardConfig)
    critic: CriticConfig = field(default_factory=CriticConfig)
    lora: LoraConfigData = field(default_factory=LoraConfigData)


def _strip_none(data: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in data.items() if v is not None}


def _nested(cls: type, data: dict[str, Any] | None):
    return cls(**_strip_none(data or {}))


def _load_yaml(path: str | Path) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def load_sft_config(path: str | Path) -> SFTConfig:
    data = _load_yaml(path)
    data["lora"] = _nested(LoraConfigData, data.get("lora"))
    return SFTConfig(**_strip_none(data))


def load_cav_ppo_config(path: str | Path) -> CAVPPOConfig:
    data = _load_yaml(path)
    data["generation"] = _nested(GenerationConfigData, data.get("generation"))
    data["reward"] = _nested(RewardConfig, data.get("reward"))
    data["critic"] = _nested(CriticConfig, data.get("critic"))
    data["lora"] = _nested(LoraConfigData, data.get("lora"))
    return CAVPPOConfig(**_strip_none(data))

