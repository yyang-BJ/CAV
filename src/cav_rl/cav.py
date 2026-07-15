from __future__ import annotations

import torch

from .rollout import CAVRolloutSample


def _device(model) -> torch.device:
    return next(model.parameters()).device


def _critic_batch_values(critic, tokenizer, texts: list[str]):
    if not texts:
        device = _device(critic)
        empty = torch.empty(0, device=device)
        return empty, empty
    device = _device(critic)
    encoded = tokenizer(texts, padding=True, return_tensors="pt", add_special_tokens=False).to(device)
    with torch.no_grad():
        values = critic(encoded["input_ids"], encoded["attention_mask"])
    return values.v_high.detach(), values.v_low.detach()


def attach_cav_targets_and_advantages(
    samples: list[CAVRolloutSample],
    critic,
    tokenizer,
    gamma: float,
    gae_lambda: float,
) -> None:
    """Compute macro-step CAV/GAE advantages and critic targets.

    Budget fields receive the high-level allocation advantage:
        delta_k = r_k + gamma * V_high(H_{k+1}) - V_high(H_k)

    Reason/answer payload fields receive the low-level executor advantage:
        delta_low_k = r_k + gamma * V_high(H_{k+1}) - V_low(H_k, b_k)

    This follows the proposal optimization notes: the budget action is credited
    by the value of the next reasoning state minus the current state value and
    computation price; the price is already included in r_k.
    """
    all_high = [macro.high_prefix for sample in samples for macro in sample.macros]
    all_low = [macro.low_prefix for sample in samples for macro in sample.macros]
    high_values, _ = _critic_batch_values(critic, tokenizer, all_high)
    _, low_values = _critic_batch_values(critic, tokenizer, all_low)

    cursor = 0
    for sample in samples:
        n = len(sample.macros)
        if n == 0:
            continue
        v_high = high_values[cursor : cursor + n]
        v_low = low_values[cursor : cursor + n]
        rewards = torch.tensor([macro.reward for macro in sample.macros], device=v_high.device, dtype=v_high.dtype)
        next_high = torch.cat([v_high[1:], torch.zeros(1, device=v_high.device, dtype=v_high.dtype)])

        high_deltas = rewards + gamma * next_high - v_high
        low_deltas = rewards + gamma * next_high - v_low

        high_adv = torch.zeros_like(high_deltas)
        running = torch.zeros((), device=v_high.device, dtype=v_high.dtype)
        for t in reversed(range(n)):
            running = high_deltas[t] + gamma * gae_lambda * running
            high_adv[t] = running

        low_adv = low_deltas
        high_targets = high_adv + v_high
        low_targets = low_adv + v_low

        for i, macro in enumerate(sample.macros):
            macro.high_advantage = float(high_adv[i].detach().cpu())
            macro.low_advantage = float(low_adv[i].detach().cpu())
            macro.high_target = float(high_targets[i].detach().cpu())
            macro.low_target = float(low_targets[i].detach().cpu())

        macro_by_index = {macro.macro_index: macro for macro in sample.macros}
        for field in sample.fields:
            if field.macro_index is None or field.macro_index not in macro_by_index:
                continue
            macro = macro_by_index[field.macro_index]
            if field.name == "budget":
                field.advantage = macro.high_advantage
                field.target = macro.high_target
            else:
                field.advantage = macro.low_advantage
                field.target = macro.low_target
        cursor += n

