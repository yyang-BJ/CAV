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

    The rollout unit is a macro step. For a positive budget action, the reward
    is the computation price charged on actual consumed tokens l_k, and the
    bootstrap discount is gamma ** l_k:

        delta_k = r_k + gamma^{l_k} V_high(H_{k+1}) - V_high(H_k)

    The budget field and the generated payload field share the same macro GAE
    advantage A_k. V_low(H_k, b_k) is still trained as an executor-state value
    diagnostic, but it does not create a separate reasoning advantage.
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
        discounts = torch.tensor(
            [gamma ** max(int(macro.duration), 0) for macro in sample.macros],
            device=v_high.device,
            dtype=v_high.dtype,
        )
        next_high = torch.cat([v_high[1:], torch.zeros(1, device=v_high.device, dtype=v_high.dtype)])

        high_deltas = rewards + discounts * next_high - v_high

        high_adv = torch.zeros_like(high_deltas)
        running = torch.zeros((), device=v_high.device, dtype=v_high.dtype)
        for t in reversed(range(n)):
            running = high_deltas[t] + discounts[t] * gae_lambda * running
            high_adv[t] = running

        low_adv = high_adv
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
            field.advantage = macro.high_advantage
            field.target = macro.high_target
        cursor += n
