from __future__ import annotations

import torch


def _masked_whiten(values: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    valid = mask > 0
    if valid.sum() <= 1:
        return values * mask
    mean = values[valid].mean()
    std = values[valid].std().clamp_min(eps)
    return (values - mean) / std * mask


def compute_cav_gae_advantage_return(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    config=None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """veRL custom estimator for CAV.

    CAV decisions are macro-step actions, so rewards and advantages are first
    aggregated by macro_id. The resulting A_k^GAE is then broadcast back to all
    budget/reason/answer tokens that belong to that macro step, preserving the
    tensor shape expected by veRL's standard actor and critic losses.
    """
    gamma = float(getattr(config, "gamma", 1.0)) if config is not None else 1.0
    lam = float(getattr(config, "lam", 0.95)) if config is not None else 0.95
    values = kwargs.get("values")
    budget_mask = kwargs.get("cav_budget_mask")
    executor_mask = kwargs.get("cav_executor_mask")
    reason_mask = kwargs.get("cav_reason_mask")
    macro_ids = kwargs.get("cav_macro_ids")

    if budget_mask is None:
        budget_mask = response_mask
    if executor_mask is None:
        executor_mask = response_mask - budget_mask
    if reason_mask is None:
        reason_mask = executor_mask
    if values is None:
        values = torch.zeros_like(token_level_rewards)

    if macro_ids is None:
        return _token_level_fallback(token_level_rewards, response_mask, values, budget_mask, executor_mask, gamma, lam)

    with torch.no_grad():
        advantages = torch.zeros_like(token_level_rewards)
        returns = torch.zeros_like(token_level_rewards)

        for batch_idx in range(token_level_rewards.shape[0]):
            valid_ids = macro_ids[batch_idx][(response_mask[batch_idx] > 0) & (macro_ids[batch_idx] >= 0)]
            if valid_ids.numel() == 0:
                continue
            macro_count = int(valid_ids.max().item()) + 1
            macro_rewards = []
            macro_values = []
            macro_discounts = []
            macro_token_masks = []

            for macro_idx in range(macro_count):
                macro_mask = (macro_ids[batch_idx] == macro_idx) & (response_mask[batch_idx] > 0)
                if not macro_mask.any():
                    macro_rewards.append(token_level_rewards.new_tensor(0.0))
                    macro_values.append(token_level_rewards.new_tensor(0.0))
                    macro_discounts.append(token_level_rewards.new_tensor(1.0))
                    macro_token_masks.append(macro_mask)
                    continue

                reward = token_level_rewards[batch_idx][macro_mask].sum()
                value_positions = torch.nonzero(
                    macro_mask & (budget_mask[batch_idx] > 0),
                    as_tuple=False,
                ).flatten()
                if value_positions.numel() == 0:
                    value_positions = torch.nonzero(macro_mask, as_tuple=False).flatten()
                value = values[batch_idx, value_positions[0]]
                duration = int(reason_mask[batch_idx][macro_mask].sum().item())
                macro_rewards.append(reward)
                macro_values.append(value)
                macro_discounts.append(token_level_rewards.new_tensor(gamma ** max(duration, 0)))
                macro_token_masks.append(macro_mask)

            rewards = torch.stack(macro_rewards)
            macro_value_tensor = torch.stack(macro_values)
            discounts = torch.stack(macro_discounts)
            next_values = torch.cat(
                [
                    macro_value_tensor[1:],
                    torch.zeros(1, device=macro_value_tensor.device, dtype=macro_value_tensor.dtype),
                ]
            )
            deltas = rewards + discounts * next_values - macro_value_tensor
            macro_adv = torch.zeros_like(deltas)
            running = torch.zeros((), device=deltas.device, dtype=deltas.dtype)
            for macro_idx in reversed(range(macro_count)):
                running = deltas[macro_idx] + discounts[macro_idx] * lam * running
                macro_adv[macro_idx] = running

            for macro_idx, macro_mask in enumerate(macro_token_masks):
                if macro_mask.any():
                    advantages[batch_idx, macro_mask] = macro_adv[macro_idx]

        returns = (advantages + values) * response_mask
        advantages = _masked_whiten(advantages, response_mask)
    return advantages, returns


def _token_level_fallback(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    values: torch.Tensor,
    budget_mask: torch.Tensor,
    executor_mask: torch.Tensor,
    gamma: float,
    lam: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        next_value = torch.zeros(token_level_rewards.shape[0], device=token_level_rewards.device)
        running = torch.zeros_like(next_value)
        advantages_rev = []
        for t in reversed(range(token_level_rewards.shape[1])):
            delta = token_level_rewards[:, t] + gamma * next_value - values[:, t]
            running = delta + gamma * lam * running
            active = response_mask[:, t]
            running = running * active
            next_value = values[:, t] * active + next_value * (1 - active)
            advantages_rev.append(running)
        macro_advantages = torch.stack(advantages_rev[::-1], dim=1) * response_mask

        executor_returns = torch.zeros_like(token_level_rewards)
        running_return = torch.zeros(token_level_rewards.shape[0], device=token_level_rewards.device)
        for t in reversed(range(token_level_rewards.shape[1])):
            running_return = token_level_rewards[:, t] + gamma * running_return
            running_return = running_return * response_mask[:, t]
            executor_returns[:, t] = running_return

        advantages = macro_advantages * budget_mask + (executor_returns - values) * executor_mask
        returns = advantages + values
        advantages = _masked_whiten(advantages, response_mask)
    return advantages, returns


def register_cav_advantage() -> None:
    from verl.trainer.ppo.core_algos import register_adv_est

    register_adv_est("cav_gae")(compute_cav_gae_advantage_return)


def patch_ray_trainer_compute_advantage() -> None:
    """Patch veRL's driver-side advantage dispatcher for CAV masks.

    veRL's extension registry passes standard tensors to custom estimators, but
    CAV needs structural masks that are created by `CAVRewardManager`. This patch
    keeps all existing estimators untouched and only intercepts
    `algorithm.adv_estimator=cav_gae`.
    """
    from verl.trainer.ppo import ray_trainer

    original = ray_trainer.compute_advantage

    def compute_advantage_with_cav(
        data,
        adv_estimator,
        gamma: float = 1.0,
        lam: float = 1.0,
        num_repeat: int = 1,
        norm_adv_by_std_in_grpo: bool = True,
        config=None,
        **kwargs,
    ):
        if str(adv_estimator) != "cav_gae":
            return original(
                data=data,
                adv_estimator=adv_estimator,
                gamma=gamma,
                lam=lam,
                num_repeat=num_repeat,
                norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                config=config,
                **kwargs,
            )

        if "response_mask" not in data.batch:
            from verl.trainer.ppo.ray_trainer import compute_response_mask

            data.batch["response_mask"] = compute_response_mask(data)

        advantages, returns = compute_cav_gae_advantage_return(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=data.batch["response_mask"],
            values=data.batch.get("values"),
            cav_budget_mask=data.batch.get("cav_budget_mask"),
            cav_executor_mask=data.batch.get("cav_executor_mask"),
            cav_reason_mask=data.batch.get("cav_reason_mask"),
            cav_macro_ids=data.batch.get("cav_macro_ids"),
            config=config,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
        return data

    ray_trainer.compute_advantage = compute_advantage_with_cav
