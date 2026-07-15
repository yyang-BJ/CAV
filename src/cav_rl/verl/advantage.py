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

    This estimator is intentionally token-shaped so it can flow through the
    standard veRL actor loss. Budget tokens get allocation-level CAV/GAE credit,
    while reason/answer tokens get executor-level return credit. If a standard
    critic value tensor is present in kwargs/config plumbing, it is used as the
    baseline; otherwise it falls back to reward-to-go, which is useful for early
    debugging or critic-free ablations.
    """
    gamma = float(getattr(config, "gamma", 1.0)) if config is not None else 1.0
    lam = float(getattr(config, "lam", 0.95)) if config is not None else 0.95
    values = kwargs.get("values")
    budget_mask = kwargs.get("cav_budget_mask")
    executor_mask = kwargs.get("cav_executor_mask")

    if budget_mask is None:
        budget_mask = response_mask
    if executor_mask is None:
        executor_mask = response_mask - budget_mask
    if values is None:
        values = torch.zeros_like(token_level_rewards)

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
            config=config,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
        return data

    ray_trainer.compute_advantage = compute_advantage_with_cav
