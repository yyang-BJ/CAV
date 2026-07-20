from __future__ import annotations

import inspect
import warnings
from functools import wraps
from typing import Any

import torch


def _segment_ppo_loss(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    cav_macro_ids: torch.Tensor,
    cav_budget_mask: torch.Tensor,
    cav_executor_mask: torch.Tensor,
    cliprange: float,
    cliprange_low: float | None = None,
    cliprange_high: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """CAV segment-level PPO objective.

    veRL stores token log probabilities, but CAV's policy factorization is
    macro/field based:

        pi(b_k, z_k | H_k) = pi(b_k | H_k) pi(z_k | H_k, b_k)

    Each budget block and each executor block gets one PPO ratio based on the
    sum of token log probabilities in that block. All segments in the same macro
    step use the same A_k that the CAV advantage estimator broadcasts.
    """
    if cliprange_low is None:
        cliprange_low = cliprange
    if cliprange_high is None:
        cliprange_high = cliprange

    losses = []
    clip_hits = []
    lower_clip_hits = []
    kls = []

    for batch_idx in range(log_prob.shape[0]):
        valid_ids = cav_macro_ids[batch_idx][
            (response_mask[batch_idx] > 0) & (cav_macro_ids[batch_idx] >= 0)
        ]
        if valid_ids.numel() == 0:
            continue
        macro_count = int(valid_ids.max().item()) + 1
        for macro_idx in range(macro_count):
            macro_mask = (cav_macro_ids[batch_idx] == macro_idx) & (response_mask[batch_idx] > 0)
            for field_mask in (
                macro_mask & (cav_budget_mask[batch_idx] > 0),
                macro_mask & (cav_executor_mask[batch_idx] > 0),
            ):
                if not field_mask.any():
                    continue
                current = log_prob[batch_idx][field_mask].sum()
                old = old_log_prob[batch_idx][field_mask].sum()
                ratio = torch.exp(current - old)
                clipped_ratio = torch.clamp(ratio, 1.0 - cliprange_low, 1.0 + cliprange_high)
                adv = advantages[batch_idx][field_mask][0]
                losses.append(-torch.min(ratio * adv, clipped_ratio * adv))
                clip_hits.append((torch.abs(ratio - 1.0) > cliprange).to(log_prob.dtype))
                lower_clip_hits.append((ratio < 1.0 - cliprange_low).to(log_prob.dtype))
                kls.append(current - old)

    if not losses:
        zero = log_prob.new_tensor(0.0)
        return zero, zero, zero, zero

    pg_loss = torch.stack(losses).mean()
    pg_clipfrac = torch.stack(clip_hits).mean()
    ppo_kl = torch.stack(kls).mean()
    pg_clipfrac_lower = torch.stack(lower_clip_hits).mean()
    return pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower


def compute_cav_policy_loss(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    *,
    cav_macro_ids: torch.Tensor | None = None,
    cav_budget_mask: torch.Tensor | None = None,
    cav_executor_mask: torch.Tensor | None = None,
    cliprange: float = 0.2,
    cliprange_low: float | None = None,
    cliprange_high: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if cav_macro_ids is None or cav_budget_mask is None or cav_executor_mask is None:
        raise ValueError("CAV policy loss requires macro_ids, budget_mask, and executor_mask.")
    return _segment_ppo_loss(
        old_log_prob=old_log_prob,
        log_prob=log_prob,
        advantages=advantages,
        response_mask=response_mask,
        cav_macro_ids=cav_macro_ids,
        cav_budget_mask=cav_budget_mask,
        cav_executor_mask=cav_executor_mask,
        cliprange=cliprange,
        cliprange_low=cliprange_low,
        cliprange_high=cliprange_high,
    )


def _extract_from_kwargs(kwargs: dict[str, Any]) -> dict[str, torch.Tensor | None]:
    return {
        "cav_macro_ids": kwargs.get("cav_macro_ids"),
        "cav_budget_mask": kwargs.get("cav_budget_mask"),
        "cav_executor_mask": kwargs.get("cav_executor_mask"),
    }


def _get_tensor(container: Any, key: str) -> torch.Tensor | None:
    if isinstance(container, dict):
        value = container.get(key)
    elif hasattr(container, "batch"):
        value = getattr(container, "batch", {}).get(key)
    else:
        try:
            value = container[key]
        except Exception:
            value = None
    return value if isinstance(value, torch.Tensor) else None


def _extract_from_stack(reference: torch.Tensor) -> dict[str, torch.Tensor | None]:
    """Best-effort lookup for CAV masks in veRL worker micro-batch locals."""
    keys = ("cav_macro_ids", "cav_budget_mask", "cav_executor_mask")
    frame = inspect.currentframe()
    if frame is not None:
        frame = frame.f_back
    while frame is not None:
        for value in frame.f_locals.values():
            tensors = {key: _get_tensor(value, key) for key in keys}
            if all(tensor is not None and tensor.shape == reference.shape for tensor in tensors.values()):
                return tensors
        frame = frame.f_back
    return {key: None for key in keys}


def patch_core_algos_policy_loss() -> bool:
    """Patch veRL core policy loss when the installed version allows it.

    If veRL passes CAV masks into compute_policy_loss, the wrapper uses them
    directly. Otherwise it performs a conservative stack lookup for the current
    micro-batch and delegates to the original loss if masks are unavailable.
    """
    try:
        from verl.trainer.ppo import core_algos
    except Exception as exc:  # pragma: no cover - depends on external veRL.
        warnings.warn(f"CAV could not import veRL core_algos for policy-loss patch: {exc}")
        return False

    original = getattr(core_algos, "compute_policy_loss", None)
    if original is None:
        warnings.warn("CAV could not find verl.trainer.ppo.core_algos.compute_policy_loss.")
        return False
    if getattr(original, "_cav_patched", False):
        return True

    signature = inspect.signature(original)

    @wraps(original)
    def compute_policy_loss_with_cav(*args, **kwargs):
        masks = _extract_from_kwargs(kwargs)
        original_kwargs = {key: value for key, value in kwargs.items() if not key.startswith("cav_")}
        bound = signature.bind_partial(*args, **original_kwargs)
        old_log_prob = bound.arguments.get("old_log_prob")
        if old_log_prob is None:
            old_log_prob = bound.arguments.get("old_log_probs")
        log_prob = bound.arguments.get("log_prob")
        if log_prob is None:
            log_prob = bound.arguments.get("log_probs")
        advantages = bound.arguments.get("advantages")
        response_mask = bound.arguments.get("response_mask")
        if response_mask is None:
            response_mask = bound.arguments.get("eos_mask")

        if isinstance(log_prob, torch.Tensor) and not all(value is not None for value in masks.values()):
            masks = _extract_from_stack(log_prob)

        if (
            isinstance(old_log_prob, torch.Tensor)
            and isinstance(log_prob, torch.Tensor)
            and isinstance(advantages, torch.Tensor)
            and isinstance(response_mask, torch.Tensor)
            and all(value is not None for value in masks.values())
        ):
            cliprange = bound.arguments.get("cliprange", kwargs.get("cliprange", 0.2))
            cliprange_low = bound.arguments.get("cliprange_low", kwargs.get("cliprange_low"))
            cliprange_high = bound.arguments.get("cliprange_high", kwargs.get("cliprange_high"))
            return compute_cav_policy_loss(
                old_log_prob=old_log_prob,
                log_prob=log_prob,
                advantages=advantages,
                response_mask=response_mask,
                cliprange=float(cliprange),
                cliprange_low=None if cliprange_low is None else float(cliprange_low),
                cliprange_high=None if cliprange_high is None else float(cliprange_high),
                **masks,
            )
        return original(*args, **original_kwargs)

    compute_policy_loss_with_cav._cav_patched = True
    core_algos.compute_policy_loss = compute_policy_loss_with_cav
    return True


def patch_verl_policy_loss() -> None:
    patched = patch_core_algos_policy_loss()
    if not patched:
        warnings.warn(
            "CAV segment-level policy-loss patch was not installed. "
            "Training can still run, but veRL will use its original token-level PPO loss."
        )
