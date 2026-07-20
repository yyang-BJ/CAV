from __future__ import annotations

from typing import Any

import numpy as np


def _as_float_array(values) -> np.ndarray | None:
    if values is None:
        return None
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size == 0:
        return None
    return array


def compute_cav_metrics(batch, lambda_c: float | None = None) -> dict[str, float]:
    """Aggregate CAV-specific train/val metrics from reward-manager fields."""
    non_tensor = getattr(batch, "non_tensor_batch", None) or {}
    metrics: dict[str, float] = {}

    accuracy = _as_float_array(non_tensor.get("cav_accuracy"))
    if accuracy is not None:
        metrics["cav/accuracy"] = float(accuracy.mean())

    reason_tokens = _as_float_array(non_tensor.get("cav_actual_reason_tokens"))
    if reason_tokens is not None:
        metrics["cav/actual_reason_tokens/mean"] = float(reason_tokens.mean())
        metrics["cav/actual_reason_tokens/max"] = float(reason_tokens.max())
        metrics["cav/actual_reason_tokens/min"] = float(reason_tokens.min())

    allocated = _as_float_array(non_tensor.get("cav_allocated_budget"))
    if allocated is not None:
        metrics["cav/allocated_budget/mean"] = float(allocated.mean())
        metrics["cav/allocated_budget/max"] = float(allocated.max())
        metrics["cav/allocated_budget/min"] = float(allocated.min())

    valid_format = _as_float_array(non_tensor.get("cav_valid_format"))
    if valid_format is not None:
        metrics["cav/format_valid_rate"] = float(valid_format.mean())

    has_stop = _as_float_array(non_tensor.get("cav_has_stop"))
    if has_stop is not None:
        metrics["cav/stop_rate"] = float(has_stop.mean())

    overflow_count = _as_float_array(non_tensor.get("cav_overflow_count"))
    if overflow_count is not None:
        metrics["cav/overflow_rate"] = float((overflow_count > 0).mean())
        metrics["cav/overflow_count/mean"] = float(overflow_count.mean())

    parse_failed = non_tensor.get("cav_rollout_parse_failed")
    if parse_failed is not None:
        failed_arr = np.asarray(parse_failed, dtype=np.float64).reshape(-1)
        if failed_arr.size > 0:
            metrics["cav/rollout_parse_fail_rate"] = float(failed_arr.mean())

    stored_lambda = non_tensor.get("cav_lambda_c")
    if stored_lambda is not None:
        lambda_arr = _as_float_array(stored_lambda)
        if lambda_arr is not None:
            metrics["cav/lambda_c"] = float(lambda_arr.mean())
    elif lambda_c is not None:
        metrics["cav/lambda_c"] = float(lambda_c)

    return metrics


def patch_compute_data_metrics() -> None:
    """Append CAV metrics to veRL's compute_data_metrics output."""
    from verl.trainer.ppo import metric_utils
    from verl.trainer.ppo import ray_trainer

    if getattr(metric_utils.compute_data_metrics, "_cav_patched", False):
        return

    original = metric_utils.compute_data_metrics

    def compute_data_metrics_with_cav(batch, use_critic: bool = True) -> dict[str, Any]:
        metrics = original(batch=batch, use_critic=use_critic)
        metrics.update(compute_cav_metrics(batch))
        return metrics

    compute_data_metrics_with_cav._cav_patched = True
    metric_utils.compute_data_metrics = compute_data_metrics_with_cav

    # T3 fork also calls a local compute_data_metrics_ before the standard one.
    original_env = getattr(ray_trainer, "compute_data_metrics_", None)
    if original_env is not None and not getattr(original_env, "_cav_patched", False):

        def compute_data_metrics_env_with_cav(batch, use_critic: bool = True) -> dict[str, Any]:
            metrics = original_env(batch, use_critic=use_critic)
            metrics.update(compute_cav_metrics(batch))
            return metrics

        compute_data_metrics_env_with_cav._cav_patched = True
        ray_trainer.compute_data_metrics_ = compute_data_metrics_env_with_cav


def patch_ray_trainer_validate() -> None:
    """Make validation report accuracy and actual reasoning tokens.

    Upstream `_validate` names mean reward as `val/success_rate`, which is
    misleading for CAV. We keep that value as `val/reward_mean` and add the
    effectiveness/efficiency metrics collected by `CAVValidationRewardManager`.
    """
    from verl.trainer.ppo import ray_trainer

    original = ray_trainer.RayPPOTrainer._validate
    if getattr(original, "_cav_patched", False):
        return

    def _validate_with_cav(self):
        if hasattr(self.val_reward_fn, "reset_stats"):
            self.val_reward_fn.reset_stats()

        metric_dict = original(self)

        if "val/success_rate" in metric_dict:
            metric_dict["val/reward_mean"] = metric_dict.pop("val/success_rate")

        if hasattr(self.val_reward_fn, "pop_metrics"):
            metric_dict.update(self.val_reward_fn.pop_metrics())

        print(f"CAV validation metrics at step {self.global_steps}: {metric_dict}")
        return metric_dict

    _validate_with_cav._cav_patched = True
    ray_trainer.RayPPOTrainer._validate = _validate_with_cav
