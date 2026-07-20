from __future__ import annotations

from typing import Any, Sequence

import numpy as np


# Percentiles used to calibrate CAV target budget B from unconstrained CoT length.
LENGTH_PERCENTILES = (50, 70, 75, 90)


def _as_float_array(values) -> np.ndarray | None:
    if values is None:
        return None
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size == 0:
        return None
    return array


def _length_distribution_metrics(
    lengths: np.ndarray,
    *,
    prefix: str,
    percentiles: Sequence[int] = LENGTH_PERCENTILES,
) -> dict[str, float]:
    """Emit mean/min/max + median/Pxx for a length array under ``prefix``."""
    if lengths.size == 0:
        return {}
    out = {
        f"{prefix}/mean": float(lengths.mean()),
        f"{prefix}/min": float(lengths.min()),
        f"{prefix}/max": float(lengths.max()),
        f"{prefix}/median": float(np.median(lengths)),
        f"{prefix}/count": float(lengths.size),
    }
    # np.percentile: P50 == median; keep both names for clarity in logs.
    qs = np.percentile(lengths, list(percentiles))
    for p, value in zip(percentiles, qs):
        out[f"{prefix}/p{int(p)}"] = float(value)
    return out


def compute_baseline_metrics(batch) -> dict[str, float]:
    non_tensor = getattr(batch, "non_tensor_batch", None) or {}
    metrics: dict[str, float] = {}

    accuracy = _as_float_array(non_tensor.get("baseline_accuracy"))
    if accuracy is not None:
        metrics["baseline/accuracy"] = float(accuracy.mean())

    response_len = _as_float_array(non_tensor.get("baseline_response_length"))
    if response_len is not None:
        metrics.update(_length_distribution_metrics(response_len, prefix="baseline/response_length"))

        if accuracy is not None and accuracy.shape == response_len.shape:
            correct_mask = accuracy > 0.5
            correct_lens = response_len[correct_mask]
            wrong_lens = response_len[~correct_mask]
            metrics.update(
                _length_distribution_metrics(correct_lens, prefix="baseline/response_length_correct")
            )
            metrics.update(
                _length_distribution_metrics(wrong_lens, prefix="baseline/response_length_wrong")
            )

    strict_fmt = _as_float_array(non_tensor.get("baseline_has_strict_format"))
    if strict_fmt is not None:
        metrics["baseline/strict_format_rate"] = float(strict_fmt.mean())

    return metrics


def patch_compute_data_metrics_baseline() -> None:
    from verl.trainer.ppo import metric_utils
    from verl.trainer.ppo import ray_trainer

    if getattr(metric_utils.compute_data_metrics, "_baseline_patched", False):
        return

    original = metric_utils.compute_data_metrics

    def compute_data_metrics_with_baseline(batch, use_critic: bool = True) -> dict[str, Any]:
        metrics = original(batch=batch, use_critic=use_critic)
        metrics.update(compute_baseline_metrics(batch))
        return metrics

    compute_data_metrics_with_baseline._baseline_patched = True
    metric_utils.compute_data_metrics = compute_data_metrics_with_baseline

    original_env = getattr(ray_trainer, "compute_data_metrics_", None)
    if original_env is not None and not getattr(original_env, "_baseline_patched", False):

        def compute_data_metrics_env_with_baseline(batch, use_critic: bool = True) -> dict[str, Any]:
            metrics = original_env(batch, use_critic=use_critic)
            metrics.update(compute_baseline_metrics(batch))
            return metrics

        compute_data_metrics_env_with_baseline._baseline_patched = True
        ray_trainer.compute_data_metrics_ = compute_data_metrics_env_with_baseline
