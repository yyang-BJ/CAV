"""Dual update for the computation-price lambda_c.

lambda_dual <- [lambda + eta * (E[C] - B(t))]_+
lambda_eff  = s(t) * lambda_dual

where C = sum_k l_k (actual reasoning tokens), B(t) is an annealed budget
target, and s(t) is a cosine warmup scale on the cost coefficient.

Schedule endpoints are expressed as fractions of total_training_steps so the
same ratios work when T changes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class DualLambdaConfig:
    initial_lambda_c: float = 0.0005
    # Final budget target B_final (also used when B anneal is disabled).
    target_expected_tokens: float = 128.0
    dual_lr: float = 0.01
    min_lambda_c: float = 0.0
    max_lambda_c: float = 0.02
    enabled: bool = True
    # B anneal: B(t) from b_start -> target_expected_tokens over
    # b_anneal_ratio * total_training_steps. None => no anneal.
    b_start: float | None = None
    b_anneal_ratio: float = 0.7
    # lambda_eff = s(t) * lambda_dual; cosine s: 0 on [0,t0], 0→1 on (t0,t1], 1 after.
    # Ratios of total_training_steps. If end_ratio <= start_ratio, s(t)=1 always.
    lambda_scale_start_ratio: float = 0.0
    lambda_scale_end_ratio: float = 0.0
    total_training_steps: int = 100


class LambdaController:
    """Projected dual ascent with optional cosine λ-scale warmup and B anneal."""

    def __init__(self, config: DualLambdaConfig):
        self.value = float(config.initial_lambda_c)  # lambda_dual
        self.b_final = float(config.target_expected_tokens)
        self.b_start = None if config.b_start is None else float(config.b_start)
        self.b_anneal_ratio = float(config.b_anneal_ratio)
        self.lambda_scale_start_ratio = float(config.lambda_scale_start_ratio)
        self.lambda_scale_end_ratio = float(config.lambda_scale_end_ratio)
        self.total_training_steps = max(int(config.total_training_steps), 1)
        self.lr = float(config.dual_lr)
        self.min_value = float(config.min_lambda_c)
        self.max_value = float(config.max_lambda_c)
        self.enabled = bool(config.enabled)
        # Current constraint used by dual / dual_gap (starts at B(0)).
        self.target_expected_tokens = self.budget_at(0)
        self.last_scale = self.scale_at(0)
        self.last_effective = self.value * self.last_scale

    def _resolve_total_steps(self, total_steps: int | None) -> int:
        if total_steps is not None and int(total_steps) > 0:
            return int(total_steps)
        return self.total_training_steps

    def budget_at(self, global_step: int, total_steps: int | None = None) -> float:
        """Linear anneal B_start → B_final over b_anneal_ratio * T."""
        if self.b_start is None:
            return self.b_final
        t = max(int(global_step), 0)
        t_b = self.b_anneal_ratio * self._resolve_total_steps(total_steps)
        if t_b <= 0:
            return self.b_final
        if t >= t_b:
            return self.b_final
        u = t / t_b
        return float(self.b_start + u * (self.b_final - self.b_start))

    def scale_at(self, global_step: int, total_steps: int | None = None) -> float:
        """Cosine warmup of λ coefficient scale s(t) ∈ [0, 1]."""
        t0_ratio = self.lambda_scale_start_ratio
        t1_ratio = self.lambda_scale_end_ratio
        if t1_ratio <= t0_ratio:
            return 1.0
        t = max(int(global_step), 0)
        T = self._resolve_total_steps(total_steps)
        t0 = t0_ratio * T
        t1 = t1_ratio * T
        if t <= t0:
            return 0.0
        if t >= t1:
            return 1.0
        u = (t - t0) / max(t1 - t0, 1e-12)
        return float(0.5 * (1.0 - math.cos(math.pi * u)))

    def effective_lambda_at(self, global_step: int, total_steps: int | None = None) -> float:
        return float(self.value * self.scale_at(global_step, total_steps))

    def update(
        self,
        observed_tokens: float,
        global_step: int | None = None,
        total_steps: int | None = None,
    ) -> float:
        """Update dual with B(t); return λ_eff = s(t) * λ_dual for the reward.

        If ``global_step`` is None (legacy callers), use B_final and s=1.
        """
        if global_step is None:
            budget = self.b_final
            scale = 1.0
        else:
            step = int(global_step)
            budget = self.budget_at(step, total_steps)
            scale = self.scale_at(step, total_steps)

        self.target_expected_tokens = float(budget)
        self.last_scale = float(scale)

        if self.enabled:
            self.value += self.lr * (float(observed_tokens) - budget)
            self.value = min(max(self.value, self.min_value), self.max_value)

        self.last_effective = float(self.value * scale)
        return self.last_effective

    @property
    def dual_gap(self) -> float:
        """Last observed gap is not stored; callers should compute E[C]-B directly."""
        return 0.0
