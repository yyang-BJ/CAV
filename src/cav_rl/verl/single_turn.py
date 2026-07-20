"""Replace T3's multi-turn agent rollout loop with standard single-turn PPO.

The shared T3 veRL fork routes every PPO step through LLMGenerationManager and
dataset-specific controllers. CAV/GSM8K only needs actor_rollout_wg.generate_sequences,
optionally wrapped by hierarchical macro-step rollout.
"""

from __future__ import annotations

import uuid
from copy import deepcopy
from pprint import pprint

import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm


def _gen_non_tensor_keys(batch) -> list[str]:
    keys = []
    for key in (
        "raw_prompt_ids",
        "multi_modal_data",
        "raw_prompt",
        "tools_kwargs",
        "interaction_kwargs",
        "agent_name",
        "index",
    ):
        if key in batch.non_tensor_batch:
            keys.append(key)
    return keys


def _generate(self, gen_batch):
    if getattr(self, "async_rollout_mode", False):
        return self.async_rollout_manager.generate_sequences(gen_batch)
    return self.actor_rollout_wg.generate_sequences(gen_batch)


def _generate_cav(self, gen_batch):
    """Hierarchical macro rollout when enabled; otherwise one-shot generate_sequences."""
    from cav_rl.verl.hierarchical_rollout import generate_hierarchical_sequences, hierarchical_enabled

    if not hierarchical_enabled(self.config):
        return _generate(self, gen_batch)

    cav = self.config.get("cav", {}) or {}
    allowed_budgets = list(cav.get("budget_actions", [0, 16, 32, 64, 128]))
    max_macro_steps = int(cav.get("max_macro_steps", 6))
    max_response_length = int(self.config.data.max_response_length)
    max_model_len = int(
        self.config.actor_rollout_ref.rollout.get(
            "max_model_len",
            self.config.data.max_prompt_length + max_response_length,
        )
    )
    return generate_hierarchical_sequences(
        gen_batch,
        lambda batch: _generate(self, batch),
        self.tokenizer,
        max_response_length=max_response_length,
        max_model_len=max_model_len,
        allowed_budgets=allowed_budgets,
        max_macro_steps=max_macro_steps,
        budget_max_tokens=int(cav.get("budget_max_tokens", 64)),
        answer_max_tokens=int(cav.get("answer_max_tokens", 96)),
        parse_fail_keep_tokens=int(cav.get("parse_fail_keep_tokens", 32)),
    )


def _update_dual_lambda(self, batch, metrics: dict) -> None:
    controller = getattr(self, "lambda_controller", None)
    if controller is None or not getattr(controller, "enabled", False):
        return
    reason = batch.non_tensor_batch.get("cav_actual_reason_tokens")
    if reason is None:
        return
    mean_c = float(np.asarray(reason, dtype=np.float64).mean())
    global_step = int(getattr(self, "global_steps", 0))
    total_steps = None
    try:
        total_steps = int(self.config.trainer.total_training_steps)
    except Exception:
        total_steps = getattr(controller, "total_training_steps", None)
    # λ_eff = s(t)*λ_dual; dual uses annealed B(t).
    new_lambda = controller.update(mean_c, global_step=global_step, total_steps=total_steps)
    if hasattr(self, "reward_fn") and hasattr(self.reward_fn, "reward_config"):
        self.reward_fn.reward_config.lambda_c = new_lambda
        self.reward_fn.reward_config.target_expected_tokens = float(controller.target_expected_tokens)
    if hasattr(self, "val_reward_fn") and self.val_reward_fn is not None and hasattr(
        self.val_reward_fn, "reward_config"
    ):
        self.val_reward_fn.reward_config.lambda_c = new_lambda
        self.val_reward_fn.reward_config.target_expected_tokens = float(controller.target_expected_tokens)
    metrics["cav/lambda_c"] = float(new_lambda)  # effective (enters reward)
    metrics["cav/lambda_dual"] = float(controller.value)
    metrics["cav/lambda_scale"] = float(getattr(controller, "last_scale", 1.0))
    metrics["cav/B_t"] = float(controller.target_expected_tokens)
    metrics["cav/dual_gap"] = float(mean_c - controller.target_expected_tokens)


def _validate_single_turn(self):
    from verl import DataProto
    from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
    from verl.trainer.ppo.ray_trainer import compute_response_mask

    from cav_rl.verl.baseline_reward import BaselineRewardManager
    from cav_rl.verl.case_dump import (
        collect_case_records,
        dump_case_records,
        resolve_case_dump_dir,
        summarize_error_types,
    )
    from cav_rl.verl.grpo_correct_reward import GrpoCorrectRewardManager

    reward_tensor_lst = []
    data_source_lst = []
    case_records: list[dict] = []
    world_size = int(getattr(self.actor_rollout_wg, "world_size", 1) or 1)
    dump_dir = resolve_case_dump_dir(self.config)
    mode = (
        "baseline"
        if isinstance(self.val_reward_fn, (BaselineRewardManager, GrpoCorrectRewardManager))
        else "cav"
    )
    cav_cfg = self.config.get("cav", {}) or {}
    allowed_budgets = list(cav_cfg.get("budget_actions", [0, 16, 32, 64, 128]))

    for batch_dict in self.val_dataloader:
        test_batch: DataProto = DataProto.from_single_dict(batch_dict)
        test_gen_batch = test_batch.pop(
            batch_keys=["input_ids", "attention_mask", "position_ids"],
            non_tensor_batch_keys=_gen_non_tensor_keys(test_batch),
        )
        test_gen_batch.meta_info = {
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
            "recompute_log_prob": False,
            "do_sample": False,
            "validate": True,
            "global_steps": self.global_steps,
        }
        # Last val batch (e.g. 1319 % 64 = 39) must be divisible by DP world size.
        test_gen_batch, pad_size = pad_dataproto_to_divisor(test_gen_batch, world_size)
        test_output = _generate_cav(self, test_gen_batch)
        test_output = unpad_dataproto(test_output, pad_size=pad_size)
        if "timing" in test_output.meta_info:
            test_output.meta_info.pop("timing", None)
        test_batch = test_batch.union(test_output)
        if "response_mask" not in test_batch.batch:
            test_batch.batch["response_mask"] = compute_response_mask(test_batch)

        reward_tensor = self.val_reward_fn(test_batch)
        reward_tensor_lst.append(reward_tensor)
        data_source_lst.append(
            test_batch.non_tensor_batch.get("data_source", ["unknown"] * reward_tensor.shape[0])
        )
        if dump_dir is not None:
            case_records.extend(
                collect_case_records(
                    test_batch,
                    self.tokenizer,
                    mode=mode,
                    allowed_budgets=allowed_budgets,
                    global_step=int(self.global_steps),
                )
            )

    reward_tensor = torch.cat([rw.sum(-1) for rw in reward_tensor_lst], dim=0).float().cpu()
    metric_dict = {
        "val/success_rate": float(torch.mean(reward_tensor).item()),
        "val/reward_mean": float(torch.mean(reward_tensor).item()),
    }
    if dump_dir is not None and case_records:
        dump_case_records(case_records, dump_dir, global_step=int(self.global_steps), also_bad_only=True)
        metric_dict.update(summarize_error_types(case_records))
    print(f"CAV single-turn validation at step {self.global_steps}: {metric_dict}")
    return metric_dict


def _fit_single_turn(self):
    from verl import DataProto
    from verl.experimental.dataset.sampler import AbstractCurriculumSampler
    from verl.trainer.ppo.core_algos import AdvantageEstimator, agg_loss
    from verl.trainer.ppo.metric_utils import (
        compute_data_metrics,
        compute_throughout_metrics,
        compute_timing_metrics,
    )
    from verl.trainer.ppo.ray_trainer import (
        apply_kl_penalty,
        compute_advantage,
        compute_data_metrics_,
        compute_response_mask,
    )
    from verl.utils.checkpoint.checkpoint_manager import should_save_ckpt_esi
    from verl.utils.debug import marked_timer
    from verl.utils.metric import reduce_metrics
    from verl.utils.tracking import Tracking

    logger = Tracking(
        project_name=self.config.trainer.project_name,
        experiment_name=self.config.trainer.experiment_name,
        default_backend=self.config.trainer.logger,
        config=OmegaConf.to_container(self.config, resolve=True),
    )

    self.global_steps = 0
    self._load_checkpoint()

    if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
        val_metrics = self._validate()
        assert val_metrics, f"{val_metrics=}"
        pprint(f"Initial validation metrics: {val_metrics}")
        logger.log(data=val_metrics, step=self.global_steps)
        if self.config.trainer.get("val_only", False):
            return

    progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")
    self.global_steps += 1
    last_val_metrics = None
    self.max_steps_duration = 0
    self.best_reward = float("-inf")

    total_epochs = int(self.config.trainer.get("total_epochs", 30))
    for epoch in range(total_epochs):
        for batch_dict in self.train_dataloader:
            print(f"[CAV] epoch {epoch}, step {self.global_steps}")
            metrics = {}
            timing_raw = {}

            do_profile = (
                self.global_steps in self.config.trainer.profile_steps
                if self.config.trainer.profile_steps is not None
                else False
            )
            with marked_timer("start_profile", timing_raw):
                self._start_profiling(do_profile)

            batch: DataProto = DataProto.from_single_dict(batch_dict)
            batch.batch["input_ids"] = batch.batch["input_ids"].long()
            gen_batch = batch.pop(
                batch_keys=["input_ids", "attention_mask", "position_ids"],
                non_tensor_batch_keys=_gen_non_tensor_keys(batch),
            )
            gen_batch.meta_info["global_steps"] = self.global_steps
            gen_batch = gen_batch.repeat(
                repeat_times=self.config.actor_rollout_ref.rollout.n,
                interleave=True,
            )

            is_last_step = self.global_steps >= self.total_training_steps

            with marked_timer("step", timing_raw):
                with marked_timer("gen", timing_raw, color="red"):
                    print(
                        f"[CAV] step {self.global_steps}: start hierarchical/one-shot generate "
                        f"batch={len(gen_batch)} ...",
                        flush=True,
                    )
                    gen_batch_output = _generate_cav(self, gen_batch)
                    print(f"[CAV] step {self.global_steps}: generate done", flush=True)
                    if "timing" in gen_batch_output.meta_info:
                        timing_raw.update(gen_batch_output.meta_info["timing"])
                        gen_batch_output.meta_info.pop("timing", None)

                if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                    with marked_timer("gen_max", timing_raw, color="purple"):
                        gen_baseline_batch = deepcopy(gen_batch)
                        gen_baseline_batch.meta_info["do_sample"] = False
                        gen_baseline_output = _generate_cav(self, gen_baseline_batch)
                        batch = batch.union(gen_baseline_output)
                        reward_baseline = self.reward_fn(batch)
                        if isinstance(reward_baseline, tuple):
                            reward_baseline = reward_baseline[0]
                        reward_baseline_tensor = reward_baseline.sum(dim=-1)
                        batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))
                        batch.batch["reward_baselines"] = reward_baseline_tensor
                        del gen_baseline_batch, gen_baseline_output

                batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(batch.batch))],
                    dtype=object,
                )
                batch = batch.repeat(
                    repeat_times=self.config.actor_rollout_ref.rollout.n,
                    interleave=True,
                )
                batch = batch.union(gen_batch_output)

                if "response_mask" not in batch.batch.keys():
                    batch.batch["response_mask"] = compute_response_mask(batch)
                else:
                    response_length = batch.batch["responses"].shape[-1]
                    batch.batch["response_mask"] = batch.batch["response_mask"][:, -response_length:]

                if self.config.trainer.balance_batch:
                    self._balance_batch(batch, metrics=metrics)

                batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()
                batch.meta_info["global_steps"] = self.global_steps

                with marked_timer("old_log_prob", timing_raw, color="blue"):
                    print(f"[CAV] step {self.global_steps}: compute_log_prob ...", flush=True)
                    batch.batch["input_ids"] = batch.batch["input_ids"].long()
                    old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                    print(f"[CAV] step {self.global_steps}: compute_log_prob done", flush=True)
                    entropys = old_log_prob.batch["entropys"]
                    response_masks = batch.batch["response_mask"]
                    loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                    entropy_agg = agg_loss(
                        loss_mat=entropys,
                        loss_mask=response_masks,
                        loss_agg_mode=loss_agg_mode,
                    )
                    metrics.update({"actor/entropy": entropy_agg.detach().item()})
                    old_log_prob.batch.pop("entropys")
                    batch = batch.union(old_log_prob)

                if self.use_reference_policy:
                    with marked_timer("ref", timing_raw, color="olive"):
                        print(f"[CAV] step {self.global_steps}: compute_ref_log_prob ...", flush=True)
                        if not self.ref_in_actor:
                            ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                        else:
                            ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
                        print(f"[CAV] step {self.global_steps}: compute_ref_log_prob done", flush=True)
                        batch = batch.union(ref_log_prob)

                if self.use_critic:
                    with marked_timer("values", timing_raw, color="cyan"):
                        print(f"[CAV] step {self.global_steps}: compute_values ...", flush=True)
                        values = self.critic_wg.compute_values(batch)
                        print(f"[CAV] step {self.global_steps}: compute_values done", flush=True)
                        batch = batch.union(values)

                with marked_timer("adv", timing_raw, color="brown"):
                    print(f"[CAV] step {self.global_steps}: reward + advantage ...", flush=True)
                    reward_out = self.reward_fn(batch)
                    if isinstance(reward_out, tuple):
                        reward_tensor, final_rewards = reward_out
                        batch.non_tensor_batch["final_rewards"] = final_rewards
                    else:
                        reward_tensor = reward_out
                    batch.batch["token_level_scores"] = reward_tensor

                    if self.config.algorithm.use_kl_in_reward:
                        batch, kl_metrics = apply_kl_penalty(
                            batch,
                            kl_ctrl=self.kl_ctrl_in_reward,
                            kl_penalty=self.config.algorithm.kl_penalty,
                        )
                        metrics.update(kl_metrics)
                    else:
                        batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                    norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)
                    batch = compute_advantage(
                        batch,
                        adv_estimator=self.config.algorithm.adv_estimator,
                        gamma=self.config.algorithm.gamma,
                        lam=self.config.algorithm.lam,
                        num_repeat=self.config.actor_rollout_ref.rollout.n,
                        norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                        config=self.config.algorithm,
                        save_adv_name=self.config.trainer.experiment_name,
                    )
                    print(f"[CAV] step {self.global_steps}: reward + advantage done", flush=True)

                _update_dual_lambda(self, batch, metrics)

                if self.use_critic:
                    with marked_timer("update_critic", timing_raw, color="pink"):
                        print(f"[CAV] step {self.global_steps}: update_critic ...", flush=True)
                        critic_output = self.critic_wg.update_critic(batch)
                        print(f"[CAV] step {self.global_steps}: update_critic done", flush=True)
                    metrics.update(reduce_metrics(critic_output.meta_info["metrics"]))

                if self.config.trainer.critic_warmup <= self.global_steps:
                    with marked_timer("update_actor", timing_raw, color="red"):
                        print(f"[CAV] step {self.global_steps}: update_actor ...", flush=True)
                        batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                        actor_output = self.actor_rollout_wg.update_actor(batch)
                        print(f"[CAV] step {self.global_steps}: update_actor done", flush=True)
                    metrics.update(reduce_metrics(actor_output.meta_info["metrics"]))

                if (
                    self.val_reward_fn is not None
                    and self.config.trainer.test_freq > 0
                    and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
                ):
                    with marked_timer("testing", timing_raw, color="green"):
                        val_metrics = self._validate()
                        if is_last_step:
                            last_val_metrics = val_metrics
                    metrics.update(val_metrics)

                esi_close_to_expiration = should_save_ckpt_esi(
                    max_steps_duration=self.max_steps_duration,
                    redundant_time=self.config.trainer.esi_redundant_time,
                )
                if self.config.trainer.save_freq > 0 and self.global_steps > 2 and (
                    is_last_step
                    or self.global_steps % self.config.trainer.save_freq == 0
                    or esi_close_to_expiration
                ):
                    with marked_timer("save_checkpoint", timing_raw, color="green"):
                        self._save_checkpoint()

            with marked_timer("stop_profile", timing_raw):
                self._stop_profiling(do_profile)

            steps_duration = timing_raw["step"]
            self.max_steps_duration = max(self.max_steps_duration, steps_duration)

            metrics.update(
                {
                    "training/global_step": self.global_steps,
                    "training/epoch": epoch,
                }
            )
            metrics.update(compute_data_metrics_(batch=batch))
            metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
            metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
            n_gpus = self.resource_pool_manager.get_n_gpus()
            metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))

            if isinstance(self.train_dataloader.sampler, AbstractCurriculumSampler):
                self.train_dataloader.sampler.update(batch=batch)

            # T3's best-reward path triggers a full validation whenever the score
            # improves. That is far too expensive for GSM8K; rely on test_freq instead.
            score_mean = metrics.get("critic/score/mean", None)
            if score_mean is not None and score_mean > self.best_reward:
                self.best_reward = float(score_mean)

            print(
                f"[CAV] step {self.global_steps} metrics: "
                f"score={metrics.get('critic/score/mean')} "
                f"entropy={metrics.get('actor/entropy')} "
                f"cav_acc={metrics.get('cav/accuracy')} "
                f"lambda_c={metrics.get('cav/lambda_c')}",
                flush=True,
            )
            logger.log(data=metrics, step=self.global_steps)
            progress_bar.update(1)
            self.global_steps += 1

            if is_last_step:
                pprint(f"Final validation metrics: {last_val_metrics}")
                progress_bar.close()
                return

            if hasattr(self.train_dataset, "on_batch_end"):
                self.train_dataset.on_batch_end(batch=batch)


def patch_ray_trainer_single_turn() -> None:
    """Swap T3 multi-turn fit/validate for CAV single-turn generate_sequences."""
    from verl.trainer.ppo import ray_trainer

    if getattr(ray_trainer.RayPPOTrainer.fit, "_cav_single_turn", False):
        return

    _fit_single_turn._cav_single_turn = True
    _validate_single_turn._cav_single_turn = True
    ray_trainer.RayPPOTrainer.fit = _fit_single_turn
    ray_trainer.RayPPOTrainer._validate = _validate_single_turn
