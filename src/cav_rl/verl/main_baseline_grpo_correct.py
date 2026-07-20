"""GRPO-correct on GSM8K: outcome + correct-only rank length bonus (no critic)."""

from __future__ import annotations

import os
import socket

import hydra
import ray
from omegaconf import OmegaConf

from cav_rl.verl.baseline_metrics import patch_compute_data_metrics_baseline
from cav_rl.verl.grpo_correct_reward import (
    GrpoCorrectRewardConfig,
    GrpoCorrectRewardManager,
    GrpoCorrectValidationRewardManager,
    compute_grpo_correct_train_metrics,
)
from cav_rl.verl.metrics import patch_ray_trainer_validate
from cav_rl.verl.sampling_patch import apply_vllm_sampling_patch, get_cav_actor_rollout_cls
from cav_rl.verl.single_turn import patch_ray_trainer_single_turn


@hydra.main(config_path=None, config_name=None, version_base=None)
def main(config):
    run_grpo_correct(config)


def _get_ppo_ray_runtime_env():
    try:
        from verl.trainer.constants_ppo import get_ppo_ray_runtime_env

        return get_ppo_ray_runtime_env()
    except ImportError:
        return {
            "env_vars": {
                "TOKENIZERS_PARALLELISM": "true",
                "NCCL_DEBUG": "WARN",
                "VLLM_LOGGING_LEVEL": "WARN",
                "VLLM_ALLOW_RUNTIME_LORA_UPDATING": "true",
            }
        }


def _wants_critic(config) -> bool:
    critic_cfg = config.get("critic", {}) or {}
    enable = critic_cfg.get("enable", None)
    if enable is not None:
        return bool(enable)
    adv = str(config.algorithm.adv_estimator).lower()
    return adv in {"gae"}


def _patch_compute_data_metrics_grpo_correct() -> None:
    """Layer grpo_correct train metrics on top of the baseline metric patch."""
    from verl.trainer.ppo import metric_utils
    from verl.trainer.ppo import ray_trainer

    if getattr(metric_utils.compute_data_metrics, "_grpo_correct_patched", False):
        return

    original = metric_utils.compute_data_metrics

    def compute_data_metrics_with_grpo_correct(batch, use_critic: bool = True):
        metrics = original(batch=batch, use_critic=use_critic)
        metrics.update(compute_grpo_correct_train_metrics(batch))
        return metrics

    compute_data_metrics_with_grpo_correct._grpo_correct_patched = True
    # Preserve baseline patch marker if present.
    if getattr(original, "_baseline_patched", False):
        compute_data_metrics_with_grpo_correct._baseline_patched = True
    metric_utils.compute_data_metrics = compute_data_metrics_with_grpo_correct

    original_env = getattr(ray_trainer, "compute_data_metrics_", None)
    if original_env is not None and not getattr(original_env, "_grpo_correct_patched", False):

        def compute_data_metrics_env_with_grpo_correct(batch, use_critic: bool = True):
            metrics = original_env(batch, use_critic=use_critic)
            metrics.update(compute_grpo_correct_train_metrics(batch))
            return metrics

        compute_data_metrics_env_with_grpo_correct._grpo_correct_patched = True
        if getattr(original_env, "_baseline_patched", False):
            compute_data_metrics_env_with_grpo_correct._baseline_patched = True
        ray_trainer.compute_data_metrics_ = compute_data_metrics_env_with_grpo_correct


def run_grpo_correct(config) -> None:
    if not ray.is_initialized():
        num_cpus = None
        if hasattr(config, "ray_init") and config.ray_init is not None:
            num_cpus = config.ray_init.get("num_cpus", None)
        ray.init(runtime_env=_get_ppo_ray_runtime_env(), num_cpus=num_cpus)
    runner = BaselineGRPOCorrectTaskRunner.remote()
    ray.get(runner.run.remote(config))

    if hasattr(config, "ray_init") and config.ray_init is not None:
        timeline_json_file = config.ray_init.get("timeline_json_file", None)
        if timeline_json_file:
            ray.timeline(filename=timeline_json_file)


@ray.remote(num_cpus=1)
class BaselineGRPOCorrectTaskRunner:
    def run(self, config):
        from pprint import pprint

        import torch
        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler
        from verl.trainer.ppo.ray_trainer import RayPPOTrainer, ResourcePoolManager, Role
        from verl.utils import hf_tokenizer
        from verl.utils.dataset.rl_dataset import collate_fn
        from verl.utils.fs import copy_to_local

        print(
            f"BaselineGRPOCorrectTaskRunner hostname: {socket.gethostname()}, PID: {os.getpid()}"
        )
        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)

        patch_ray_trainer_single_turn()
        patch_compute_data_metrics_baseline()
        _patch_compute_data_metrics_grpo_correct()
        patch_ray_trainer_validate()
        apply_vllm_sampling_patch()

        local_path = copy_to_local(
            config.actor_rollout_ref.model.path,
            use_shm=config.actor_rollout_ref.model.get("use_shm", False),
        )
        trust_remote_code = config.data.get("trust_remote_code", True)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        processor = None

        use_critic = _wants_critic(config)
        if str(config.algorithm.adv_estimator).lower() != "grpo":
            print(
                f"[grpo-correct] WARNING: algorithm.adv_estimator="
                f"{config.algorithm.adv_estimator!r} (expected 'grpo')",
                flush=True,
            )
        if int(config.actor_rollout_ref.rollout.n) < 2:
            print(
                "[grpo-correct] WARNING: rollout.n < 2; GRPO needs group sampling "
                "(set actor_rollout_ref.rollout.n >= 2).",
                flush=True,
            )

        if config.actor_rollout_ref.actor.strategy in {"fsdp", "fsdp2"}:
            from verl.single_controller.ray import RayWorkerGroup
            from verl.workers.fsdp_workers import AsyncActorRolloutRefWorker

            use_legacy_worker_impl = config.trainer.get("use_legacy_worker_impl", "auto")
            CriticWorker = None
            if use_critic:
                assert config.critic.strategy in {"fsdp", "fsdp2"}
                if use_legacy_worker_impl in ["auto", "enable"]:
                    from verl.workers.fsdp_workers import CriticWorker
                else:
                    from verl.workers.roles import CriticWorker

            actor_rollout_cls = (
                AsyncActorRolloutRefWorker
                if config.actor_rollout_ref.rollout.get("mode", "sync") == "async"
                else get_cav_actor_rollout_cls()
            )
            ray_worker_group_cls = RayWorkerGroup
        elif config.actor_rollout_ref.actor.strategy == "megatron":
            from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
            from verl.workers.megatron_workers import ActorRolloutRefWorker, AsyncActorRolloutRefWorker

            CriticWorker = None
            if use_critic:
                assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
                from verl.workers.megatron_workers import CriticWorker

            actor_rollout_cls = (
                AsyncActorRolloutRefWorker
                if config.actor_rollout_ref.rollout.get("mode", "sync") == "async"
                else ActorRolloutRefWorker
            )
            ray_worker_group_cls = NVMegatronRayWorkerGroup
        else:
            raise NotImplementedError(f"Unsupported actor strategy: {config.actor_rollout_ref.actor.strategy}")

        role_worker_mapping = {
            Role.ActorRollout: ray.remote(actor_rollout_cls),
        }
        global_pool_id = "global_pool"
        resource_pool_spec = {global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes}
        mapping = {
            Role.ActorRollout: global_pool_id,
        }

        if use_critic:
            role_worker_mapping[Role.Critic] = ray.remote(CriticWorker)
            mapping[Role.Critic] = global_pool_id

        if config.algorithm.use_kl_in_reward or config.actor_rollout_ref.actor.use_kl_loss:
            role_worker_mapping[Role.RefPolicy] = ray.remote(actor_rollout_cls)
            mapping[Role.RefPolicy] = global_pool_id

        if config.reward_model.enable:
            if config.reward_model.strategy in {"fsdp", "fsdp2"}:
                from verl.workers.fsdp_workers import RewardModelWorker
            elif config.reward_model.strategy == "megatron":
                from verl.workers.megatron_workers import RewardModelWorker
            else:
                raise NotImplementedError(f"Unsupported reward model strategy: {config.reward_model.strategy}")
            role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
            mapping[Role.RewardModel] = global_pool_id

        gc_cfg = config.get("grpo_correct", {}) or {}
        reward_cfg = GrpoCorrectRewardConfig(
            correct_reward=float(gc_cfg.get("correct_reward", 1.0)),
            wrong_reward=float(gc_cfg.get("wrong_reward", -0.5)),
            unparsable_reward=float(gc_cfg.get("unparsable_reward", -1.0)),
            strict_format_bonus=float(gc_cfg.get("strict_format_bonus", 0.2)),
            missing_format_penalty=float(gc_cfg.get("missing_format_penalty", -0.2)),
            length_bonus_weight=float(gc_cfg.get("length_bonus_weight", 0.01)),
            length_floor=float(gc_cfg.get("length_floor", 270.0)),
            length_floor_softness=float(gc_cfg.get("length_floor_softness", 40.0)),
            length_excess_ref=float(gc_cfg.get("length_excess_ref", 0.0)),
            length_score_max=float(gc_cfg.get("length_score_max", 1.0)),
            length_score_min=float(gc_cfg.get("length_score_min", 0.0)),
            length_score_neutral=float(gc_cfg.get("length_score_neutral", 0.0)),
            min_correct_for_length_ranking=int(gc_cfg.get("min_correct_for_length_ranking", 2)),
            extract_method=str(gc_cfg.get("extract_method", "flexible")),
            debug_print_responses=str(gc_cfg.get("debug_print_responses", False)).lower()
            in {"1", "true", "yes", "y"},
        )
        reward_fn = GrpoCorrectRewardManager(tokenizer=tokenizer, reward_config=reward_cfg)
        val_reward_fn = GrpoCorrectValidationRewardManager(tokenizer=tokenizer, reward_config=reward_cfg)

        resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)
        try:
            train_dataset = create_rl_dataset(
                config.data.train_files, config.data, tokenizer, processor, is_train=True
            )
            val_dataset = create_rl_dataset(
                config.data.val_files, config.data, tokenizer, processor, is_train=False
            )
        except TypeError:
            train_dataset = create_rl_dataset(config.data.train_files, config.data, tokenizer, processor)
            val_dataset = create_rl_dataset(config.data.val_files, config.data, tokenizer, processor)
        train_sampler = create_rl_sampler(config.data, train_dataset)

        trainer = RayPPOTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            collate_fn=collate_fn,
            train_sampler=train_sampler,
        )
        torch.set_float32_matmul_precision("high")
        trainer.lambda_controller = None
        print(
            f"[grpo-correct] use_critic={use_critic} "
            f"adv={config.algorithm.adv_estimator} rollout.n={config.actor_rollout_ref.rollout.n} "
            f"length_bonus_weight={reward_cfg.length_bonus_weight} "
            f"wrong_reward={reward_cfg.wrong_reward}",
            flush=True,
        )
        trainer.init_workers()
        trainer.fit()


if __name__ == "__main__":
    main()
