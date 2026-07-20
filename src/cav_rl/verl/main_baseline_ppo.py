"""Plain PPO baseline on GSM8K CoT (no CAV hierarchy / dual lambda / cav_gae)."""

from __future__ import annotations

import os
import socket

import hydra
import ray
from omegaconf import OmegaConf

from cav_rl.verl.baseline_metrics import patch_compute_data_metrics_baseline
from cav_rl.verl.baseline_reward import (
    BaselineRewardConfig,
    BaselineRewardManager,
    BaselineValidationRewardManager,
)
from cav_rl.verl.metrics import patch_ray_trainer_validate
from cav_rl.verl.sampling_patch import apply_vllm_sampling_patch, get_cav_actor_rollout_cls
from cav_rl.verl.single_turn import patch_ray_trainer_single_turn


@hydra.main(config_path=None, config_name=None, version_base=None)
def main(config):
    run_ppo(config)


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


def run_ppo(config) -> None:
    if not ray.is_initialized():
        num_cpus = None
        if hasattr(config, "ray_init") and config.ray_init is not None:
            num_cpus = config.ray_init.get("num_cpus", None)
        ray.init(runtime_env=_get_ppo_ray_runtime_env(), num_cpus=num_cpus)
    runner = BaselineTaskRunner.remote()
    ray.get(runner.run.remote(config))

    if hasattr(config, "ray_init") and config.ray_init is not None:
        timeline_json_file = config.ray_init.get("timeline_json_file", None)
        if timeline_json_file:
            ray.timeline(filename=timeline_json_file)


@ray.remote(num_cpus=1)
class BaselineTaskRunner:
    def run(self, config):
        from pprint import pprint

        import torch
        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler
        from verl.trainer.ppo.ray_trainer import RayPPOTrainer, ResourcePoolManager, Role
        from verl.utils import hf_tokenizer
        from verl.utils.dataset.rl_dataset import collate_fn
        from verl.utils.fs import copy_to_local

        print(f"BaselineTaskRunner hostname: {socket.gethostname()}, PID: {os.getpid()}")
        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)

        # One-shot rollout via the same single-turn fit path; hierarchical stays off.
        patch_ray_trainer_single_turn()
        patch_compute_data_metrics_baseline()
        patch_ray_trainer_validate()
        apply_vllm_sampling_patch()

        local_path = copy_to_local(
            config.actor_rollout_ref.model.path,
            use_shm=config.actor_rollout_ref.model.get("use_shm", False),
        )
        trust_remote_code = config.data.get("trust_remote_code", True)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        processor = None

        if config.actor_rollout_ref.actor.strategy in {"fsdp", "fsdp2"}:
            assert config.critic.strategy in {"fsdp", "fsdp2"}
            from verl.single_controller.ray import RayWorkerGroup
            from verl.workers.fsdp_workers import AsyncActorRolloutRefWorker

            use_legacy_worker_impl = config.trainer.get("use_legacy_worker_impl", "auto")
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
            assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
            from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
            from verl.workers.megatron_workers import ActorRolloutRefWorker, AsyncActorRolloutRefWorker, CriticWorker

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
            Role.Critic: ray.remote(CriticWorker),
        }
        global_pool_id = "global_pool"
        resource_pool_spec = {global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes}
        mapping = {
            Role.ActorRollout: global_pool_id,
            Role.Critic: global_pool_id,
        }

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

        baseline_cfg = config.get("baseline", {}) or {}
        reward_cfg = BaselineRewardConfig(
            correct_reward=float(baseline_cfg.get("correct_reward", 1.0)),
            wrong_reward=float(baseline_cfg.get("wrong_reward", 0.0)),
            format_score=float(baseline_cfg.get("format_score", 0.0)),
            extract_method=str(baseline_cfg.get("extract_method", "flexible")),
            debug_print_responses=str(baseline_cfg.get("debug_print_responses", False)).lower()
            in {"1", "true", "yes", "y"},
        )
        reward_fn = BaselineRewardManager(tokenizer=tokenizer, reward_config=reward_cfg)
        val_reward_fn = BaselineValidationRewardManager(tokenizer=tokenizer, reward_config=reward_cfg)

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
        # No dual-lambda controller for the outcome-only baseline.
        trainer.lambda_controller = None
        trainer.init_workers()
        trainer.fit()


if __name__ == "__main__":
    main()
