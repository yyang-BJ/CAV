from __future__ import annotations

import os
import socket

import hydra
import ray
from omegaconf import OmegaConf

from cav_rl.verl.advantage import patch_ray_trainer_compute_advantage, register_cav_advantage
from cav_rl.verl.reward import CAVRewardConfig, CAVRewardManager, CAVValidationRewardManager


@hydra.main(config_path=None, config_name=None, version_base=None)
def main(config):
    run_ppo(config)


def run_ppo(config) -> None:
    from verl.trainer.constants_ppo import get_ppo_ray_runtime_env

    if not ray.is_initialized():
        ray.init(runtime_env=get_ppo_ray_runtime_env(), num_cpus=config.ray_init.num_cpus)
    runner = CAVTaskRunner.remote()
    ray.get(runner.run.remote(config))

    timeline_json_file = config.ray_init.get("timeline_json_file", None)
    if timeline_json_file:
        ray.timeline(filename=timeline_json_file)


@ray.remote(num_cpus=1)
class CAVTaskRunner:
    def run(self, config):
        from pprint import pprint

        import torch
        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler
        from verl.trainer.ppo.ray_trainer import RayPPOTrainer, ResourcePoolManager, Role
        from verl.utils import hf_tokenizer
        from verl.utils.dataset.rl_dataset import collate_fn
        from verl.utils.fs import copy_to_local

        print(f"CAVTaskRunner hostname: {socket.gethostname()}, PID: {os.getpid()}")
        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)

        register_cav_advantage()
        patch_ray_trainer_compute_advantage()

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
            from verl.workers.fsdp_workers import ActorRolloutRefWorker, AsyncActorRolloutRefWorker

            use_legacy_worker_impl = config.trainer.get("use_legacy_worker_impl", "auto")
            if use_legacy_worker_impl in ["auto", "enable"]:
                from verl.workers.fsdp_workers import CriticWorker
            else:
                from verl.workers.roles import CriticWorker

            actor_rollout_cls = (
                AsyncActorRolloutRefWorker
                if config.actor_rollout_ref.rollout.get("mode", "sync") == "async"
                else ActorRolloutRefWorker
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

        cav_cfg = config.get("cav", {})
        allowed_budgets = list(cav_cfg.get("budget_actions", [0, 16, 32, 64, 128]))
        reward_cfg = CAVRewardConfig(
            correct_reward=float(cav_cfg.get("correct_reward", 1.0)),
            wrong_reward=float(cav_cfg.get("wrong_reward", 0.0)),
            computation_price=float(cav_cfg.get("computation_price", 0.0005)),
            actual_token_price=float(cav_cfg.get("actual_token_price", 0.0001)),
            format_penalty=float(cav_cfg.get("format_penalty", 0.1)),
            missing_stop_penalty=float(cav_cfg.get("missing_stop_penalty", 0.2)),
            invalid_budget_penalty=float(cav_cfg.get("invalid_budget_penalty", 0.1)),
            target_expected_budget=float(cav_cfg.get("target_expected_budget", 128.0)),
            lambda_c=float(cav_cfg.get("lambda_c", 0.0005)),
        )
        reward_fn = CAVRewardManager(tokenizer=tokenizer, allowed_budgets=allowed_budgets, reward_config=reward_cfg)
        val_reward_fn = CAVValidationRewardManager(
            tokenizer=tokenizer,
            allowed_budgets=allowed_budgets,
            reward_config=reward_cfg,
        )

        resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)
        train_dataset = create_rl_dataset(config.data.train_files, config.data, tokenizer, processor, is_train=True)
        val_dataset = create_rl_dataset(config.data.val_files, config.data, tokenizer, processor, is_train=False)
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
        trainer.init_workers()
        trainer.fit()


if __name__ == "__main__":
    main()

