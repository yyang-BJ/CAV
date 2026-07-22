from __future__ import annotations

import os
import socket

import hydra
import ray
from omegaconf import OmegaConf

from cav_rl.lambda_dual import DualLambdaConfig, LambdaController
from cav_rl.verl.advantage import patch_ray_trainer_compute_advantage, register_cav_advantage
from cav_rl.verl.metrics import patch_compute_data_metrics, patch_ray_trainer_validate
from cav_rl.verl.policy_loss import patch_verl_policy_loss
from cav_rl.verl.reward import CAVRewardConfig, CAVRewardManager, CAVValidationRewardManager
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
        # Older veRL builds inline this dict in main_ppo instead of constants_ppo.
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
    runner = CAVTaskRunner.remote()
    ray.get(runner.run.remote(config))

    if hasattr(config, "ray_init") and config.ray_init is not None:
        timeline_json_file = config.ray_init.get("timeline_json_file", None)
        if timeline_json_file:
            ray.timeline(filename=timeline_json_file)


@ray.remote(num_cpus=1)
class CAVTaskRunner:
    def run(self, config):
        from pprint import pprint

        import torch

        from verl.trainer.ppo.ray_trainer import RayPPOTrainer, ResourcePoolManager, Role
        from verl.trainer.ppo.utils import create_rl_dataset, create_rl_sampler
        from verl.utils import hf_tokenizer
        from verl.utils.dataset.rl_dataset import collate_fn
        from verl.utils.fs import copy_to_local

        print(f"CAVTaskRunner hostname: {socket.gethostname()}, PID: {os.getpid()}")
        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)

        register_cav_advantage()
        patch_ray_trainer_compute_advantage()
        patch_verl_policy_loss()
        patch_compute_data_metrics()
        # Must run before validate metrics wrap: replaces T3 multi-turn fit/_validate.
        patch_ray_trainer_single_turn()
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
            from verl.workers.engine_workers import ActorRolloutRefWorker

            actor_rollout_cls = get_cav_actor_rollout_cls()

            ray_worker_group_cls = RayWorkerGroup
        elif config.actor_rollout_ref.actor.strategy == "megatron":
            assert config.actor_rollout_ref.actor.strategy == config.critic.strategy

            from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
            from verl.workers.engine_workers import ActorRolloutRefWorker

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
        initial_lambda = float(cav_cfg.get("lambda_c", 0.0005))
        dual_update = str(cav_cfg.get("dual_update", True)).lower() in {"1", "true", "yes", "y"}
        b_final = float(cav_cfg.get("target_expected_tokens", 128.0))
        b_start_raw = cav_cfg.get("b_start", None)
        if b_start_raw is None or str(b_start_raw).strip().lower() in {"", "none", "null"}:
            b_start = None
        else:
            b_start = float(b_start_raw)
        total_training_steps = int(config.trainer.get("total_training_steps", 100) or 100)
        dual_cfg = DualLambdaConfig(
            initial_lambda_c=initial_lambda,
            target_expected_tokens=b_final,
            dual_lr=float(cav_cfg.get("dual_lr", 0.01)),
            min_lambda_c=float(cav_cfg.get("min_lambda_c", 0.0)),
            max_lambda_c=float(cav_cfg.get("max_lambda_c", 0.02)),
            enabled=dual_update,
            b_start=b_start,
            b_anneal_ratio=float(cav_cfg.get("b_anneal_ratio", 0.7)),
            lambda_scale_start_ratio=float(cav_cfg.get("lambda_scale_start_ratio", 0.1)),
            lambda_scale_end_ratio=float(cav_cfg.get("lambda_scale_end_ratio", 0.4)),
            total_training_steps=total_training_steps,
        )
        lambda_controller = LambdaController(dual_cfg)
        # Reward uses λ_eff from step 0 (cosine scale may be 0 during early warmup).
        initial_lambda_eff = lambda_controller.effective_lambda_at(0)
        initial_b = lambda_controller.budget_at(0)
        import dataclasses as _dc

        reward_cfg_kwargs = {
            "correct_reward": float(cav_cfg.get("correct_reward", 1.0)),
            "wrong_reward": float(cav_cfg.get("wrong_reward", 0.0)),
            "format_penalty": float(cav_cfg.get("format_penalty", 0.5)),
            "invalid_action_penalty": (
                float(cav_cfg["invalid_action_penalty"])
                if cav_cfg.get("invalid_action_penalty") is not None
                else 0.5
            ),
            "missing_stop_penalty": float(cav_cfg.get("missing_stop_penalty", 0.2)),
            "invalid_budget_penalty": float(cav_cfg.get("invalid_budget_penalty", 0.1)),
            "correctness_requires_valid_format": str(
                cav_cfg.get("correctness_requires_valid_format", True)
            ).lower()
            in {"1", "true", "yes", "y"},
            "target_expected_tokens": initial_b,
            "lambda_c": initial_lambda_eff,
            "dual_lr": dual_cfg.dual_lr,
            "min_lambda_c": dual_cfg.min_lambda_c,
            "max_lambda_c": dual_cfg.max_lambda_c,
            "dual_update": dual_update,
            "debug_print_responses": str(cav_cfg.get("debug_print_responses", False)).lower()
            in {"1", "true", "yes", "y"},
            # reward1 extras (ignored if CAVRewardConfig lacks these fields)
            "overflow_budget_penalty": float(cav_cfg.get("overflow_budget_penalty", 0.05)),
            "correct_overflow_margin": float(cav_cfg.get("correct_overflow_margin", 0.05)),
        }
        valid_fields = {f.name for f in _dc.fields(CAVRewardConfig)}
        reward_cfg = CAVRewardConfig(**{k: v for k, v in reward_cfg_kwargs.items() if k in valid_fields})
        reward_fn = CAVRewardManager(tokenizer=tokenizer, allowed_budgets=allowed_budgets, reward_config=reward_cfg)
        val_reward_fn = CAVValidationRewardManager(
            tokenizer=tokenizer,
            allowed_budgets=allowed_budgets,
            reward_config=reward_cfg,
        )
        print(
            "[CAV] dual schedule: "
            f"B_start={dual_cfg.b_start} B_final={dual_cfg.target_expected_tokens} "
            f"b_anneal_ratio={dual_cfg.b_anneal_ratio} "
            f"lambda_scale=[{dual_cfg.lambda_scale_start_ratio},{dual_cfg.lambda_scale_end_ratio}] "
            f"T={total_training_steps} lambda_eff(0)={initial_lambda_eff:.6g}",
            flush=True,
        )

        resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)
        try:
            train_dataset = create_rl_dataset(
                config.data.train_files, config.data, tokenizer, processor, is_train=True
            )
            val_dataset = create_rl_dataset(
                config.data.val_files, config.data, tokenizer, processor, is_train=False
            )
        except TypeError:
            # Older veRL create_rl_dataset has no is_train kwarg.
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
        trainer.lambda_controller = lambda_controller
        trainer.init_workers()
        trainer.fit()


if __name__ == "__main__":
    main()
