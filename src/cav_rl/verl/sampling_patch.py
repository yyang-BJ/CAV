"""Allow per-call max_tokens overrides via DataProto.meta_info for hierarchical rollout."""

from __future__ import annotations

_CAV_ACTOR_CLS = None


def apply_vllm_sampling_patch() -> None:
    """Patch vLLM sync rollout to honor meta_info max_tokens / stop."""
    try:
        from verl.workers.rollout.vllm_rollout import vllm_rollout_spmd as mod
    except ImportError:
        return

    original = mod.vLLMRollout.generate_sequences
    if getattr(original, "_cav_sampling_patch", False):
        return

    def generate_sequences(self, prompts, **kwargs):
        meta = getattr(prompts, "meta_info", None) or {}
        if "max_tokens" in meta:
            kwargs = {**kwargs, "max_tokens": int(meta["max_tokens"])}
        if "stop" in meta and meta["stop"] is not None:
            kwargs = {**kwargs, "stop": list(meta["stop"])}
        return original(self, prompts, **kwargs)

    generate_sequences._cav_sampling_patch = True
    mod.vLLMRollout.generate_sequences = generate_sequences


def get_cav_actor_rollout_cls():
    """Return ActorRolloutRefWorker subclass that applies the sampling patch on workers."""
    global _CAV_ACTOR_CLS
    if _CAV_ACTOR_CLS is not None:
        return _CAV_ACTOR_CLS

    from verl.workers.fsdp_workers import ActorRolloutRefWorker

    apply_vllm_sampling_patch()

    class CAVActorRolloutRefWorker(ActorRolloutRefWorker):
        def __init__(self, *args, **kwargs):
            apply_vllm_sampling_patch()
            super().__init__(*args, **kwargs)

    _CAV_ACTOR_CLS = CAVActorRolloutRefWorker
    return _CAV_ACTOR_CLS
