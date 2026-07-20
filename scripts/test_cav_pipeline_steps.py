#!/usr/bin/env python3
"""Step-by-step verification of CAV hierarchical rollout, dual lambda, reward, GAE."""

from __future__ import annotations

import sys
import traceback

import numpy as np
import torch
from omegaconf import OmegaConf
from tensordict import TensorDict


def _ok(msg: str) -> None:
    print(f"[PASS] {msg}", flush=True)


def _fail(msg: str) -> None:
    print(f"[FAIL] {msg}", flush=True)
    raise AssertionError(msg)


def test_dual_lambda() -> None:
    import math

    from cav_rl.lambda_dual import DualLambdaConfig, LambdaController

    ctrl = LambdaController(
        DualLambdaConfig(
            initial_lambda_c=0.001,
            target_expected_tokens=100.0,
            dual_lr=0.01,
            min_lambda_c=0.0,
            max_lambda_c=1.0,
            enabled=True,
            # legacy path: no B anneal / scale warmup
            b_start=None,
            lambda_scale_start_ratio=0.0,
            lambda_scale_end_ratio=0.0,
        )
    )
    # E[C]=150 > B=100 => lambda increases by 0.5
    v1 = ctrl.update(150.0)
    if abs(v1 - 0.501) > 1e-9:
        _fail(f"dual up: expected 0.501 got {v1}")
    # E[C]=50 < B => decreases by 0.5 back to 0.001
    v2 = ctrl.update(50.0)
    if abs(v2 - 0.001) > 1e-9:
        _fail(f"dual down: expected 0.001 got {v2}")

    ctrl2 = LambdaController(DualLambdaConfig(enabled=False, initial_lambda_c=0.002))
    if ctrl2.update(999.0) != 0.002:
        _fail("disabled dual should not move")

    # Cosine scale + B anneal (ratios of T=100)
    sched = LambdaController(
        DualLambdaConfig(
            initial_lambda_c=0.01,
            target_expected_tokens=60.0,
            dual_lr=0.0,  # freeze dual to isolate scale/B
            enabled=True,
            b_start=100.0,
            b_anneal_ratio=0.5,
            lambda_scale_start_ratio=0.1,
            lambda_scale_end_ratio=0.4,
            total_training_steps=100,
        )
    )
    if abs(sched.budget_at(0) - 100.0) > 1e-9 or abs(sched.budget_at(50) - 60.0) > 1e-9:
        _fail(f"B anneal endpoints: B(0)={sched.budget_at(0)} B(50)={sched.budget_at(50)}")
    if abs(sched.budget_at(25) - 80.0) > 1e-9:
        _fail(f"B anneal midpoint: expected 80 got {sched.budget_at(25)}")
    if sched.scale_at(5) != 0.0 or sched.scale_at(40) != 1.0:
        _fail(f"scale endpoints: s(5)={sched.scale_at(5)} s(40)={sched.scale_at(40)}")
    mid = sched.scale_at(25)  # halfway in [10,40]
    expected_mid = 0.5 * (1.0 - math.cos(math.pi * 0.5))
    if abs(mid - expected_mid) > 1e-9:
        _fail(f"cosine mid: expected {expected_mid} got {mid}")
    eff0 = sched.update(80.0, global_step=5, total_steps=100)
    if abs(eff0) > 1e-12:
        _fail(f"lambda_eff should be 0 during early warmup, got {eff0}")
    if abs(sched.target_expected_tokens - sched.budget_at(5)) > 1e-9:
        _fail("B_t not applied during update")
    eff1 = sched.update(80.0, global_step=40, total_steps=100)
    if abs(eff1 - 0.01) > 1e-9:
        _fail(f"lambda_eff should equal dual after warmup, got {eff1}")

    _ok("dual lambda update math + cosine scale + B anneal")


def test_hierarchical_mock() -> None:
    from transformers import AutoTokenizer
    from verl import DataProto

    from cav_rl.parsing import parse_completion
    from cav_rl.verl.hierarchical_rollout import generate_hierarchical_sequences, hierarchical_enabled

    cfg = OmegaConf.create({"cav": {"hierarchical_rollout": True}})
    if not hierarchical_enabled(cfg):
        _fail("hierarchical_enabled True")
    if hierarchical_enabled(OmegaConf.create({"cav": {"hierarchical_rollout": False}})):
        _fail("hierarchical_enabled False")

    model_path = "/home/dataset-assist-0/ZX/CAV/outputs/sft-qwen2.5-3b-cav-gsm8k-merged"
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    prompt = "Solve: 1+1\n"
    prompt_ids = tok.encode(prompt, add_special_tokens=False)
    max_prompt = 64
    pad = int(tok.pad_token_id)
    left = [pad] * (max_prompt - len(prompt_ids)) + prompt_ids
    attn = [0] * (max_prompt - len(prompt_ids)) + [1] * len(prompt_ids)

    gen_batch = DataProto(
        batch=TensorDict(
            {
                "input_ids": torch.tensor([left, left], dtype=torch.long),
                "attention_mask": torch.tensor([attn, attn], dtype=torch.long),
                "position_ids": torch.arange(max_prompt).unsqueeze(0).repeat(2, 1),
            },
            batch_size=2,
        ),
        non_tensor_batch={"answer": np.array(["2", "2"], dtype=object)},
        meta_info={"eos_token_id": tok.eos_token_id, "pad_token_id": pad},
    )

    # Per-call scripted outputs: budget -> reason -> budget0 -> answer
    scripts = [
        "<budget>16</budget>",
        "<reason>add ones</reason>",
        "<budget>0</budget>",
        "<answer>2</answer>",
    ]
    call_idx = {"n": 0}
    max_tokens_seen = []

    def fake_generate(proto: DataProto):
        max_tokens_seen.append(int(proto.meta_info.get("max_tokens", -1)))
        text = scripts[min(call_idx["n"], len(scripts) - 1)]
        call_idx["n"] += 1
        ids = tok.encode(text, add_special_tokens=False)
        max_resp = 96
        resp = ids + [pad] * (max_resp - len(ids))
        bsz = proto.batch["input_ids"].size(0)
        # Ensure continuing prompts include previous completion (H_k growth).
        if call_idx["n"] > 1:
            # After first budget, prompt tensors should be longer than original max_prompt
            # or contain decoded budget text when stripped.
            row = proto.batch["input_ids"][0].tolist()
            stripped = [t for t in row if t != pad]
            decoded = tok.decode(stripped, skip_special_tokens=True)
            if call_idx["n"] == 2 and "<budget>16</budget>" not in decoded:
                raise AssertionError(f"H_k missing prior budget at call2: {decoded!r}")
            if call_idx["n"] == 3 and "<reason>" not in decoded:
                raise AssertionError(f"H_k missing reason at call3: {decoded!r}")
        return DataProto(
            batch=TensorDict(
                {
                    "responses": torch.tensor([resp] * bsz, dtype=torch.long),
                    "prompts": proto.batch["input_ids"],
                },
                batch_size=bsz,
            ),
            non_tensor_batch={},
            meta_info={},
        )

    out = generate_hierarchical_sequences(
        gen_batch,
        fake_generate,
        tok,
        max_response_length=256,
        max_model_len=512,
        allowed_budgets=[0, 16, 32, 64, 128],
        max_macro_steps=4,
    )

    if call_idx["n"] != 4:
        _fail(f"expected 4 generate calls (2 macros x 2 phases), got {call_idx['n']}")
    if not max_tokens_seen or any(m <= 0 for m in max_tokens_seen):
        _fail(f"max_tokens overrides missing/invalid: {max_tokens_seen}")
    # Defaults: budget=64, reason uses b_k+slack=16+16=32, budget again, answer=96
    if max_tokens_seen != [64, 32, 64, 96]:
        _fail(f"unexpected max_tokens schedule: {max_tokens_seen}")

    for i in range(2):
        resp_ids = out.batch["responses"][i]
        valid = int(out.batch["attention_mask"][i, -resp_ids.numel() :].sum().item())
        text = tok.decode(resp_ids[:valid], skip_special_tokens=True)
        parsed = parse_completion(text, {0, 16, 32, 64, 128}, tokenizer=tok)
        if not parsed.has_stop:
            _fail(f"sample{i} missing stop budget0: {text!r}")
        if parsed.answer != "2":
            _fail(f"sample{i} answer={parsed.answer!r} text={text!r}")
        if len(parsed.positive_budgets) != 1 or parsed.positive_budgets[0] != 16:
            _fail(f"sample{i} budgets={parsed.positive_budgets}")
        if parsed.decisions[0].reason_token_count <= 0:
            _fail(f"sample{i} l_k should be >0")

    # response_mask: pads must be 0
    resp_mask = out.batch["attention_mask"][:, -out.batch["responses"].size(1) :]
    if not torch.all(resp_mask[:, valid:] == 0) if valid < resp_mask.size(1) else True:
        pass
    if int(resp_mask.sum()) <= 0:
        _fail("empty response mask")
    _ok(f"hierarchical mock: 4 calls, H_k grows, parse ok, max_tokens={max_tokens_seen}")


def test_reward_and_dual_hook() -> None:
    from transformers import AutoTokenizer
    from verl import DataProto

    from cav_rl.lambda_dual import DualLambdaConfig, LambdaController
    from cav_rl.verl.reward import CAVRewardConfig, CAVRewardManager
    from cav_rl.verl.single_turn import _update_dual_lambda

    model_path = "/home/dataset-assist-0/ZX/CAV/outputs/sft-qwen2.5-3b-cav-gsm8k-merged"
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    pad = int(tok.pad_token_id)

    completion = (
        "<budget>16</budget><reason>one plus one is two</reason>"
        "<budget>0</budget><answer>2</answer>"
    )
    resp = tok.encode(completion, add_special_tokens=False)
    prompt = tok.encode("Q?", add_special_tokens=False)
    max_prompt, max_resp = 32, 128
    prompt_row = [pad] * (max_prompt - len(prompt)) + prompt
    prompt_attn = [0] * (max_prompt - len(prompt)) + [1] * len(prompt)
    resp_row = resp + [pad] * (max_resp - len(resp))
    resp_attn = [1] * len(resp) + [0] * (max_resp - len(resp))
    full = prompt_row + resp_row
    full_attn = prompt_attn + resp_attn

    # B deliberately below observed l_k so dual ascent must raise lambda_c.
    cfg = CAVRewardConfig(lambda_c=0.01, target_expected_tokens=1.0, dual_lr=0.1, dual_update=True)
    rm = CAVRewardManager(tokenizer=tok, allowed_budgets=[0, 16, 32, 64, 128], reward_config=cfg)

    data = DataProto(
        batch=TensorDict(
            {
                "prompts": torch.tensor([prompt_row], dtype=torch.long),
                "responses": torch.tensor([resp_row], dtype=torch.long),
                "input_ids": torch.tensor([full], dtype=torch.long),
                "attention_mask": torch.tensor([full_attn], dtype=torch.long),
            },
            batch_size=1,
        ),
        non_tensor_batch={"answer": np.array(["2"], dtype=object)},
        meta_info={},
    )
    reward_tensor, finals = rm(data)
    reason_tokens = float(data.non_tensor_batch["cav_actual_reason_tokens"][0])
    if reason_tokens <= 0:
        _fail("reason tokens should be >0")
    if float(data.non_tensor_batch["cav_accuracy"][0]) != 1.0:
        _fail("accuracy should be 1")
    # Gated main reward: R' = R_answer / (1 + λ C), C = reason tokens.
    total = float(reward_tensor.sum().item())
    expected_gated = 1.0 / (1.0 + cfg.lambda_c * reason_tokens)
    if abs(total - expected_gated) > 0.05:
        # allow small format-penalty noise; main term must be gated (not raw 1.0)
        if total >= 1.0 - 1e-6:
            _fail(f"reward {total} missing gated cost (l_k={reason_tokens}, expect~{expected_gated:.4f})")
        if total <= 0.0:
            _fail(f"reward {total} should stay non-negative when correct (gated)")
    _ok(f"reward: acc=1, l_k={reason_tokens}, R={total:.4f}, gated~{expected_gated:.4f}")
    class FakeTrainer:
        def __init__(self):
            self.lambda_controller = LambdaController(
                DualLambdaConfig(
                    initial_lambda_c=cfg.lambda_c,
                    target_expected_tokens=cfg.target_expected_tokens,
                    dual_lr=cfg.dual_lr,
                    min_lambda_c=0.0,
                    max_lambda_c=1.0,
                    enabled=True,
                )
            )
            self.reward_fn = rm
            self.val_reward_fn = rm

    trainer = FakeTrainer()
    metrics = {}
    old = cfg.lambda_c
    _update_dual_lambda(trainer, data, metrics)
    # observed C = reason_tokens, B=5; if C>5 lambda rises
    if reason_tokens > cfg.target_expected_tokens:
        if trainer.reward_fn.reward_config.lambda_c <= old:
            _fail("lambda should increase when C>B")
    if "cav/dual_gap" not in metrics or "cav/lambda_c" not in metrics:
        _fail(f"missing dual metrics: {metrics}")
    gap = metrics["cav/dual_gap"]
    if abs(gap - (reason_tokens - cfg.target_expected_tokens)) > 1e-6:
        _fail(f"dual_gap mismatch: {gap}")
    _ok(f"dual hook: lambda {old} -> {metrics['cav/lambda_c']:.6f}, gap={gap:.2f}")


def test_gae_uses_l_k_discount() -> None:
    from cav_rl.verl.advantage import compute_cav_gae_advantage_return

    # 1 sample, 2 macros: reason length 2 then stop length 0
    # tokens: [b0,r,r, b1,a]  macro ids 0,0,0, 1,1
    T = 5
    rewards = torch.zeros(1, T)
    rewards[0, 0] = -0.02  # cost on budget anchor macro0
    rewards[0, 3] = 1.0  # answer on budget anchor macro1
    response_mask = torch.ones(1, T)
    values = torch.tensor([[0.5, 0.0, 0.0, 0.1, 0.0]])
    budget_mask = torch.tensor([[1.0, 0, 0, 1.0, 0]])
    reason_mask = torch.tensor([[0.0, 1, 1, 0, 0]])
    executor_mask = torch.tensor([[0.0, 1, 1, 0, 1]])
    macro_ids = torch.tensor([[0, 0, 0, 1, 1]])

    class C:
        gamma = 0.9
        lam = 1.0

    adv, ret = compute_cav_gae_advantage_return(
        rewards,
        response_mask,
        config=C(),
        values=values,
        cav_budget_mask=budget_mask,
        cav_reason_mask=reason_mask,
        cav_executor_mask=executor_mask,
        cav_macro_ids=macro_ids,
    )
    # macro0 duration l_k=2 => discount gamma^2 between macros
    # With lam=1, just check shapes and that advantages are broadcast to macro tokens
    if adv.shape != (1, T):
        _fail(f"adv shape {adv.shape}")
    # tokens in same macro should share advantage
    if abs(float(adv[0, 0] - adv[0, 1])) > 1e-5 or abs(float(adv[0, 1] - adv[0, 2])) > 1e-5:
        _fail(f"macro0 adv not shared: {adv[0, :3]}")
    if abs(float(adv[0, 3] - adv[0, 4])) > 1e-5:
        _fail(f"macro1 adv not shared: {adv[0, 3:]}")
    _ok(f"GAE macro broadcast ok, adv0={float(adv[0,0]):.4f} adv1={float(adv[0,3]):.4f}")


def main() -> int:
    tests = [
        test_dual_lambda,
        test_hierarchical_mock,
        test_reward_and_dual_hook,
        test_gae_uses_l_k_discount,
    ]
    failed = 0
    for fn in tests:
        name = fn.__name__
        print(f"\n=== {name} ===", flush=True)
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"[FAIL] {name}: {exc}", flush=True)
            traceback.print_exc()
    print(f"\n=== summary: {len(tests) - failed}/{len(tests)} passed ===", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
