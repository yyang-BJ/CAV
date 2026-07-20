# Method Alignment: Revised Proposal ↔ Code

This note maps the revised CAV proposal to the current implementation.

## Dual time axes / hierarchical rollout

| Proposal | Code |
|----------|------|
| Macro state \(H_k\), sample \(b_k\sim\pi(\cdot\|H_k)\) | `cav_rl/verl/hierarchical_rollout.py` budget phase |
| Then \(z_k\) or answer given \(b_k\) | reason / answer phase in the same module |
| \(H_{k+1}=h_{t_k+l_k}\) | next step prompt = original prompt + completion so far |
| VeRL one-shot fallback | `cav.hierarchical_rollout=false` |

Local debug reference: `cav_rl/rollout.py::generate_macro_completion`.

## Cost and rewards

| Proposal | Code |
|----------|------|
| \(l_k\) = tokens inside `<reason>` | `masks.actual_reason_tokens` / `decision.reason_token_count` |
| \(C=\sum l_k\) | `cav_actual_reason_tokens` |
| Main reward \(R'=R_{answer}/(1+\lambda_c C)\) | `gated_answer_reward` on stop/answer macro |
| Format / stop / invalid-budget penalties | additive extras on the terminal anchor |
| SFT targets must satisfy \(l_k\le b_k\) | `sft_fit_budget` + `build_validated_sft_completion` |

## TD / GAE

| Proposal | Code |
|----------|------|
| Macro GAE with duration \(l_k\); terminal \(r\) carries gated \(R'\) | `cav_rl/verl/advantage.py` (`duration=l_k`) |
| Variable-length GAE, shared \(A_k\) on budget+reason tokens | macro broadcast in `compute_cav_gae_advantage_return` |

## Dual \(\lambda_c\)

| Proposal | Code |
|----------|------|
| \(\lambda\leftarrow[\lambda+\eta(E[C]-B)]_+\) | `cav_rl/lambda_dual.py` + `single_turn._update_dual_lambda` |
| \(B=\) target expected tokens | `cav.target_expected_tokens` |
| Metrics | `cav/lambda_c`, `cav/dual_gap` |

Disable with `cav.dual_update=false`.

## Deferred

- Dual value heads \(V^{high}/V^{low}\): not implemented yet (marked 优化).
