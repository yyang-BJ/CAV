#!/usr/bin/env python3
"""Unit tests for GRPO-correct reward (no GPU)."""

from __future__ import annotations

from cav_rl.verl.grpo_correct_reward import (
    GrpoCorrectRewardConfig,
    compute_rank_based_length_scores,
    reward_group,
)


def test_rank_scores_shortest_highest():
    scores = compute_rank_based_length_scores([100.0, 50.0, 75.0])
    assert scores[1] == 1.0
    assert scores[0] == 0.0
    assert abs(scores[2] - 0.5) < 1e-9


def test_reward_branches_format_and_length():
    cfg = GrpoCorrectRewardConfig()
    completions = [
        "step\n#### 42",  # correct + strict, short eligible
        "long " * 80 + "\n#### 42",  # correct + strict, long eligible
        "The answer is 42",  # correct, no ####
        "step\n#### 7",  # wrong + strict
        "The answer is 7",  # wrong, no ####
        "no numeric answer",  # unparsable
    ]
    lens = [280.0, 400.0, 100.0, 120.0, 90.0, 20.0]
    rewards, infos = reward_group(completions, "42", lens, cfg)

    assert infos[0]["correct"] and infos[0]["has_strict_format"]
    assert infos[1]["correct"] and infos[1]["has_strict_format"]
    assert infos[2]["correct"] and not infos[2]["has_strict_format"]
    assert infos[3]["parsable"] and not infos[3]["correct"] and infos[3]["has_strict_format"]
    assert infos[4]["parsable"] and not infos[4]["correct"] and not infos[4]["has_strict_format"]
    assert not infos[5]["parsable"]

    # R = 1.2 + 0.01 * length_score for correct+strict
    assert abs(rewards[0] - (1.0 + 0.2 + 0.01 * 1.0)) < 1e-9
    assert abs(rewards[1] - (1.0 + 0.2 + 0.01 * 0.0)) < 1e-9
    assert abs(rewards[2] - (1.0 - 0.2)) < 1e-9
    assert abs(rewards[3] - (-0.5 + 0.2)) < 1e-9
    assert abs(rewards[4] - (-0.5 - 0.2)) < 1e-9
    assert abs(rewards[5] - (-1.0)) < 1e-9

    # Ordering: correct+fmt+short > correct+fmt+long > correct-no-fmt
    #           > wrong+fmt > wrong-no-fmt > unparsable
    assert rewards[0] > rewards[1] > rewards[2] > rewards[3] > rewards[4] > rewards[5]


def test_rank_with_floor_excludes_short():
    cfg = GrpoCorrectRewardConfig(length_floor=270.0)
    completions = [
        "a\n#### 42",
        "b\n#### 42",
        "c\n#### 42",
        "d\n#### 7",
    ]
    lens = [280.0, 400.0, 50.0, 300.0]
    rewards, infos = reward_group(completions, "42", lens, cfg)

    assert infos[0]["length_score"] == 1.0
    assert infos[1]["length_score"] == 0.0
    assert infos[2]["length_score"] == 0.0
    assert rewards[0] > rewards[1]
    assert abs(rewards[1] - rewards[2]) < 1e-9  # both length_score=0, both strict correct
    assert rewards[2] > rewards[3]


def test_correct_no_format_gets_no_length():
    cfg = GrpoCorrectRewardConfig()
    completions = ["answer 42", "#### 42"]
    lens = [280.0, 400.0]
    rewards, infos = reward_group(completions, "42", lens, cfg)
    assert infos[0]["length_score"] == 0.0  # no strict → not ranked
    assert infos[1]["length_score"] == 1.0  # singleton eligible
    assert abs(rewards[0] - 0.8) < 1e-9
    assert abs(rewards[1] - (1.2 + 0.01)) < 1e-9


if __name__ == "__main__":
    test_rank_scores_shortest_highest()
    test_reward_branches_format_and_length()
    test_rank_with_floor_excludes_short()
    test_correct_no_format_gets_no_length()
    print("ok: grpo_correct reward tests passed")
