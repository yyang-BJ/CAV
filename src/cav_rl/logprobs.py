from __future__ import annotations

import torch
import torch.nn.functional as F


def token_logprobs_from_logits(logits: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
    shifted_logits = logits[:, :-1, :]
    shifted_labels = input_ids[:, 1:]
    log_probs = F.log_softmax(shifted_logits, dim=-1)
    gathered = log_probs.gather(-1, shifted_labels.unsqueeze(-1)).squeeze(-1)
    pad = torch.zeros(gathered.shape[0], 1, device=gathered.device, dtype=gathered.dtype)
    return torch.cat([pad, gathered], dim=1)


def sum_masked_logprobs(logprobs: torch.Tensor, mask: list[bool]) -> torch.Tensor:
    if not mask:
        return logprobs.new_tensor(0.0)
    mask_tensor = torch.tensor(mask, dtype=torch.bool, device=logprobs.device)
    n = min(mask_tensor.numel(), logprobs.numel())
    if n == 0:
        return logprobs.new_tensor(0.0)
    return logprobs[:n][mask_tensor[:n]].sum()

