"""Helpers for format-strengthened CAV SFT training."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from transformers import Trainer


class FormatWeightedSFTTrainer(Trainer):
    """Causal LM trainer with optional per-token loss weights (for tag tokens)."""

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        loss_weights = inputs.pop("loss_weights", None)
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        # Standard shift for causal LM
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        vocab_size = shift_logits.size(-1)
        flat_logits = shift_logits.view(-1, vocab_size)
        flat_labels = shift_labels.view(-1)
        token_loss = F.cross_entropy(flat_logits, flat_labels, reduction="none", ignore_index=-100)

        if loss_weights is None:
            loss = token_loss.sum() / (flat_labels != -100).sum().clamp_min(1)
        else:
            shift_weights = loss_weights[..., 1:].contiguous().view(-1)
            # Ignored labels already have weight 0 from the collator, but clamp for safety.
            valid = (flat_labels != -100).float()
            weighted = token_loss * shift_weights * valid
            denom = (shift_weights * valid).sum().clamp_min(1.0)
            loss = weighted.sum() / denom

        return (loss, outputs) if return_outputs else loss
