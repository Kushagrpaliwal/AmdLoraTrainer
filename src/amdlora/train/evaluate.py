"""Step 10: validation loss on held-out samples (no gradients, LoRA dropout off)."""
from __future__ import annotations
import torch


def count_loss_tokens(labels: torch.Tensor) -> int:
    """Number of tokens that count in the loss.

    Hugging Face shifts the labels by one (the token at position t predicts position t+1),
    so position 0 of every row is never a target. Padding (-100) never counts.
    """
    return int((labels[:, 1:] != -100).sum())


def evaluate(model, loader, device) -> float:
    """Average loss per target token over the whole loader.

    Uses the plain model, never the torch.compile wrapper: validation batches can have
    other shapes than training batches, which would make a compiled model recompile.
    """
    was_training = model.training
    model.eval()                                    # LoRA dropout off, checkpointing hooks idle
    total_loss, total_tokens = 0.0, 0
    try:
        with torch.no_grad():
            for batch in loader:
                batch = {name: tensor.to(device) for name, tensor in batch.items()}
                tokens = count_loss_tokens(batch["labels"])
                if tokens == 0:
                    continue
                loss = model(**batch).loss          # mean over this batch's target tokens
                total_loss += loss.item() * tokens  # back to a sum: every token weighs the same
                total_tokens += tokens
    finally:
        model.train(was_training)                   # restored even if something above fails
    if total_tokens == 0:
        raise ValueError("The validation set has no target tokens.")
    return total_loss / total_tokens
