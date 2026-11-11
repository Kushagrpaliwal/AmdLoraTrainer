from __future__ import annotations
import torch

def count_loss_tokens(labels: torch.Tensor):
  return int((labels[:,1:] != -100).sum())

def evaluate(model,loader,device) -> float:
  was_training = model.training
  model.eval()
  total_loss , total_tokens = 0.0 , 0
  try:
    with torch.no_grad():
      for batch in loader:
        batch = {name: tensor.to(device) for name , tensor in batch.items()}
        tokens = count_loss_tokens(batch["labels"])
        if tokens == 0:
          continue;
        loss = model(**batch).loss
        total_loss += loss.item() * tokens
        total_tokens += tokens
  
  finally:
    model.train(was_training)
    if total_tokens == 0:
      raise ValueError("The validation set has no target tokens.")
    return total_loss / total_tokens