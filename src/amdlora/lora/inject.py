"""Step 3: walk modules, match target names, swap in LoRALinear, freeze everything else."""
from __future__ import annotations
import torch
import torch.nn as nn 
from .layer import LoRALinear

def _matches(name:str , patterns:list[str]) -> bool:
  return any(name == p or name.endswith("." + p) for p in patterns)

def inject_lora(
  model:nn.Module,
  target_modules:list[str],
  rank:int = 16,
  alpha:float = 16.0,
  dropout:float = 0.0,
  lora_dtype: torch.dtype = torch.float32
) -> list[str]:
  """Replace matching Linear modules with LoRA layers and freeze the model.

  Target names may be full module paths or suffixes such as ``q_proj``.
  Returns the full names of modules that were replaced.
  """
  if not target_modules:
    raise ValueError("target_modules must contain at least one module name")

  model.requires_grad_(False)
  replaced: list[str] = []

  # Snapshot the module paths before replacing children.
  for name, module in list(model.named_modules()):
    if not name or not isinstance(module, nn.Linear) or not _matches(name, target_modules):
      continue

    parent_name, _, child_name = name.rpartition(".")
    parent = model.get_submodule(parent_name) if parent_name else model
    setattr(parent, child_name, LoRALinear(
      module,
      rank=rank,
      alpha=alpha,
      dropout=dropout,
      lora_dtype=lora_dtype,
    ))
    replaced.append(name)

  if not replaced:
    raise ValueError(f"No Linear modules matched target_modules: {target_modules}")

  return replaced
