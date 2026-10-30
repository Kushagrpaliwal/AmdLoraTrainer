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

  for p in model.parameters():
    p.requires_grad_(False)
 
  found = []
  for name,module in model.named_modules():
    if isinstance(module,nn.Linear) and _matches(name,target_modules):
     found.append(name)

  replaced = []
  for name in found:
    parent_name , _ , child_name = name.rpartition(".")
    parent = model.get_submodule(parent_name) if parent_name else module ;
    if isinstance(parent,LoRALinear):
      continue
    child = getattr( parent, child_name )
    setattr(parent,child_name,LoRALinear(child,rank,alpha,dropout,lora_dtype))
    replaced.append(name)
  
  if not replaced:
    raise ValueError(f"No Linear Layers Matched target_modules={target_modules}")
  return replaced

def count_parameters(model:nn.Module) -> tuple[int,int]:
  trainable = sum(p.numel() for p in model.parameters() if p.requires_grad )
  total = sum(p.numel() for p in model.parameters())
  return trainable , total


    