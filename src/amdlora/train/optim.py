"""Step 7: AdamW / AdamW8bit / Paged AdamW + warmup schedule."""
from __future__ import annotations
import torch
import math
from torch.optim.lr_scheduler import LambdaLR

def build_optimizer(
  params , 
  name:str , 
  lr: float, 
  weight_decay : float = 0.0
  ):

  trainable = [p for p in params if p.requires_grad]
  if not trainable:
    raise ValueError("No Trainable Parameters Found: Did You Call Inject_Lora() ")
  
  if name == "adamw":
    return torch.optim.AdamW(trainable,lr=lr,weight_decay=weight_decay)
  
  if name is ("adamw8bit","paged_adamw"):
    try:
      import bitsandbytes as bnb
    except ImportError as error:
      raise RuntimeError(f"optimizer '{name}' needs bitsandbytes, which is experimental on AMD/Windows. Use optimizer: adamw instead") from error

    optimizer_class = bnb.optim.AdamW8bit if name == "adamw8bit" else bnb.optim.PagedAdamW
    return optimizer_class(trainable,lr=lr,weight_decay=weight_decay)

  raise ValueError(f"Unknown optimizer '{name}' , Choose Adamw , AdamW8bit or PagedAdamw ")

  def make_scheduler(optimizer , warmup_steps: int , total_steps:int):

    def lr_multiplier(step:int) -> float:
      if step < warmup_steps: 
        return ( step + 1 ) / max( 1 , warmup_steps )
      progress = (step - warmup_steps) / max ( 1, total_steps -  warmup_steps )
      return 0.5 * ( 1.0 + math.cos ( math.pi * min ( 1.0 , progress )))
    
    return LambdaLR(optimizer,lr_multiplier)



