"""Step 2: LoRALinear wraps a frozen nn.Linear. A=small random, B=zeros, scale=alpha/r. 
   Output = base weights + (alpha / rank) x B x A x X """

from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
  def __init__(
    self,
    base: nn.Linear , 
    rank: int = 16 , 
    alpha: float = 16.0 , 
    dropout:float = 0.0 , 
    lora_dtype:torch.dtype = torch.float32
    ):
    
    super().__init__()
    if rank <= 0:
      raise ValueError("Rank must be greater than 0")
    
    self.base = base
    for p in self.base.parameters():
      p.requires_grad_(False)
    
    self.rank = rank
    self.alpha = alpha 
    self.scale = alpha / rank
    self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
    device = base.weight.device
    self.A = nn.Parameter(torch.empty(rank,base.in_features , device = device , dtype = lora_dtype))
    self.B = nn.Parameter(torch.zeros(base.out_features , rank , device = device  ,dtype = lora_dtype))
    nn.init.kaiming_uniform_(self.A,a=math.sqrt(5))  


  def forward(self, x: torch.Tensor) -> torch.Tensor:
      out = self.base(x)
      h = self.dropout(x).to(self.A.dtype)
      lora = F.linear(F.linear(h,self.A),self.B) * self.scale
      return out + lora.to(out.dtype)

    
  @torch.no_grad()
  def delta_weight(self) -> torch.Tensor:
      return (self.B @ self.A) * self.scale


