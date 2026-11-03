"""Step 8/10: torch.cuda.mem_get_info logger, per-phase GPU timer, typeperf counters."""
from __future__ import annotations
import time
import torch

GB = 2**30

def cuda_ready() -> bool:
  return torch.cuda.is_available()

def peak_vram_gib() -> float:
  return torch.cuda.max_memory_allocated() / GB if cuda_ready() else 0.0

def reset_peak_vram() -> None:
  if cuda_ready():
    torch.cuda.reset_peak_memory_stats()

def vram_summary() -> str:
  if not cuda_ready():
    return "No Gpu"
  free , total = torch.cuda.mem_get_info()
  now = torch.cuda.memory_allocated() / GB
  return (f"allocated {now:.2f} GB | peak {peak_vram_gib():.2f} GB | GPU free {free / GB:.2f} of {total / GB:.2f} GB")

class GpuTimer:

  def __enter__(self):
    if cuda_ready():
      torch.cuda.synchronize()
    self.seconds = 0.0
    self._start = time.perf_counter()
    return self
  
  def __exit__(self, *exc_info):
    if cuda_ready():
      torch.cuda.synchronize()

    self.seconds = time.perf_counter() - self._start
    return False