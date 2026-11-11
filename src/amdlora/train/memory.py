"""Step 8: gradient checkpointing, offload/block swap, per-step VRAM logging."""
from __future__ import annotations

def set_gradient_checkpointing(model,enabled: bool) -> None:

  if enabled and hasattr(model, "gradient_checkpointing_enable"):
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant":False})
  
  elif not enabled and hasattr(model,"gradient_checkpointing_disable"):
    model.gradient_checkpointing_disable()

def apply_memory_settings(model,cfg) -> None:

  if cfg.memory.cpu_offload:
    raise NotImplementedError("cpu_offload is not built yet")
  
  set_gradient_checkpointing(model,cfg.memory.gradient_checkpointing)
