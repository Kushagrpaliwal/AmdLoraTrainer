"""Step 3/4: default target_modules per model family."""
from __future__ import annotations

DEFAULT_TARGETS: dict[str, list[str]] = {
    "llm": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "diffusion": ["to_q", "to_k", "to_v", "to_out.0"],
}

def default_targets(family:str) -> list[str]:
  if family not in DEFAULT_TARGETS:
    raise ValueError(f"Unknown Model Family '{family}'. Known family '{list[DEFAULT_TARGETS]}' ")

  return list(DEFAULT_TARGETS[family])