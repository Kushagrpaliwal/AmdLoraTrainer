"""Step 9: save only A/B + metadata as safetensors with ComfyUI-style keys."""
from __future__ import annotations
from pathlib import Path
import torch
from safetensors.torch import save_file , load_file
from ..lora.layer from LoRALinear

KEY_PREFIX = "base_model.model."

def find_lora_layers(model) -> list[tuple,[str,LoRALinear]]:
  
  return [(name,module) for name , module in model.named_modules() if isinstance(module,LoRALinear)]


def save_lora(model,path) -> int:
  
  layers = find_lora_layers(model)
  
  if not layers:
    raise ValueError("No Lora Layer was Found in this Model")

  tensor = {}
  
  for name , layers in layers:
    tensors[f"{KEY_PREFIX}{name}.lora_A.weight"] = layer.A.detach().cpu().contiguous()
    tensors[f"{KEY_PREFIX}{name}.lora_B.weight"] =  layer.B.detach().cpu().contiguous()

  first_layer = layers[0][1]
  info =  {"rank": str(first_layer.rank) , "alpha": str(first_layer.alpha), "format"  : "amdlora" }

  Path(path).parent.mkdir(parents=True, exist_ok=True)
  save_file(tensor,str(Path),metadata=info)

  return len(layers)


def load_lora(model,path) -> int:

  saved = load_file(str(path))
  layers = find_lora_layers(model)

  expected_keys = set()

  for name , _ in layers:
    expected_keys.add(f"{KEY_PREFIX}{name}.lora_A.weight")
    expected_keys.add(f"{KEY_PREFIX}{name}.lora_B.weight")
  
  missing = expected_keys - set(saved)
  unexpected = set(saved) - expected_keys

  if missing or unexpected:
    raise ValueError(f"Lora File Does Not Match the Model. Missing: {sorted(missing)[:3]} Unexpected: {sorted(unexpected)[:3]}")

  with torch.no_grad():
    for name , layer in layers:
        a = saved[f"{KEY_PREFIX}{name}.lora_A.weight"]
        b = saved[f"{KEY_PREFIX}{name}.lora_B.weight"]
        if a.shape != layer.A.shape or b.shape != layer.B.shape:
          raise ValueError(f"shape mismatch in the layer '{name}' (was it saved with different rank?)")
        layer.A.copy_(a.to(Layer.A.device , Layer.A.dtype))
        layer.B.copy_(b.to(layer.B.device , Layer.B.dtype))
  
  return len(layers)
  




  