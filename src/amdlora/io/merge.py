"""Step 11: merge a trained LoRA into a copy of the base model's weights.

    merged_weight = original_weight + (alpha / rank) * (B @ A)

Rules (from the plan):
  - Merge into the ORIGINAL full-precision (BF16/FP16/FP32) weights, never into a quantized
    model. A LoRA trained on a 4-bit base is still merged into the original BF16 files.
  - Work one shard at a time, so RAM use stays near the size of one shard.
  - The original model folder and the LoRA file are never modified.
"""
from __future__ import annotations
import json
import shutil
from pathlib import Path
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from .save import KEY_PREFIX

FLOAT_TYPES = (torch.float32, torch.bfloat16, torch.float16)
WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt", ".gguf")      # not copied as "other files"


def read_lora(path) -> tuple[dict[str, dict[str, torch.Tensor]], int, float]:
    """Returns ({module_name: {"A": tensor, "B": tensor}}, rank, alpha)."""
    with safe_open(str(path), framework="pt") as f:
        info = f.metadata() or {}
        tensors = {key: f.get_tensor(key) for key in f.keys()}
    if "rank" not in info or "alpha" not in info:
        raise ValueError("LoRA file has no rank/alpha metadata (was it saved by amdlora?)")

    pairs: dict[str, dict[str, torch.Tensor]] = {}
   
    for key, tensor in tensors.items():
        if not key.startswith(KEY_PREFIX):
            raise ValueError(f"Unexpected key in LoRA file: {key}")
        body = key[len(KEY_PREFIX):]
        for suffix, part in ((".lora_A.weight", "A"), (".lora_B.weight", "B")):
            if body.endswith(suffix):
                pairs.setdefault(body[: -len(suffix)], {})[part] = tensor
   
    for module, pair in pairs.items():
        if set(pair) != {"A", "B"}:
            raise ValueError(f"LoRA file is missing an A or B matrix for '{module}'")
   
    return pairs, int(info["rank"]), float(info["alpha"])


def find_shards(base_dir: Path) -> list[Path]:
    """The weight files of a model folder (one file, or several listed in an index)."""
    index = base_dir / "model.safetensors.index.json"
    if index.exists():
        names = sorted(set(json.loads(index.read_text(encoding="utf-8"))["weight_map"].values()))
        return [base_dir / name for name in names]
    
    single = base_dir / "model.safetensors"
    
    if single.exists():
        return [single]
    
    raise FileNotFoundError(f"No .safetensors weights found in {base_dir} "
                            "(only safetensors models are supported)")


def merge_one(weight: torch.Tensor, A: torch.Tensor, B: torch.Tensor, scale: float, name: str):
    """Add the LoRA change to one weight tensor, keeping the tensor's own number format."""
    if weight.dtype not in FLOAT_TYPES:
        raise ValueError(f"'{name}' is stored as {weight.dtype}. Never merge into a quantized "
                         "model: point base_model at the original BF16/FP16 weights.")

    delta = scale * (B.float() @ A.float())              # computed in FP32 for accuracy
    if delta.shape != weight.shape:
        raise ValueError(f"Shape mismatch for '{name}': weight {tuple(weight.shape)}, "
                         f"LoRA change {tuple(delta.shape)}")

    return (weight.float() + delta).to(weight.dtype)


def merge_lora_into_model_dir(base_dir, lora_path, out_dir) -> int:
    """Write a merged copy of the model to out_dir. Returns how many layers were merged."""
    base_dir, out_dir = Path(base_dir), Path(out_dir)
    pairs, rank, alpha = read_lora(lora_path)
    scale = alpha / rank
    wanted = {f"{module}.weight": pair for module, pair in pairs.items()}

    # Check first, without loading any weights, that every LoRA layer exists in the model.
    shards = find_shards(base_dir)
    available: set[str] = set()
   
    for shard in shards:
        with safe_open(str(shard), framework="pt") as f:
            available.update(f.keys())
   
    missing = sorted(set(wanted) - available)
    if missing:
        raise ValueError(f"{len(missing)} LoRA layers were not found in the base model, "
                         f"for example: {missing[:3]}. Is this the right base model?")

    out_dir.mkdir(parents=True, exist_ok=True)
    merged = 0
    for shard in shards:                                 # one shard in memory at a time
        tensors = {}
        with safe_open(str(shard), framework="pt") as f:
            for key in f.keys():
                tensor = f.get_tensor(key)
                if key in wanted:
                    pair = wanted[key]
                    tensor = merge_one(tensor, pair["A"], pair["B"], scale, key)
                    merged += 1
                tensors[key] = tensor
        save_file(tensors, str(out_dir / shard.name), metadata={"format": "pt"})
        del tensors

    # Copy everything else a model folder needs (config, tokenizer files, index, ...).
    for item in base_dir.iterdir():
        if item.is_file() and item.suffix not in WEIGHT_SUFFIXES:
            shutil.copy2(item, out_dir / item.name)
    
    return merged


def merge(cfg) -> int:
    """Command line entry: merges <output_dir>/lora_final.safetensors into the base model."""
    lora_path = Path(cfg.output_dir) / "lora_final.safetensors"
    out_dir = Path(cfg.output_dir) / "merged"
    
    if not lora_path.exists():
        raise FileNotFoundError(f"{lora_path} not found. Train first.")
    
    count = merge_lora_into_model_dir(cfg.base_model, lora_path, out_dir)
    print(f"merged {count} layers from {lora_path}")
    print(f"standalone model written to: {out_dir}")
    print("(the original model and the LoRA file were not changed)")
    return 0