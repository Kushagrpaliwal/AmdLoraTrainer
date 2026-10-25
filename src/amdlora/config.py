"""Step 1: single typed config for every setting. Loaded from YAML."""
from __future__ import annotations
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any
import yaml


@dataclass
class LoraCfg:
    rank: int = 16
    alpha: float = 16.0
    dropout: float = 0.0
    target_modules: list[str] = field(default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"])


@dataclass
class TrainCfg:
    epochs: int = 1
    micro_batch_size: int = 1
    grad_accum: int = 1
    lr: float = 1e-4
    warmup_steps: int = 0
    max_grad_norm: float = 1.0
    optimizer: str = "adamw"      # adamw | adamw8bit | paged_adamw
    seed: int = 42
    save_every: int = 500


@dataclass
class PrecisionCfg:
    compute_dtype: str = "bf16"
    base_load: str = "bf16"       # bf16 | fp8 | nf4


@dataclass
class MemoryCfg:
    gradient_checkpointing: bool = True
    cpu_offload: bool = False
    cache_latents: bool = True
    cache_text_embeds: bool = True
    tiled_vae: bool = False


@dataclass
class AmdCfg:
    compile: bool = False
    tunableop: bool = True
    hipblaslt: bool = False
    miopen_find_mode: str = "FAST"
    aotriton_experimental: bool = True


@dataclass
class Config:
    model_family: str = "llm"     # llm | diffusion
    base_model: str = ""
    dataset_dir: str = ""
    output_dir: str = "outputs/run"
    lora: LoraCfg = field(default_factory=LoraCfg)
    train: TrainCfg = field(default_factory=TrainCfg)
    precision: PrecisionCfg = field(default_factory=PrecisionCfg)
    memory: MemoryCfg = field(default_factory=MemoryCfg)
    attention: str = "sdpa_flash"
    amd: AmdCfg = field(default_factory=AmdCfg)


def _build(cls, data: dict[str, Any]):
    known = {f.name for f in fields(cls)}
    kwargs = {}
    for k, v in (data or {}).items():
        if k not in known:
            raise ValueError(f"Unknown config key '{k}' in {cls.__name__}")
        default = getattr(cls(), k)
        kwargs[k] = _build(type(default), v) if is_dataclass(default) and isinstance(v, dict) else v
    return cls(**kwargs)


def load_config(path: str | Path) -> Config:
    with open(path, "r", encoding="utf-8") as f:
        return _build(Config, yaml.safe_load(f))
