"""Step 1: AMD environment variables. MUST run before `import torch`."""
from __future__ import annotations
import os
from pathlib import Path
from .config import AmdCfg


def apply_amd_env(cfg: AmdCfg | None = None, cache_root: str = ".cache") -> dict[str, str]:
    cfg = cfg or AmdCfg()
    root = Path(cache_root).resolve()
    (root / "miopen").mkdir(parents=True, exist_ok=True)
    (root / "triton").mkdir(parents=True, exist_ok=True)
    wanted = {
        "TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL": "1" if cfg.aotriton_experimental else "0",
        "TORCH_BLAS_PREFER_HIPBLASLT": "1" if cfg.hipblaslt else "0",   # PDF addendum: rocBLAS + TunableOp
        "FLASH_ATTENTION_TRITON_AMD_ENABLE": "TRUE",
        "MIOPEN_FIND_MODE": cfg.miopen_find_mode,
        "MIOPEN_USER_DB_PATH": str(root / "miopen"),
        "TRITON_CACHE_DIR": str(root / "triton"),
        # expandable_segments may be unsupported on Windows ROCm: test in Step 8
        "PYTORCH_HIP_ALLOC_CONF": "garbage_collection_threshold:0.8,max_split_size_mb:512",
    }
    if cfg.tunableop:
        wanted.update({
            "PYTORCH_TUNABLEOP_ENABLED": "1",
            "PYTORCH_TUNABLEOP_TUNING": "0",   # reuse saved results; set to 1 for a tuning run
            "PYTORCH_TUNABLEOP_FILENAME": str(root / "tunableop_results.csv"),
        })
    applied = {}
    for k, v in wanted.items():
        if k not in os.environ:          # never override what the user set explicitly
            os.environ[k] = v
            applied[k] = v
    return applied
