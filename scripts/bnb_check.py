"""Step 8: quick check of bitsandbytes (4-bit NF4 and 8-bit optimizers) on your setup.

Usage:  python scripts/bnb_check.py
bitsandbytes is experimental on AMD + Windows, so a failure here is normal and fine:
BF16 training does not need it.
"""
from amdlora.env import apply_amd_env

apply_amd_env()

import torch  # noqa: E402


def main():
    try:
        import bitsandbytes as bnb
    except Exception as error:
        print(f"bitsandbytes is not usable: {type(error).__name__}: {str(error)[:150]}")
        print("RESULT: NF4 / AdamW8bit unavailable. That is fine, keep using BF16 + adamw.")
        return 0

    print("bitsandbytes imported, version", getattr(bnb, "__version__", "?"))

    try:
        layer = bnb.nn.Linear4bit(64, 64, quant_type="nf4", compute_dtype=torch.bfloat16).cuda()
        x = torch.randn(2, 64, device="cuda", dtype=torch.bfloat16)
        print("NF4 4-bit layer forward: OK, output", tuple(layer(x).shape))
    except Exception as error:
        print(f"NF4 4-bit layer FAILED: {type(error).__name__}: {str(error)[:150]}")

    try:
        weight = torch.nn.Parameter(torch.randn(64, 64, device="cuda"))
        weight.grad = torch.randn_like(weight)
        bnb.optim.AdamW8bit([weight], lr=1e-3).step()
        print("AdamW8bit step: OK")
    except Exception as error:
        print(f"AdamW8bit FAILED: {type(error).__name__}: {str(error)[:150]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())