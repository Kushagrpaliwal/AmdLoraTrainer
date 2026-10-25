"""Step 1: smoke test. Confirms the ROCm PyTorch build works and times a BF16 matmul."""
from __future__ import annotations
import time
from .env import apply_amd_env


def run_doctor() -> int:
    applied = apply_amd_env()
    import torch  # after env vars on purpose

    print(f"torch {torch.__version__} | hip {getattr(torch.version, 'hip', None)}")
    if not torch.cuda.is_available():
        print("FAIL: torch.cuda.is_available() is False (ROCm build not active or driver issue)")
        return 1
    dev = torch.device("cuda:0")
    props = torch.cuda.get_device_properties(0)
    free, total = torch.cuda.mem_get_info()
    print(f"device: {props.name} | arch: {getattr(props, 'gcnArchName', '?')} | "
          f"VRAM {total/2**30:.1f} GiB ({free/2**30:.1f} free)")
    print("BF16 supported:", torch.cuda.is_bf16_supported())

    n = 4096
    a = torch.randn(n, n, device=dev, dtype=torch.bfloat16)
    b = torch.randn(n, n, device=dev, dtype=torch.bfloat16)
    for _ in range(3):
        a @ b
    torch.cuda.synchronize()
    t = time.perf_counter()
    iters = 20
    for _ in range(iters):
        a @ b
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t) / iters
    print(f"BF16 matmul {n}x{n}: {dt*1e3:.2f} ms  (~{2*n**3/dt/1e12:.1f} TFLOPS)")

    q = torch.randn(1, 8, 1024, 64, device=dev, dtype=torch.bfloat16)
    from torch.nn.attention import sdpa_kernel, SDPBackend
    for name, be in [("flash", SDPBackend.FLASH_ATTENTION), ("efficient", SDPBackend.EFFICIENT_ATTENTION)]:
        try:
            with sdpa_kernel(be):
                torch.nn.functional.scaled_dot_product_attention(q, q, q)
            print(f"SDPA {name}: OK")
        except Exception as e:
            print(f"SDPA {name}: unavailable ({type(e).__name__})")
    print("env applied:", applied)
    return 0
