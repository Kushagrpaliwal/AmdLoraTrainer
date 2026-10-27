"""AMD / ROCm 10.1 feature test v2 for the LoRA trainer (Windows, RX 7800 XT / gfx1101).

Run:
    python feature_smoke_test.py                  # with the recommended env vars
    python feature_smoke_test.py --baseline       # no env vars set (for A/B comparison)
    python feature_smoke_test.py --only 7,8       # run only tests whose number is listed

Needs (besides the ROCm torch wheels):
    pip install "triton-windows<3.9" einops
    pip install --no-deps flash_attn==2.8.3 --find-links https://rocm.frameworks.amd.com/whl/device-all-rdna/

Statuses:  PASS = works and results are correct
           WARN = runs, but degraded / slower than expected / needs a fix (see "hint")
           FAIL = crashed or gave wrong numbers
           SKIP = package not installed
           INFO = expected-unsupported hardware feature (not a bug)

Output: console summary + amd_report_v2[_baseline].json
"""
import os
import shutil
import subprocess
import sys


def load_msvc_env():
    """Put cl.exe on PATH by importing the vcvars64.bat environment (Inductor's CPU path needs it)."""
    if shutil.which("cl"):
        return "cl.exe already on PATH"
    vswhere = os.path.join(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"),
                           "Microsoft Visual Studio", "Installer", "vswhere.exe")
    if not os.path.isfile(vswhere):
        return None
    root = subprocess.run([vswhere, "-latest", "-products", "*", "-requires",
                           "Microsoft.VisualStudio.Component.VC.Tools.x86.x64", "-property", "installationPath"],
                          capture_output=True, text=True).stdout.strip()
    bat = os.path.join(root, "VC", "Auxiliary", "Build", "vcvars64.bat")
    if not os.path.isfile(bat):
        return None
    out = subprocess.run(f'cmd /d /s /c ""{bat}" >nul 2>&1 && set"', capture_output=True, text=True).stdout
    for line in out.splitlines():
        k, sep, v = line.partition("=")
        if sep and k:
            os.environ[k] = v
    return bat if shutil.which("cl") else None


# ---- env vars must be set BEFORE torch is imported -------------------------------------------
BASELINE = "--baseline" in sys.argv
HERE = os.path.dirname(os.path.abspath(__file__))
MSVC_STATUS = None
if not BASELINE:
    MSVC_STATUS = load_msvc_env()
    os.environ.setdefault("BNB_ROCM_VERSION", "715")          # bnb ships rocm714/715/72 DLLs, none for HIP 7.16
    os.environ.setdefault("TORCH_BLAS_PREFER_HIPBLASLT", "0")  # rocBLAS: faster on LoRA shapes, allows HIP graphs
    os.makedirs(os.path.join(HERE, ".miopen_db"), exist_ok=True)
    os.makedirs(os.path.join(HERE, ".triton_cache"), exist_ok=True)
    os.environ.setdefault("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1")  # built-in flash/efficient SDPA
    os.environ.setdefault("MIOPEN_FIND_MODE", "FAST")                      # faster first conv runs
    os.environ.setdefault("MIOPEN_USER_DB_PATH", os.path.join(HERE, ".miopen_db"))
    os.environ.setdefault("TRITON_CACHE_DIR", os.path.join(HERE, ".triton_cache"))
    os.environ.setdefault("FLASH_ATTENTION_TRITON_AMD_ENABLE", "TRUE")     # flash_attn pkg -> Triton backend
    os.environ.setdefault("PYTORCH_HIP_ALLOC_CONF", "garbage_collection_threshold:0.8,max_split_size_mb:512")
WATCHED_ENV = ["TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "MIOPEN_FIND_MODE", "MIOPEN_USER_DB_PATH",
               "TRITON_CACHE_DIR", "FLASH_ATTENTION_TRITON_AMD_ENABLE", "PYTORCH_HIP_ALLOC_CONF",
               "HSA_OVERRIDE_GFX_VERSION", "BNB_ROCM_VERSION", "TORCH_BLAS_PREFER_HIPBLASLT"]

import copy
import json
import statistics
import time
import traceback
import warnings
from importlib import metadata

import torch
import torch.nn.functional as F

RESULTS = {}
ONLY = None
if "--only" in sys.argv:
    ONLY = {x.strip() for x in sys.argv[sys.argv.index("--only") + 1].split(",")}


# ---- harness ---------------------------------------------------------------------------------
def check(name):
    """Run one test, record status, never crash the script."""
    def wrap(fn):
        num = name.split(".")[0]
        if ONLY and num not in ONLY:
            return fn
        print(f"\n=== {name} ===", flush=True)
        t0 = time.time()
        try:
            res = fn()
            status, info = res if isinstance(res, tuple) else ("PASS", res)
        except Exception as e:  # noqa: BLE001
            status, info = "FAIL", f"{type(e).__name__}: {str(e)[:600]}"
            traceback.print_exc(limit=3)
        RESULTS[name] = {"status": status, "info": info, "seconds": round(time.time() - t0, 1)}
        print("->", status, "|", json.dumps(info, indent=2, default=str) if isinstance(info, (dict, list)) else info,
              flush=True)
        try:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass
        return fn
    return wrap


def sync():
    torch.cuda.synchronize()


def bench(fn, iters=20, warmup=5, repeats=5, wall=False):
    """Median / min / max milliseconds per call, timed with CUDA events.
    wall=True times with perf_counter + synchronize instead: on this HIP build events do not include the time of
    pinned host->device copies (test 27), so anything that moves data between CPU and GPU must use it."""
    for _ in range(warmup):
        fn()
    sync()
    times = []
    for _ in range(repeats):
        if wall:
            t0 = time.perf_counter()
            for _ in range(iters):
                fn()
            sync()
            times.append((time.perf_counter() - t0) * 1e3 / iters)
            continue
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(iters):
            fn()
        e.record()
        e.synchronize()
        times.append(s.elapsed_time(e) / iters)
    times.sort()
    return {"med": statistics.median(times), "min": times[0], "max": times[-1]}


def rel_l2(a, b):
    a, b = a.float(), b.float()
    return ((a - b).norm() / (b.norm() + 1e-12)).item()


def pkg_version(name):
    try:
        return metadata.version(name)
    except Exception:  # noqa: BLE001
        return None


def subproc(code, set_env=None, drop_env=(), timeout=600):
    """Run code in a fresh python (env vars only take effect at startup; crashes stay contained).
    The child reports by printing one line 'RESULT <json>'. Returns (returncode, result_or_None, stderr_tail)."""
    env = {k: v for k, v in os.environ.items() if k not in drop_env}
    env.update(set_env or {})
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=timeout,
                       cwd=HERE)
    res = None
    for line in r.stdout.splitlines():
        if line.startswith("RESULT "):
            res = json.loads(line[7:])
    tail = [l for l in (r.stderr or "").strip().splitlines() if "Warning [ParseAndLoadDb]" not in l][-3:]
    return r.returncode, res, " | ".join(tail)[:400]


if not torch.cuda.is_available():
    sys.exit("No GPU visible to PyTorch. Install the ROCm build of PyTorch first.")
dev = torch.device("cuda")
BF16, FP16 = torch.bfloat16, torch.float16


# ---- 1 -----------------------------------------------------------------------------------------
@check("1. Environment")
def _():
    p = torch.cuda.get_device_properties(0)
    free, total = torch.cuda.mem_get_info()
    return {
        "mode": "baseline (no env vars)" if BASELINE else "recommended env vars",
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "hip_runtime": getattr(torch.version, "hip", None),
        "hip_note": "ROCm 10.1 reports HIP 7.16 - this is normal, not a version mismatch",
        "device": p.name,
        "arch": getattr(p, "gcnArchName", "unknown"),
        "vram_total_gb": round(total / 1e9, 1),
        "vram_free_gb": round(free / 1e9, 1),
        "bf16": torch.cuda.is_bf16_supported(),
        "miopen_enabled": torch.backends.cudnn.enabled,
        "blas_default": str(torch.backends.cuda.preferred_blas_library()),
        "triton_windows": pkg_version("triton-windows") or pkg_version("triton"),
        "bitsandbytes": pkg_version("bitsandbytes"),
        "flash_attn": pkg_version("flash_attn") or pkg_version("flash-attn"),
        "rocm_pkg": pkg_version("rocm"),
        "env": {k: os.environ.get(k) for k in WATCHED_ENV},
    }


# ---- 2 -----------------------------------------------------------------------------------------
@check("2. ROCm CLI tools + VRAM telemetry")
def _():
    wanted = ("hipinfo", "hipcc", "rocminfo", "amd-smi", "rocm-smi", "rocprofv3")
    found = {t: shutil.which(t) for t in wanted if shutil.which(t)}
    sdk_bin = None
    try:
        r = subprocess.run(["rocm-sdk", "path", "--bin"], capture_output=True, text=True, timeout=30)
        sdk_bin = r.stdout.strip() or None
    except Exception:  # noqa: BLE001
        pass
    if sdk_bin and os.path.isdir(sdk_bin):
        for t in wanted:
            for ext in (".exe", ".bat", ""):
                cand = os.path.join(sdk_bin, t + ext)
                if t not in found and os.path.isfile(cand):
                    found[t] = cand
    free, total = torch.cuda.mem_get_info()
    out = {"found": found, "rocm_sdk_bin": sdk_bin,
           "vram_used_mb_via_torch": round((total - free) / 1e6)}
    if "amd-smi" not in found:
        return "WARN", {**out, "hint": "amd-smi not found (ROCm SMI was removed from the standard build in 10.x). "
                                      "Use torch.cuda.mem_get_info() / max_memory_allocated() for VRAM logging."}
    return out


# ---- 3 -----------------------------------------------------------------------------------------
@check("3. BLAS backends on real LoRA shapes: hipBLASLt vs rocBLAS (subprocess A/B)")
def _():
    # torch.backends.cuda.preferred_blas_library() is a no-op on Windows (it warns "not supported on Windows"),
    # so the library is chosen with TORCH_BLAS_PREFER_HIPBLASLT, which is only read at startup.
    code = r'''
import json, statistics, torch
shapes = [("square4096", 4096, 4096, 4096, torch.bfloat16), ("square4096_fp16", 4096, 4096, 4096, torch.float16),
          ("lora_down 8192x4096x16", 8192, 16, 4096, torch.bfloat16),
          ("lora_up   8192x16x4096", 8192, 4096, 16, torch.bfloat16),
          ("mlp 8192x4096x11008", 8192, 11008, 4096, torch.bfloat16)]
out = {}
for name, M, N, K, dt in shapes:
    a, b = torch.randn(M, K, device="cuda", dtype=dt), torch.randn(K, N, device="cuda", dtype=dt)
    err = ((a @ b).float() - a.float() @ b.float()).norm() / (a.float() @ b.float()).norm()
    for _ in range(5): a @ b
    ts = []
    for _ in range(5):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(15): a @ b
        e.record(); e.synchronize(); ts.append(s.elapsed_time(e) / 15)
    out[name] = {"ms": statistics.median(ts), "tflops": 2 * M * N * K / (statistics.median(ts) / 1e3) / 1e12,
                 "rel_err": err.item()}
print("RESULT " + json.dumps(out))
'''
    runs = {}
    for lib, val in (("hipBLASLt", "1"), ("rocBLAS", "0")):
        rc, res, err = subproc(code, {"TORCH_BLAS_PREFER_HIPBLASLT": val})
        if res is None:
            return "FAIL", f"{lib} run crashed (rc {rc}): {err}"
        runs[lib] = res
    table, bad, rocblas_wins = {}, [], []
    for name in runs["rocBLAS"]:
        h, r = runs["hipBLASLt"][name], runs["rocBLAS"][name]
        table[name] = {"hipBLASLt": f"{h['tflops']:.1f} TFLOPS ({h['ms']:.3f} ms)",
                       "rocBLAS": f"{r['tflops']:.1f} TFLOPS ({r['ms']:.3f} ms)",
                       "faster": "rocBLAS" if r["ms"] < h["ms"] else "hipBLASLt"}
        if r["ms"] < h["ms"] * 0.9:
            rocblas_wins.append(name)
        bad += [f"{name}/{lib}: rel err {runs[lib][name]['rel_err']:.3f}" for lib in runs
                if runs[lib][name]["rel_err"] > 0.02]
    out = {"tflops": table, "torch_default": "hipBLASLt (TORCH_BLAS_PREFER_HIPBLASLT unset)",
           "this_run_uses": "rocBLAS" if os.environ.get("TORCH_BLAS_PREFER_HIPBLASLT") == "0" else "hipBLASLt"}
    if bad:
        return "FAIL", {**out, "numerical_problems": bad}
    if rocblas_wins:
        out["hint"] = (f"rocBLAS is >10% faster on {rocblas_wins}: set TORCH_BLAS_PREFER_HIPBLASLT=0 before importing "
                       "torch (or use TunableOp, test 5/26, which picks per shape).")
    return out


# ---- 4 -----------------------------------------------------------------------------------------
@check("4. Which GEMM kernels run (profiler, GPU-side names)")
def _():
    from torch.profiler import ProfilerActivity, profile
    a, b = torch.randn(2048, 2048, device=dev, dtype=BF16), torch.randn(2048, 2048, device=dev, dtype=BF16)
    a @ b
    sync()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(5):
            a @ b
        sync()
    names = sorted({e.name for e in prof.events()
                    if "CUDA" in str(getattr(e, "device_type", "")) and not e.name.startswith("aten::")})
    if not names:
        return "WARN", {"gpu_kernels": [], "hint": "Kineto returned no GPU-side events on this Windows build. "
                        "Not a trainer blocker. Test 5 identifies the GEMM library through the TunableOp CSV instead."}
    return {"gpu_kernels": names[:8],
            "library_guess": ("hipBLASLt" if any("hipblaslt" in n.lower() or "Cijk" in n for n in names)
                              else "rocBLAS/other")}


# ---- 5 -----------------------------------------------------------------------------------------
@check("5. TunableOp (rocBLAS vs hipBLASLt auto-tuning per GEMM shape)")
def _():
    import torch.cuda.tunable as tunable
    csv_path = os.path.join(HERE, "tunableop_results.csv")
    if os.path.exists(csv_path):
        os.remove(csv_path)
    shapes = [(8192, 16, 4096), (8192, 4096, 16), (8192, 4096, 4096), (4096, 4096, 4096)]
    mats = [(torch.randn(M, K, device=dev, dtype=BF16), torch.randn(K, N, device=dev, dtype=BF16))
            for M, N, K in shapes]
    tunable.enable(False)
    before = [bench(lambda a=a, b=b: a @ b, iters=15)["med"] for a, b in mats]
    tunable.set_filename(csv_path, insert_device_ordinal=False)
    tunable.enable(True)
    tunable.tuning_enable(True)
    try:
        tunable.set_max_tuning_duration(100)
        tunable.set_max_tuning_iterations(50)
    except Exception:  # noqa: BLE001
        pass
    for a, b in mats:
        for _ in range(3):
            a @ b
    sync()
    tunable.tuning_enable(False)  # use the tuned solutions, no more tuning
    after = [bench(lambda a=a, b=b: a @ b, iters=15)["med"] for a, b in mats]
    try:
        tunable.write_file(csv_path)
    except Exception:  # noqa: BLE001
        pass
    tunable.enable(False)
    libs = {}
    if os.path.exists(csv_path):
        text = open(csv_path, encoding="utf-8", errors="ignore").read().lower()
        libs = {k: text.count(k) for k in ("hipblaslt", "rocblas", "default")}
    speed = {f"{M}x{N}x{K}": f"{b / a_:.2f}x ({b:.3f} -> {a_:.3f} ms)"
             for (M, N, K), b, a_ in zip(shapes, before, after)}
    out = {"speedup": speed, "solutions_in_csv": libs, "csv": csv_path if libs else None}
    if not libs:
        return "WARN", {**out, "hint": "TunableOp wrote no results; this build may not support it on Windows."}
    return out


# ---- 6 -----------------------------------------------------------------------------------------
@check("6. MIOpen at real VAE sizes (layout, correctness, first-call delay)")
def _():
    if not torch.backends.cudnn.is_available():
        return "SKIP", "MIOpen not available"
    out, problems = {}, []
    for name, C, H in (("vae_mid 512ch @128x128", 512, 128), ("vae_up 128ch @1024x1024", 128, 1024)):
        conv = torch.nn.Conv2d(C, C, 3, padding=1).to(dev, BF16)
        x = torch.randn(1, C, H, H, device=dev, dtype=BF16)
        t0 = time.time()
        y = conv(x)
        sync()
        first_ms = (time.time() - t0) * 1e3
        r_plain = bench(lambda: conv(x), iters=5, warmup=2, repeats=3)
        conv_cl = copy.deepcopy(conv).to(memory_format=torch.channels_last)
        x_cl = x.to(memory_format=torch.channels_last)
        y_cl = conv_cl(x_cl)
        r_cl = bench(lambda: conv_cl(x_cl), iters=5, warmup=2, repeats=3)
        with torch.no_grad():
            ref = F.conv2d(x.float(), conv.weight.float(), conv.bias.float(), padding=1)
        flops = 2 * 9 * C * C * H * H
        e_plain, e_cl = rel_l2(y, ref), rel_l2(y_cl, ref)
        try:
            with torch.backends.cudnn.flags(enabled=False):
                r_off = bench(lambda: conv(x), iters=2, warmup=1, repeats=2)["med"]
        except Exception:  # noqa: BLE001
            r_off = None
        out[name] = {
            "first_call_ms": round(first_ms, 1),
            "contiguous": f"{r_plain['med']:.2f} ms ({flops / r_plain['med'] / 1e9:.1f} TFLOPS)",
            "channels_last": f"{r_cl['med']:.2f} ms ({flops / r_cl['med'] / 1e9:.1f} TFLOPS)",
            "miopen_off": f"{r_off:.2f} ms" if r_off else "n/a",
            "rel_err_vs_fp32": {"contiguous": round(e_plain, 4), "channels_last": round(e_cl, 4)},
        }
        if max(e_plain, e_cl) > 0.05 or torch.isnan(y_cl).any().item():
            problems.append(name)
    out["recommendation"] = "use whichever layout is faster for your VAE; keep it fixed for the whole run"
    if problems:
        return "FAIL", {**out, "wrong_results": problems}
    return out


# ---- 7 -----------------------------------------------------------------------------------------
@check("7. Attention backends: forward + backward, vs fp32 reference")
def _():
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from torch.utils.checkpoint import checkpoint

    def chunked(q, k, v, chunk=1024):
        outs = [checkpoint(lambda qi, k_, v_: F.scaled_dot_product_attention(qi, k_, v_),
                           q[:, :, i:i + chunk], k, v, use_reentrant=False)
                for i in range(0, q.shape[2], chunk)]
        return torch.cat(outs, dim=2)

    backends = {"flash": lambda q, k, v: _sdpa(SDPBackend.FLASH_ATTENTION, q, k, v),
                "efficient": lambda q, k, v: _sdpa(SDPBackend.EFFICIENT_ATTENTION, q, k, v),
                "math": lambda q, k, v: _sdpa(SDPBackend.MATH, q, k, v),
                "chunked_fallback": chunked}

    def _sdpa(be, q, k, v):
        with sdpa_kernel(be):
            return F.scaled_dot_product_attention(q, k, v)

    cases = [("bf16 N2048 D64", 1, 16, 2048, 64, BF16), ("fp16 N2048 D64", 1, 16, 2048, 64, FP16),
             ("bf16 N4096 D128 (FLUX-like)", 1, 16, 4096, 128, BF16)]
    report, fast_ok = {}, {"flash": True, "efficient": True}
    for cname, B, H, N, D, dt in cases:
        torch.manual_seed(0)
        q = torch.randn(B, H, N, D, device=dev, dtype=dt, requires_grad=True)
        k = torch.randn_like(q, requires_grad=True)
        v = torch.randn_like(q, requires_grad=True)
        g = torch.randn(B, H, N, D, device=dev, dtype=dt)
        qf, kf, vf = (t.detach().float().requires_grad_(True) for t in (q, k, v))
        with sdpa_kernel(SDPBackend.MATH):
            ref = F.scaled_dot_product_attention(qf, kf, vf)
        ref.backward(g.float())
        ref_dq = qf.grad.clone()
        del qf, kf, vf
        row = {}
        for bname, fn in backends.items():
            try:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    q.grad = None
                    out = fn(q, k, v)
                    out.backward(g)
                    sync()
                reason = "; ".join(sorted({str(w.message)[:140] for w in caught}))[:300]
                err_f, err_b = rel_l2(out, ref), rel_l2(q.grad, ref_dq)
                torch.cuda.reset_peak_memory_stats()
                base = torch.cuda.memory_allocated()

                def step():
                    q.grad = None
                    fn(q, k, v).backward(g)
                t_f = bench(lambda: fn(q, k, v).detach(), iters=5, warmup=2, repeats=3)["med"]
                t_fb = bench(step, iters=3, warmup=1, repeats=3)["med"]
                peak = (torch.cuda.max_memory_allocated() - base) / 1e9
                ok = err_f < 0.03 and err_b < 0.05 and not torch.isnan(out).any().item()
                row[bname] = {"fwd_ms": round(t_f, 2), "fwd+bwd_ms": round(t_fb, 2),
                              "extra_vram_gb": round(peak, 2), "rel_err_fwd": round(err_f, 4),
                              "rel_err_dq": round(err_b, 4), "correct": ok, **({"warn": reason} if reason else {})}
                if not ok and bname in fast_ok:
                    fast_ok[bname] = False
            except Exception as e:  # noqa: BLE001
                row[bname] = {"unavailable": str(e)[:160]}
                if bname in fast_ok:
                    fast_ok[bname] = False
            q.grad = None
        report[cname] = row
        del q, k, v, g
        torch.cuda.empty_cache()
    out = {"cases": report}
    if any(fast_ok.values()):
        return out
    return "WARN", {**out, "hint": (
        "Flash and efficient SDPA are unavailable. Fixes in order: (1) make sure "
        "TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 is set before torch loads (run without --baseline); "
        "(2) test 11 (flash_attn Triton backend, needs triton-windows); (3) use the chunked_fallback "
        "above, which is verified correct and uses far less memory than the math path.")}


# ---- 8 -----------------------------------------------------------------------------------------
@check("8. Triton (elementwise + tl.dot matmul kernel)")
def _():
    try:
        import triton
        import triton.language as tl
    except ImportError:
        return "SKIP", 'triton not installed. Try:  pip install "triton-windows<3.9"'

    @triton.jit
    def add_kernel(x_ptr, y_ptr, o_ptr, n, BLOCK: tl.constexpr):
        i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        m = i < n
        tl.store(o_ptr + i, tl.load(x_ptr + i, mask=m) + tl.load(y_ptr + i, mask=m), mask=m)

    @triton.jit
    def mm_kernel(a_ptr, b_ptr, c_ptr, M, N, K, sam, sak, sbk, sbn, scm, scn,
                  BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
        rm = tl.program_id(0) * BM + tl.arange(0, BM)
        rn = tl.program_id(1) * BN + tl.arange(0, BN)
        rk = tl.arange(0, BK)
        a_ptrs = a_ptr + rm[:, None] * sam + rk[None, :] * sak
        b_ptrs = b_ptr + rk[:, None] * sbk + rn[None, :] * sbn
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for k in range(0, K, BK):
            a = tl.load(a_ptrs, mask=(rm[:, None] < M) & (rk[None, :] + k < K), other=0.0)
            b = tl.load(b_ptrs, mask=(rk[:, None] + k < K) & (rn[None, :] < N), other=0.0)
            acc += tl.dot(a, b)
            a_ptrs += BK * sak
            b_ptrs += BK * sbk
        tl.store(c_ptr + rm[:, None] * scm + rn[None, :] * scn, acc.to(tl.bfloat16),
                 mask=(rm[:, None] < M) & (rn[None, :] < N))

    n = 1 << 20
    x, y = torch.randn(n, device=dev), torch.randn(n, device=dev)
    o = torch.empty_like(x)
    add_kernel[((n + 1023) // 1024,)](x, y, o, n, BLOCK=1024)
    sync()
    if not torch.allclose(o, x + y):
        raise RuntimeError("Triton vector add gave wrong results")
    M = N = K = 2048
    a, b = torch.randn(M, K, device=dev, dtype=BF16), torch.randn(K, N, device=dev, dtype=BF16)
    c = torch.empty(M, N, device=dev, dtype=BF16)
    grid = (M // 64, N // 64)
    launch = lambda: mm_kernel[grid](a, b, c, M, N, K, a.stride(0), a.stride(1), b.stride(0),  # noqa: E731
                                     b.stride(1), c.stride(0), c.stride(1), BM=64, BN=64, BK=32)
    launch()
    sync()
    err = rel_l2(c, a @ b)
    if err > 0.02:
        raise RuntimeError(f"Triton tl.dot matmul wrong, rel err {err:.3f}")
    t_tri, t_torch = bench(launch, iters=10)["med"], bench(lambda: a @ b, iters=10)["med"]
    return {"triton": pkg_version("triton-windows") or pkg_version("triton"), "vector_add": "correct",
            "tl.dot_matmul": f"correct (rel err {err:.4f})",
            "naive_triton_ms": round(t_tri, 3), "torch_ms": round(t_torch, 3),
            "note": "naive untuned kernel; slower than rocBLAS is expected. This only proves tl.dot compiles and runs."}


# ---- 9 -----------------------------------------------------------------------------------------
@check("9. torch.compile (inductor needs Triton; aot_eager shown as a control)")
def _():
    model = torch.nn.Sequential(torch.nn.Linear(2048, 2048), torch.nn.GELU(),
                                torch.nn.Linear(2048, 2048)).to(dev, BF16)
    x = torch.randn(256, 2048, device=dev, dtype=BF16, requires_grad=True)

    def step(m):
        return lambda: m(x).float().pow(2).mean().backward()
    out = {"eager_fwd+bwd_ms": round(bench(step(model), iters=20)["med"], 3)}
    try:
        torch.compile(model, backend="aot_eager")(x).sum().backward()
        out["aot_eager"] = "works (compile machinery OK, no kernel fusion)"
    except Exception as e:  # noqa: BLE001
        out["aot_eager"] = f"failed: {str(e)[:120]}"
    try:
        comp = torch.compile(model)
        step(comp)()
        sync()
        out["inductor_fwd+bwd_ms"] = round(bench(step(comp), iters=20)["med"], 3)
        return out
    except Exception as e:  # noqa: BLE001
        msg = str(e)
        hint = ('Install Triton:  pip install "triton-windows<3.9"' if "riton" in msg
                else "Inductor needs a C/C++ compiler on Windows: install Visual Studio Build Tools (MSVC) and run "
                     "from an 'x64 Native Tools' prompt.")
        return "WARN", {**out, "inductor": msg[:200], "hint": hint}


# ---- 10 ----------------------------------------------------------------------------------------
@check("10. bitsandbytes: 4-bit forward + backward + 8-bit optimizers")
def _():
    try:
        import bitsandbytes as bnb
    except ImportError:
        return "SKIP", "bitsandbytes not installed"
    torch.manual_seed(0)
    lin = torch.nn.Linear(1024, 1024, bias=False)
    q = bnb.nn.Linear4bit(1024, 1024, bias=False, compute_dtype=BF16, quant_type="nf4")
    q.load_state_dict(lin.state_dict())
    q = q.to(dev)
    w_ref = lin.weight.data.to(dev, BF16)
    x = torch.randn(64, 1024, device=dev, dtype=BF16, requires_grad=True)
    A = torch.nn.Parameter(torch.randn(16, 1024, device=dev, dtype=BF16) * 0.01)
    Bm = torch.nn.Parameter(torch.randn(1024, 16, device=dev, dtype=BF16) * 0.01)
    y = q(x) + (x @ A.t()) @ Bm.t()
    y.float().pow(2).mean().backward()
    sync()
    y_dense = x.detach() @ w_ref.t()
    fwd_err = rel_l2(q(x.detach()), y_dense)
    out = {"bitsandbytes": bnb.__version__, "nf4_forward_rel_err_vs_bf16": round(fwd_err, 4),
           "grad_x_finite": bool(torch.isfinite(x.grad).all()),
           "lora_grads_nonzero": bool(A.grad.abs().sum() > 0 and Bm.grad.abs().sum() > 0)}
    problems = []
    if fwd_err > 0.2:
        problems.append("NF4 forward error too large")
    if not out["grad_x_finite"] or not out["lora_grads_nonzero"]:
        problems.append("backward through 4-bit layer broken")
    for oname in ("AdamW8bit", "PagedAdamW8bit", "PagedAdamW32bit"):
        try:
            opt = getattr(bnb.optim, oname)([A, Bm], lr=1e-3)
            before = A.detach().clone()
            opt.step()
            sync()
            out[oname] = "works" if not torch.equal(before, A.detach()) else "ran but did not update weights"
        except Exception as e:  # noqa: BLE001
            out[oname] = f"failed: {str(e)[:100]}"
    try:
        l8 = bnb.nn.Linear8bitLt(256, 256, bias=False, has_fp16_weights=False).to(dev)
        out["Linear8bitLt_forward"] = list(l8(torch.randn(4, 256, device=dev, dtype=FP16)).shape)
    except Exception as e:  # noqa: BLE001
        out["Linear8bitLt_forward"] = f"failed: {str(e)[:100]}"
    if problems:
        return "FAIL", {**out, "problems": problems}
    if any(str(out.get(o, "")).startswith("failed") for o in ("AdamW8bit",)):
        return "WARN", {**out, "hint": "8-bit optimizer failed: use torch AdamW on the small LoRA params instead."}
    return out


# ---- 11 ----------------------------------------------------------------------------------------
@check("11. flash_attn package (Triton backend) vs SDPA")
def _():
    try:
        from flash_attn import flash_attn_func
    except Exception as e:  # noqa: BLE001
        return "SKIP", ("flash_attn not importable (" + type(e).__name__ + "). Optional. AMD publishes a Triton-backend "
                        "wheel at https://rocm.frameworks.amd.com/whl/device-all-rdna/ (needs triton-windows).")
    q = torch.randn(1, 2048, 16, 64, device=dev, dtype=BF16, requires_grad=True)
    k, v = torch.randn_like(q, requires_grad=True), torch.randn_like(q, requires_grad=True)
    g = torch.randn_like(q)
    out = flash_attn_func(q, k, v)
    out.backward(g)
    dq = q.grad.clone()
    qf, kf, vf = (t.detach().transpose(1, 2).float().requires_grad_(True) for t in (q, k, v))
    ref = F.scaled_dot_product_attention(qf, kf, vf)
    ref.backward(g.transpose(1, 2).float())
    e_f, e_b = rel_l2(out.transpose(1, 2), ref), rel_l2(dq.transpose(1, 2), qf.grad)
    t = bench(lambda: flash_attn_func(q, k, v), iters=10)["med"]
    res = {"flash_attn": pkg_version("flash_attn"), "fwd_ms": round(t, 2),
           "rel_err_fwd": round(e_f, 4), "rel_err_dq": round(e_b, 4)}
    return res if (e_f < 0.03 and e_b < 0.05) else ("FAIL", {**res, "problem": "outputs differ from SDPA reference"})


# ---- 12 ----------------------------------------------------------------------------------------
@check("12. LoRA layer: init identity, merge algebra, fused addmm, train-step speed")
def _():
    class LoRALinear(torch.nn.Module):
        def __init__(self, base, r, alpha, fused):
            super().__init__()
            self.base, self.scale, self.fused = base, alpha / r, fused
            self.base.requires_grad_(False)
            self.A = torch.nn.Parameter(torch.randn(r, base.in_features, device=dev, dtype=BF16) * 0.01)
            self.B = torch.nn.Parameter(torch.zeros(base.out_features, r, device=dev, dtype=BF16))

        def forward(self, x):
            y, h = self.base(x), x @ self.A.t()
            if self.fused:  # scale + add happen inside the GEMM epilogue
                return torch.addmm(y, h, self.B.t(), alpha=self.scale)
            return y + self.scale * (h @ self.B.t())

    T, Hd, R, L = 4096, 4096, 16, 6
    bases = [torch.nn.Linear(Hd, Hd, bias=False).to(dev, BF16) for _ in range(L)]
    naive = torch.nn.Sequential(*[LoRALinear(b, R, R, False) for b in bases])
    fused = torch.nn.Sequential(*[LoRALinear(b, R, R, True) for b in bases])
    for n, f in zip(naive, fused):
        f.A.data.copy_(n.A.data)
    x = torch.randn(T, Hd, device=dev, dtype=BF16)
    problems = []

    with torch.no_grad():
        chain = x
        for b in bases:
            chain = b(chain)
        d0 = max((naive(x) - chain).abs().max().item(), (fused(x) - chain).abs().max().item())
    if d0 > 1e-6:
        problems.append(f"B=0 output differs from base output by {d0:.2e}")

    for n, f in zip(naive, fused):
        n.B.data.normal_(0, 0.02)
        f.B.data.copy_(n.B.data)
    with torch.no_grad():
        e_fused = rel_l2(fused(x), naive(x))
    if e_fused > 0.02:
        problems.append(f"fused addmm differs from naive: {e_fused:.4f}")

    l0 = naive[0]
    with torch.no_grad():
        Wm = l0.base.weight.float() + l0.scale * (l0.B.float() @ l0.A.float())
        e_merge = rel_l2(x.float() @ Wm.t(),
                         x.float() @ l0.base.weight.float().t()
                         + l0.scale * ((x.float() @ l0.A.float().t()) @ l0.B.float().t()))
    if e_merge > 1e-4:
        problems.append(f"merge algebra error {e_merge:.2e}")

    def train_step(model):
        def s():
            for p in model.parameters():
                p.grad = None
            model(x).float().pow(2).mean().backward()
        return s
    res = {}
    for nm, m in (("naive", naive), ("fused_addmm", fused)):
        torch.cuda.reset_peak_memory_stats()
        r = bench(train_step(m), iters=3, warmup=1, repeats=3)
        res[nm] = f"{r['med']:.1f} ms/step, peak {torch.cuda.max_memory_allocated() / 1e9:.2f} GB"
    res["fused_speedup"] = f"{bench(train_step(naive), iters=3, warmup=1, repeats=3)['med'] / bench(train_step(fused), iters=3, warmup=1, repeats=3)['med']:.2f}x"
    out = {"init_identity_max_diff": d0, "fused_vs_naive_rel_err": round(e_fused, 5),
           "merge_algebra_rel_err": e_merge, "train_step": res,
           "config": f"{L} layers, {Hd} hidden, {T} tokens, rank {R}, bf16"}
    return ("FAIL", {**out, "problems": problems}) if problems else out


# ---- 13 ----------------------------------------------------------------------------------------
@check("13. Allocator settings (expandable_segments probe in a subprocess)")
def _():
    code = ("import torch;x=torch.empty(1<<28,device='cuda');y=torch.empty(1<<27,device='cuda');"
            "del x;z=torch.empty(1<<28,device='cuda');torch.cuda.synchronize();print('OK')")
    res = {}
    for label, var, val in (("expandable_segments", "PYTORCH_HIP_ALLOC_CONF", "expandable_segments:True"),
                            ("gc_threshold+max_split", "PYTORCH_HIP_ALLOC_CONF",
                             "garbage_collection_threshold:0.8,max_split_size_mb:512")):
        env = {**os.environ, var: val}
        r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=180)
        err = (r.stderr or "").strip().splitlines()
        ok = r.returncode == 0 and "OK" in r.stdout
        warn = [l for l in err if "warn" in l.lower() or "not supported" in l.lower() or "ignored" in l.lower()]
        res[label] = "works" if ok and not warn else ("works with warning: " + warn[0][:160] if ok else "FAILED")
    ms = torch.cuda.memory_stats()
    res["this_process_alloc_retries"] = ms.get("num_alloc_retries", 0)
    if "FAILED" in res.values():
        return "WARN", {**res, "hint": "Do not set the failing option in the trainer."}
    if res["expandable_segments"].startswith("works with warning"):
        return "WARN", {**res, "hint": "expandable_segments is ignored on this build; rely on gc_threshold/max_split."}
    return res


# ---- 14 ----------------------------------------------------------------------------------------
@check("14. HIP graphs (CUDAGraph capture of a full train step, in a subprocess)")
def _():
    # Capturing a hipBLASLt GEMM aborts the whole process ("operation not permitted when stream is capturing",
    # hipblaslt.cpp:169), so this runs out-of-process, once per BLAS library.
    code = r'''
import json, torch
model = torch.nn.Sequential(*[torch.nn.Linear(2048, 2048) for _ in range(8)]).to("cuda", torch.bfloat16)
opt = torch.optim.AdamW(model.parameters(), lr=1e-4, capturable=True)
x = torch.randn(16, 2048, device="cuda", dtype=torch.bfloat16)
def step():
    opt.zero_grad(set_to_none=True)
    model(x).float().pow(2).mean().backward()
    opt.step()
def ms(fn, n=20):
    for _ in range(3): fn()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(n): fn()
    e.record(); e.synchronize(); return s.elapsed_time(e) / n
s = torch.cuda.Stream()
s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    for _ in range(3): step()
torch.cuda.current_stream().wait_stream(s)
eager = ms(step)
g = torch.cuda.CUDAGraph()
opt.zero_grad(set_to_none=True)
with torch.cuda.graph(g):
    model(x).float().pow(2).mean().backward()
    opt.step()
print("RESULT " + json.dumps({"eager_ms": eager, "graph_replay_ms": ms(g.replay)}))
'''
    out = {}
    for lib, val in (("hipBLASLt", "1"), ("rocBLAS", "0")):
        rc, res, err = subproc(code, {"TORCH_BLAS_PREFER_HIPBLASLT": val}, timeout=300)
        out[lib] = ({"eager_ms": round(res["eager_ms"], 3), "graph_replay_ms": round(res["graph_replay_ms"], 3),
                     "speedup": f"{res['eager_ms'] / res['graph_replay_ms']:.2f}x"}
                    if res else f"capture crashed the process (rc {rc}): {err[-160:]}")
    if not isinstance(out["rocBLAS"], dict):
        return "FAIL", out
    out["note"] = ("graphs need TORCH_BLAS_PREFER_HIPBLASLT=0 (hipBLASLt cannot be captured), static shapes and "
                   "capturable optimizers; worth it only when a step is launch-bound (small batch / small layers)")
    if out["rocBLAS"]["graph_replay_ms"] > out["rocBLAS"]["eager_ms"] * 1.05:
        return "WARN", {**out, "hint": "capture works with rocBLAS but replay is slower than eager here; skip graphs."}
    return out


# ---- 15 ----------------------------------------------------------------------------------------
@check("15. Pinned-memory transfers (for CPU offload / block swap)")
def _():
    n = 256 * 1024 * 1024  # 512 MB of bf16
    res = {}
    for label, pin in (("pageable", False), ("pinned", True)):
        h = torch.empty(n, dtype=BF16, pin_memory=pin)
        d = torch.empty(n, dtype=BF16, device=dev)
        for direction, f in (("H2D", lambda: d.copy_(h, non_blocking=True)),
                             ("D2H", lambda: h.copy_(d, non_blocking=True))):
            ms = bench(f, iters=3, warmup=1, repeats=3, wall=True)["med"]
            res[f"{label}_{direction}"] = f"{n * 2 / (ms / 1e3) / 1e9:.1f} GB/s"
        del h, d
    return res


# ---- 16 ----------------------------------------------------------------------------------------
@check("16. FP8: storage round-trip (expected) and FP8 matmul (RDNA3: expected unsupported)")
def _():
    w = torch.randn(1024, 1024, device=dev)
    w8 = w.to(torch.float8_e4m3fn)
    err = rel_l2(w8.to(BF16), w)
    out = {"fp8_e4m3_storage_roundtrip_rel_err": round(err, 4)}
    try:
        a8 = torch.randn(64, 64, device=dev).to(torch.float8_e4m3fn)
        b8 = torch.randn(64, 64, device=dev).to(torch.float8_e4m3fn).t()
        one = torch.tensor(1.0, device=dev)
        torch._scaled_mm(a8, b8, scale_a=one, scale_b=one, out_dtype=BF16)
        out["fp8_matmul"] = "works (unexpected on RDNA3 - benchmark it)"
        return out
    except Exception as e:  # noqa: BLE001
        out["fp8_matmul"] = f"not supported: {str(e)[:100]}"
        out["meaning"] = "FP8 weights are storage-only on this card: upcast to BF16 per layer. Saves VRAM, not time."
        return ("INFO", out) if err < 0.1 else ("FAIL", out)


# ---- 17 ----------------------------------------------------------------------------------------
@check("17. LoRA safetensors save/load round-trip (ComfyUI-style keys)")
def _():
    try:
        from safetensors.torch import load_file, save_file
    except ImportError:
        return "SKIP", "safetensors not installed"
    sd = {f"lora_unet_blocks_{i}_attn_to_q.lora_{ab}.weight": torch.randn(16, 64, dtype=BF16)
          for i in range(3) for ab in ("down", "up")}
    path = os.path.join(HERE, "_roundtrip_test.safetensors")
    save_file(sd, path, metadata={"rank": "16", "alpha": "16", "format": "pt"})
    back = load_file(path, device="cuda")
    os.remove(path)
    ok = all(torch.equal(sd[k].to(dev), back[k]) for k in sd)
    return {"keys": len(sd), "bit_exact": ok} if ok else ("FAIL", "tensors changed during save/load")


# ---- 18 ----------------------------------------------------------------------------------------
@check("18. BF16 autocast training (fp32 master weights, loss must drop)")
def _():
    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Linear(1024, 4096), torch.nn.GELU(), torch.nn.Linear(4096, 1024)).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    x, y = torch.randn(512, 1024, device=dev), torch.randn(512, 1024, device=dev)

    def amp_step():
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=BF16):
            out = model(x)
        loss = F.mse_loss(out.float(), y)
        loss.backward()
        opt.step()
        return out, loss

    out, loss0 = amp_step()
    for _ in range(100):
        _, loss = amp_step()
    problems = []
    if out.dtype != BF16:
        problems.append(f"autocast output is {out.dtype}, expected bfloat16")
    if any(p.dtype != torch.float32 or p.grad.dtype != torch.float32 for p in model.parameters()):
        problems.append("master weights / grads are not fp32")
    if not torch.isfinite(loss) or loss.item() > loss0.item() * 0.5:
        problems.append(f"loss did not drop: {loss0.item():.4f} -> {loss.item():.4f}")

    def fp32_step():
        opt.zero_grad(set_to_none=True)
        F.mse_loss(model(x), y).backward()
        opt.step()
    t_amp = bench(amp_step, iters=10, warmup=2, repeats=3)["med"]
    t_32 = bench(fp32_step, iters=10, warmup=2, repeats=3)["med"]
    res = {"loss_start": round(loss0.item(), 4), "loss_after_100_steps": round(loss.item(), 4),
           "autocast_step_ms": round(t_amp, 3), "fp32_step_ms": round(t_32, 3),
           "autocast_speedup": f"{t_32 / t_amp:.2f}x", "grad_scaler": "not needed for bf16"}
    return ("FAIL", {**res, "problems": problems}) if problems else res


# ---- 19 ----------------------------------------------------------------------------------------
@check("19. Gradient checkpointing (same grads, less activation memory)")
def _():
    from torch.utils.checkpoint import checkpoint

    class Block(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.norm = torch.nn.LayerNorm(1024)
            self.mlp = torch.nn.Sequential(torch.nn.Linear(1024, 4096), torch.nn.GELU(), torch.nn.Linear(4096, 1024))

        def forward(self, h):
            return h + self.mlp(self.norm(h))

    torch.manual_seed(0)
    blocks = torch.nn.ModuleList(Block() for _ in range(12)).to(dev, BF16)
    x = torch.randn(8192, 1024, device=dev, dtype=BF16)

    def step(ckpt):
        blocks.zero_grad(set_to_none=True)
        h = x
        for b in blocks:
            h = checkpoint(b, h, use_reentrant=False) if ckpt else b(h)
        h.float().pow(2).mean().backward()

    res, grads = {}, {}
    for ckpt in (False, True):
        step(ckpt)
        sync()
        grads[ckpt] = blocks[0].mlp[0].weight.grad.clone()
        blocks.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        base = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        step(ckpt)
        sync()
        peak = (torch.cuda.max_memory_allocated() - base) / 1e9
        res["checkpointed" if ckpt else "plain"] = {"peak_extra_gb": round(peak, 2),
                                                    "step_ms": round(bench(lambda: step(ckpt), iters=3, warmup=1,
                                                                           repeats=3)["med"], 1)}
    err = rel_l2(grads[True], grads[False])
    saving = 1 - res["checkpointed"]["peak_extra_gb"] / res["plain"]["peak_extra_gb"]
    res.update({"grad_rel_err": err, "memory_saved": f"{saving:.0%}",
                "time_cost": f"{res['checkpointed']['step_ms'] / res['plain']['step_ms']:.2f}x"})
    if err > 1e-2:
        return "FAIL", {**res, "problem": "checkpointed grads differ"}
    if saving < 0.3:
        return "WARN", {**res, "hint": "checkpointing saved less memory than expected"}
    return res


# ---- 20 ----------------------------------------------------------------------------------------
@check("20. Cached text embeddings + VAE latents (encode once, train from disk cache)")
def _():
    from safetensors.torch import load_file, save_file
    torch.manual_seed(0)
    # toy stand-ins with real-ish shapes: 77-token CLIP-like encoder, 8x-downsampling VAE encoder (mean+logvar)
    text_enc = torch.nn.Sequential(torch.nn.Embedding(49408, 768), torch.nn.Linear(768, 2048)).to(dev, BF16)
    vae_enc = torch.nn.Sequential(torch.nn.Conv2d(3, 128, 3, 2, 1), torch.nn.SiLU(), torch.nn.Conv2d(128, 256, 3, 2, 1),
                                  torch.nn.SiLU(), torch.nn.Conv2d(256, 8, 3, 2, 1)).to(dev, BF16)
    cache_dir = os.path.join(HERE, ".cache_test")
    os.makedirs(cache_dir, exist_ok=True)
    n, originals = 8, {}
    with torch.no_grad():  # warm-up so one-time MIOpen init (test 25) is not counted as encode time
        vae_enc(torch.randn(1, 3, 1024, 1024, device=dev, dtype=BF16))
    try:
        sync()
        t0 = time.time()
        with torch.no_grad():
            for i in range(n):
                tokens = torch.randint(0, 49408, (1, 77), device=dev)
                image = torch.randn(1, 3, 1024, 1024, device=dev, dtype=BF16)
                mean, logvar = vae_enc(image).chunk(2, dim=1)
                item = {"text_emb": text_enc(tokens)[0], "latent_mean": mean[0], "latent_logvar": logvar[0]}
                originals[i] = {k: v.clone() for k, v in item.items()}
                save_file({k: v.contiguous().cpu() for k, v in item.items()}, os.path.join(cache_dir, f"{i}.safetensors"))
        sync()
        encode_ms = (time.time() - t0) / n * 1e3
        before = torch.cuda.memory_allocated()
        del text_enc, vae_enc
        torch.cuda.empty_cache()
        freed_mb = (before - torch.cuda.memory_allocated()) / 1e6
        t0 = time.time()
        loaded = {i: load_file(os.path.join(cache_dir, f"{i}.safetensors"), device="cuda") for i in range(n)}
        sync()
        load_ms = (time.time() - t0) / n * 1e3
        exact = all(torch.equal(originals[i][k], loaded[i][k]) for i in range(n) for k in originals[i])
        z = loaded[0]["latent_mean"] + (0.5 * loaded[0]["latent_logvar"]).exp() * torch.randn_like(loaded[0]["latent_mean"])
        size_kb = sum(os.path.getsize(os.path.join(cache_dir, f)) for f in os.listdir(cache_dir)) / n / 1024
    finally:
        shutil.rmtree(cache_dir, ignore_errors=True)
    res = {"items": n, "encode_ms_per_item": round(encode_ms, 1), "load_ms_per_item": round(load_ms, 1),
           "cache_kb_per_item": round(size_kb), "encoder_vram_freed_mb": round(freed_mb),
           "bit_exact_reload": exact, "sampled_latent_shape": list(z.shape),
           "note": "toy encoders; the real gain is freeing the text encoder + VAE from VRAM during training"}
    return res if exact else ("FAIL", res)


# ---- 21 ----------------------------------------------------------------------------------------
@check("21. Tiled VAE encode/decode (overlapping tiles vs full image)")
def _():
    torch.manual_seed(0)
    up = lambda: torch.nn.Upsample(scale_factor=2, mode="nearest")  # noqa: E731
    dec = torch.nn.Sequential(torch.nn.Conv2d(4, 128, 3, 1, 1), torch.nn.SiLU(), torch.nn.Conv2d(128, 128, 3, 1, 1),
                              up(), torch.nn.Conv2d(128, 128, 3, 1, 1), torch.nn.SiLU(),
                              up(), torch.nn.Conv2d(128, 64, 3, 1, 1), torch.nn.SiLU(),
                              up(), torch.nn.Conv2d(64, 3, 3, 1, 1)).to(dev, BF16)
    enc = torch.nn.Sequential(torch.nn.Conv2d(3, 64, 3, 2, 1), torch.nn.SiLU(), torch.nn.Conv2d(64, 128, 3, 2, 1),
                              torch.nn.SiLU(), torch.nn.Conv2d(128, 128, 3, 2, 1), torch.nn.SiLU(),
                              torch.nn.Conv2d(128, 4, 3, 1, 1)).to(dev, BF16)

    def tiled(fn, x, tile, halo, scale):
        """Run fn on overlapping tiles; scale = output pixels per input pixel (8 for decode, 1/8 for encode)."""
        H, W = x.shape[-2:]
        out = None
        for y0 in range(0, H, tile):
            for x0 in range(0, W, tile):
                y1, x1 = min(y0 + tile, H), min(x0 + tile, W)
                ya, xa, yb, xb = max(y0 - halo, 0), max(x0 - halo, 0), min(y1 + halo, H), min(x1 + halo, W)
                o = fn(x[..., ya:yb, xa:xb])
                if out is None:
                    out = torch.empty(*x.shape[:-3], o.shape[-3], int(H * scale), int(W * scale),
                                      device=x.device, dtype=o.dtype)
                s = lambda v: int(v * scale)  # noqa: E731
                out[..., s(y0):s(y1), s(x0):s(x1)] = o[..., s(y0 - ya):s(y1 - ya), s(x0 - xa):s(x1 - xa)]
        return out

    def measure(fn):
        torch.cuda.empty_cache()
        base = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            t0 = time.time()
            y = fn()
            sync()
        return y, (torch.cuda.max_memory_allocated() - base) / 1e9, (time.time() - t0) * 1e3

    z = torch.randn(1, 4, 160, 160, device=dev, dtype=BF16)          # 1280x1280 image
    img = torch.randn(1, 3, 1280, 1280, device=dev, dtype=BF16)
    with torch.no_grad():                                            # warm-up so MIOpen setup is not timed
        dec(z[..., :40, :40])
        enc(img[..., :320, :320])
    res, problems = {}, []
    for name, full_fn, tile_fn in (
            ("decode 160x160 latent -> 1280px", lambda: dec(z), lambda: tiled(dec, z, 48, 4, 8)),
            ("encode 1280px -> 160x160 latent", lambda: enc(img), lambda: tiled(enc, img, 384, 32, 1 / 8))):
        y_full, m_full, t_full = measure(full_fn)
        y_tile, m_tile, t_tile = measure(tile_fn)
        err = rel_l2(y_tile, y_full)
        res[name] = {"full_peak_gb": round(m_full, 2), "tiled_peak_gb": round(m_tile, 2),
                     "full_ms": round(t_full, 1), "tiled_ms": round(t_tile, 1), "rel_err": round(err, 5)}
        if err > 1e-2:
            problems.append(f"{name}: tiled output differs (rel err {err:.4f})")
    res["note"] = ("conv-only toy VAE, so tiles with a halo wider than the receptive field are exact; real VAEs "
                   "(GroupNorm + mid-block attention) need overlap blending and are only approximately equal")
    return ("FAIL", {**res, "problems": problems}) if problems else res


# ---- 22 ----------------------------------------------------------------------------------------
@check("22. CPU offload / block swap (frozen weights in pinned RAM, streamed per layer)")
def _():
    class OffloadedLinear(torch.autograd.Function):
        """y = x @ W^T with W living in pinned CPU memory; W is re-fetched in backward instead of being saved."""
        @staticmethod
        def forward(ctx, x, w_cpu):
            ctx.w_cpu = w_cpu
            return x @ w_cpu.to(x.device, non_blocking=True).t()

        @staticmethod
        def backward(ctx, g):
            return g @ ctx.w_cpu.to(g.device, non_blocking=True), None

    torch.manual_seed(0)
    L, D, Hd, T, R = 8, 3072, 8192, 2048, 16
    w_cpu = [(torch.randn(Hd, D, dtype=BF16) * 0.02).pin_memory() for _ in range(L)]
    w2_cpu = [(torch.randn(D, Hd, dtype=BF16) * 0.02).pin_memory() for _ in range(L)]
    lora = [(torch.nn.Parameter(torch.randn(R, D, device=dev, dtype=BF16) * 0.01),
             torch.nn.Parameter(torch.randn(Hd, R, device=dev, dtype=BF16) * 0.01)) for _ in range(L)]
    x = torch.randn(T, D, device=dev, dtype=BF16)

    def forward(w1s, w2s, mm):
        h = x
        for w1, w2, (A, B) in zip(w1s, w2s, lora):
            u = mm(h, w1) + (h @ A.t()) @ B.t()
            h = h + mm(F.gelu(u), w2)
        return h

    def step(w1s, w2s, mm):
        for A, B in lora:
            A.grad = B.grad = None
        forward(w1s, w2s, mm).float().pow(2).mean().backward()

    def run(w1s, w2s, mm):
        torch.cuda.empty_cache()
        base = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        step(w1s, w2s, mm)
        sync()
        peak = (torch.cuda.max_memory_allocated() - base) / 1e9
        grads = [g.grad.clone() for pair in lora for g in pair]
        return peak, grads, bench(lambda: step(w1s, w2s, mm), iters=2, warmup=1, repeats=3, wall=True)["med"]

    off_peak, off_grads, off_ms = run(w_cpu, w2_cpu, lambda h, w: OffloadedLinear.apply(h, w))
    w_gpu, w2_gpu = [w.to(dev) for w in w_cpu], [w.to(dev) for w in w2_cpu]
    weights_gb = sum(w.numel() * 2 for w in w_gpu + w2_gpu) / 1e9
    res_peak, res_grads, res_ms = run(w_gpu, w2_gpu, lambda h, w: h @ w.t())
    res_peak += weights_gb                       # resident weights were allocated before the baseline was taken
    del w_gpu, w2_gpu
    err = max(rel_l2(a, b) for a, b in zip(off_grads, res_grads))

    # forward-only block swap with prefetch: copy layer i+1 on a side stream while layer i computes
    copy_stream = torch.cuda.Stream()

    def prefetch_forward():
        with torch.no_grad():
            with torch.cuda.stream(copy_stream):
                nxt = (w_cpu[0].to(dev, non_blocking=True), w2_cpu[0].to(dev, non_blocking=True))
            h = x
            for i in range(L):
                torch.cuda.current_stream().wait_stream(copy_stream)
                w1, w2 = nxt
                w1.record_stream(torch.cuda.current_stream())
                w2.record_stream(torch.cuda.current_stream())
                if i + 1 < L:
                    with torch.cuda.stream(copy_stream):
                        nxt = (w_cpu[i + 1].to(dev, non_blocking=True), w2_cpu[i + 1].to(dev, non_blocking=True))
                A, B = lora[i]
                h = h + F.gelu(h @ w1.t() + (h @ A.t()) @ B.t()) @ w2.t()
            return h

    def plain_offload_forward():
        with torch.no_grad():
            return forward(w_cpu, w2_cpu, lambda h, w: h @ w.to(dev, non_blocking=True).t())
    e_pf = rel_l2(prefetch_forward(), plain_offload_forward())
    t_pf = bench(prefetch_forward, iters=2, warmup=1, repeats=3, wall=True)["med"]
    t_sync = bench(plain_offload_forward, iters=2, warmup=1, repeats=3, wall=True)["med"]
    res = {"frozen_weights_gb": round(weights_gb, 2),
           "resident": f"{res_ms:.0f} ms/step, peak {res_peak:.2f} GB",
           "offloaded": f"{off_ms:.0f} ms/step, peak {off_peak:.2f} GB",
           "vram_saved_gb": round(res_peak - off_peak, 2), "lora_grad_rel_err": round(err, 5),
           "fwd_offload_no_prefetch_ms": round(t_sync, 1), "fwd_offload_prefetch_ms": round(t_pf, 1),
           "prefetch_speedup": f"{t_sync / t_pf:.2f}x", "prefetch_rel_err": round(e_pf, 5)}
    if err > 2e-2 or e_pf > 1e-3:
        return "FAIL", {**res, "problem": "offloaded results differ from resident"}
    return res


# ---- 23 ----------------------------------------------------------------------------------------
@check("23. Custom fused Triton LoRA kernel (up-proj + scale + residual add in one kernel)")
def _():
    try:
        import triton
        import triton.language as tl
    except ImportError:
        return "SKIP", 'triton not installed. Try:  pip install "triton-windows<3.9"'
    cache = os.environ.get("TRITON_CACHE_DIR") or os.path.join(os.path.expanduser("~"), ".triton", "cache")
    cache_before = len(os.listdir(cache)) if os.path.isdir(cache) else 0

    # Rank-r up-projection as r outer-product FMAs, not tl.dot: at K=16 the WMMA path spends its time converting
    # layouts through LDS (measured 1.2-2.9 ms vs 0.2 ms here). Short, wide tiles keep the y read/write coalesced.
    @triton.autotune(configs=[triton.Config({"BM": 4, "BN": 1024}, num_warps=4),
                              triton.Config({"BM": 8, "BN": 1024}, num_warps=4),
                              triton.Config({"BM": 8, "BN": 1024}, num_warps=8),
                              triton.Config({"BM": 8, "BN": 512}, num_warps=2)], key=["M", "N"],
                     restore_value=["y_ptr"])  # kernel is in place: undo each autotuning trial
    @triton.jit
    def lora_up_add(h_ptr, b_ptr, y_ptr, M, N, scale, shm, shr, sbn, sbr, sym, syn,
                    R: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr):
        """y[M,N] += scale * h[M,R] @ B[N,R]^T, in place."""
        rm = tl.program_id(0) * BM + tl.arange(0, BM)
        rn = tl.program_id(1) * BN + tl.arange(0, BN)
        mm, mn = rm < M, rn < N
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for r in tl.static_range(R):
            hc = tl.load(h_ptr + rm * shm + r * shr, mask=mm, other=0.0).to(tl.float32)
            br = tl.load(b_ptr + rn * sbn + r * sbr, mask=mn, other=0.0).to(tl.float32)
            acc += hc[:, None] * br[None, :]
        mask = mm[:, None] & mn[None, :]
        y_ptrs = y_ptr + rm[:, None] * sym + rn[None, :] * syn
        y = tl.load(y_ptrs, mask=mask, other=0.0).to(tl.float32) + scale * acc
        tl.store(y_ptrs, y.to(y_ptr.dtype.element_ty), mask=mask)

    def up_add_(h, B, y, scale):
        M, N = y.shape
        grid = lambda meta: (triton.cdiv(M, meta["BM"]), triton.cdiv(N, meta["BN"]))  # noqa: E731
        lora_up_add[grid](h, B, y, M, N, scale, h.stride(0), h.stride(1), B.stride(0), B.stride(1),
                          y.stride(0), y.stride(1), R=h.shape[1])
        return y

    class FusedLoRA(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, W, A, B, scale):
            h = x @ A.t()
            y = up_add_(h, B, x @ W.t(), scale)
            ctx.save_for_backward(x, W, A, B, h)
            ctx.scale = scale
            return y

        @staticmethod
        def backward(ctx, g):
            x, W, A, B, h = ctx.saved_tensors
            dh = ctx.scale * (g @ B)
            return g @ W + dh @ A, None, dh.t() @ x, ctx.scale * (g.t() @ h), None

    torch.manual_seed(0)
    M, K, N, R, scale = 8192, 4096, 4096, 16, 1.0
    x = torch.randn(M, K, device=dev, dtype=BF16, requires_grad=True)
    W = torch.randn(N, K, device=dev, dtype=BF16) * 0.02
    A = (torch.randn(R, K, device=dev, dtype=BF16) * 0.01).requires_grad_(True)
    B = (torch.randn(N, R, device=dev, dtype=BF16) * 0.01).requires_grad_(True)
    g = torch.randn(M, N, device=dev, dtype=BF16)

    def ref_fn():
        return x @ W.t() + scale * ((x @ A.t()) @ B.t())

    def grads(fn):
        x.grad = A.grad = B.grad = None
        out = fn()
        out.backward(g)
        return out.detach(), x.grad.clone(), A.grad.clone(), B.grad.clone()
    fused = grads(lambda: FusedLoRA.apply(x, W, A, B, scale))
    ref = grads(ref_fn)
    errs = {n: round(rel_l2(a, b), 5) for n, a, b in zip(("out", "dx", "dA", "dB"), fused, ref)}

    with torch.no_grad():
        h, y0 = x @ A.t(), x @ W.t()
        t_kernel = bench(lambda: up_add_(h, B, y0.clone(), scale), iters=20)["med"]
        t_addmm = bench(lambda: torch.addmm(y0, h, B.t(), alpha=scale), iters=20)["med"]
        t_naive = bench(lambda: y0 + scale * (h @ B.t()), iters=20)["med"]
        t_clone = bench(lambda: y0.clone(), iters=20)["med"]

    def train(fn):
        def s():
            x.grad = A.grad = B.grad = None
            fn().backward(g)
        return bench(s, iters=5, warmup=2, repeats=3)["med"]
    res = {"rel_err_vs_reference": errs,
           "epilogue_ms": {"triton_fused(in-place)": round(t_kernel - t_clone, 3), "torch.addmm": round(t_addmm, 3),
                           "naive y+s*(h@B^T)": round(t_naive, 3)},
           "layer_fwd+bwd_ms": {"triton_fused": round(train(lambda: FusedLoRA.apply(x, W, A, B, scale)), 2),
                                "torch_reference": round(train(ref_fn), 2)},
           "best_config": str(lora_up_add.best_config),
           "triton_cache_dir": cache,
           "triton_cache_entries": f"{cache_before} -> {len(os.listdir(cache)) if os.path.isdir(cache) else 0}"}
    if max(errs.values()) > 2e-2:
        return "FAIL", {**res, "problem": "fused kernel output/grads differ from the torch reference"}
    if t_kernel - t_clone > t_addmm:
        return "WARN", {**res, "hint": "fused kernel is correct but slower than torch.addmm; use addmm"}
    return res


# ---- 24 ----------------------------------------------------------------------------------------
@check("24. MSVC build tools (Inductor CPU codegen needs cl.exe)")
def _():
    cl = shutil.which("cl")
    out = {"vcvars_loaded_from": MSVC_STATUS, "cl.exe": cl}
    if not cl:
        return "WARN", {**out, "hint": "cl.exe not on PATH. Run without --baseline (the script imports vcvars64.bat) "
                                      "or start from an 'x64 Native Tools' prompt. Needs VS Build Tools with the "
                                      "'Desktop development with C++' workload."}
    banner = subprocess.run(["cl"], capture_output=True, text=True).stderr.splitlines()
    out["cl_version"] = banner[0].strip() if banner else "?"
    f = torch.compile(lambda t: (t.sin() * 2 + 1).relu().sum())
    t = torch.randn(10000)
    ok = torch.allclose(f(t), (t.sin() * 2 + 1).relu().sum(), rtol=1e-4)
    out["inductor_cpu_compile"] = "works" if ok else "wrong result"
    return out if ok else ("FAIL", out)


# ---- 25 ----------------------------------------------------------------------------------------
@check("25. MIOpen: gfx1101 system db (fdb/kdb) check, user-db warm-up, CK library load cost")
def _():
    import glob
    from importlib.util import find_spec
    spec = find_spec("_rocm_sdk_libraries")
    lib_dir = os.path.join(os.path.dirname(spec.origin), "bin") if spec else None
    arch = getattr(torch.cuda.get_device_properties(0), "gcnArchName", "gfx1101").split(":")[0]
    system_db = sorted(os.path.basename(p) for p in glob.glob(os.path.join(lib_dir or "", f"{arch}*")))
    system_db = [p for p in system_db if not p.endswith(".dll")]
    probe = r'''
import json, os, time, torch
t0 = time.time()
c = torch.nn.Conv2d(8, 8, 3, padding=1).cuda(); c(torch.randn(1, 8, 16, 16, device="cuda")); torch.cuda.synchronize()
out = {"miopen_init_ms": round((time.time() - t0) * 1e3)}
for name, C, H in (("vae_mid 512ch@128", 512, 128), ("vae_up 128ch@1024", 128, 1024)):
    conv = torch.nn.Conv2d(C, C, 3, padding=1).to("cuda", torch.bfloat16)
    x = torch.randn(1, C, H, H, device="cuda", dtype=torch.bfloat16)
    t0 = time.time(); conv(x); torch.cuda.synchronize(); first = (time.time() - t0) * 1e3
    for _ in range(3): conv(x)
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(10): conv(x)
    e.record(); e.synchronize()
    out[name] = {"first_call_ms": round(first), "steady_ms": round(s.elapsed_time(e) / 10, 2)}
print("RESULT " + json.dumps(out))
'''
    user_db = os.environ.get("MIOPEN_USER_DB_PATH") or os.path.join(os.path.expanduser("~"), ".miopen", "db")
    cold_db = os.path.join(HERE, ".miopen_db_cold")
    shutil.rmtree(cold_db, ignore_errors=True)
    os.makedirs(cold_db)
    runs = {}
    try:
        for label, mode, db in (("FAST, empty user db", "FAST", cold_db),
                                ("NORMAL warm-up (benchmarks, writes user db)", "NORMAL", user_db),
                                ("FAST, warmed user db", "FAST", user_db)):
            rc, res, err = subproc(probe, {"MIOPEN_FIND_MODE": mode, "MIOPEN_USER_DB_PATH": db})
            runs[label] = res if res else f"crashed (rc {rc}): {err}"
    finally:
        shutil.rmtree(cold_db, ignore_errors=True)
    bg = r'''
import json, os, threading, time, torch
torch.zeros(1, device="cuda")
def warm():
    c = torch.nn.Conv2d(8, 8, 3, padding=1).cuda(); c(torch.randn(1, 8, 16, 16, device="cuda")); torch.cuda.synchronize()
t0 = time.time(); th = threading.Thread(target=warm); th.start()
n, a = 0, torch.randn(2048, 2048, device=os.environ["MAIN_DEVICE"])
while th.is_alive():
    a = (a @ a).tanh(); n += 1
    if a.is_cuda: torch.cuda.synchronize()
th.join()
print("RESULT " + json.dumps({"warmup_s": round(time.time() - t0, 1), "main_thread_matmuls_meanwhile": n}))
'''
    bg_res = {}
    for label, device, tries in (("main thread on CPU (loading weights, tokenizing)", "cpu", 3),
                                 ("main thread using the GPU", "cuda", 1)):
        outcomes = []
        for _ in range(tries):
            rc, res, _ = subproc(bg, {"MAIN_DEVICE": device})
            outcomes.append(res if res else f"crashed (rc {rc:#x})")
        bg_res[label] = outcomes
    out = {"miopen_lib_dir": lib_dir, f"system_db_files_for_{arch}": system_db or "none shipped",
           "user_db": {"path": user_db, "files": sorted(os.listdir(user_db)) if os.path.isdir(user_db) else []},
           "runs": runs, "background_warmup": bg_res}
    out["meaning"] = (
        "No gfx1101 system find-db ships with ROCm 10.1 (hence the 'File is unreadable ... fdb.txt' warning - "
        "harmless). FIND_MODE=FAST alone never writes the user db; one NORMAL pass benchmarks and stores the best "
        "solvers, after which FAST reuses them. The ~9 s first-conv delay is the static init of the 367 MB "
        "MIOpenCKGroupedConv_gfx1101.dll, paid once per process. It can be hidden by running a tiny conv in a "
        "background thread while the main thread does CPU-only startup work, but the main thread must not touch "
        "the GPU until that thread is joined (concurrent GPU work during MIOpen init crashes with 0xC0000005).")
    if any(not isinstance(r, dict) for r in bg_res["main thread on CPU (loading weights, tokenizing)"]):
        return "WARN", {**out, "hint": "background MIOpen warm-up crashed even with a CPU-only main thread; "
                                      "do the warm-up synchronously instead"}
    if any(not isinstance(v, dict) for v in runs.values()):
        return "FAIL", out
    return "INFO" if not system_db else "PASS", out


# ---- 26 ----------------------------------------------------------------------------------------
@check("26. TunableOp CSV reuse in a fresh process (no re-tuning)")
def _():
    csv_path = os.path.join(HERE, "tunableop_results.csv")
    if not os.path.exists(csv_path):
        return "SKIP", "no tunableop_results.csv yet - run test 5 first"
    before = open(csv_path, "rb").read()
    code = r'''
import json, os, torch
import torch.cuda.tunable as t
csv = os.environ["CSV"]
if os.environ.get("USE_API") == "1":
    t.set_filename(csv, insert_device_ordinal=False)
    t.enable(True); t.tuning_enable(False)
    loaded = t.read_file(csv)
else:
    loaded = None
res = {}
for M, N, K in ((8192, 16, 4096), (8192, 4096, 16), (8192, 4096, 4096), (4096, 4096, 4096)):
    a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16); b = torch.randn(K, N, device="cuda", dtype=torch.bfloat16)
    for _ in range(3): a @ b
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(20): a @ b
    e.record(); e.synchronize(); res[f"{M}x{N}x{K}"] = round(s.elapsed_time(e) / 20, 3)
print("RESULT " + json.dumps({"filename": t.get_filename(), "read_file": loaded, "results_in_memory": len(t.get_results()),
                              "ms": res}))
'''
    runs = {}
    _, runs["tunableop_off"], _ = subproc(code, {"CSV": csv_path}, drop_env=("PYTORCH_TUNABLEOP_ENABLED",))
    _, runs["env_vars"], _ = subproc(code, {"CSV": csv_path, "PYTORCH_TUNABLEOP_ENABLED": "1",
                                            "PYTORCH_TUNABLEOP_TUNING": "0", "PYTORCH_TUNABLEOP_FILENAME": csv_path})
    _, runs["python_api"], _ = subproc(code, {"CSV": csv_path, "USE_API": "1"})
    rows = sum(1 for l in before.decode(errors="ignore").splitlines() if l and not l.startswith("Validator"))
    out = {"csv": csv_path, "tuned_rows_in_csv": rows, "runs": runs,
           "csv_unchanged": open(csv_path, "rb").read() == before}
    api = runs.get("python_api") or {}
    if api.get("results_in_memory", 0) < rows:
        return "FAIL", {**out, "problem": "Python API did not load the tuned results"}
    env = runs.get("env_vars") or {}
    if env.get("results_in_memory", 0) < rows:
        out["hint"] = (f"PYTORCH_TUNABLEOP_FILENAME inserts the device ordinal (looks for {env.get('filename')}), so "
                       "the env-var route silently loads nothing. In the trainer use torch.cuda.tunable."
                       "set_filename(path, insert_device_ordinal=False) + read_file(path) + tuning_enable(False).")
    return out


# ---- 27 ----------------------------------------------------------------------------------------
@check("27. VRAM logging (mem_get_info) and CUDA-event timing vs wall clock")
def _():
    def snap():
        free, total = torch.cuda.mem_get_info()
        return {"used_gb": (total - free) / 1e9, "allocated_gb": torch.cuda.memory_allocated() / 1e9,
                "reserved_gb": torch.cuda.memory_reserved() / 1e9, "peak_gb": torch.cuda.max_memory_allocated() / 1e9}
    torch.cuda.empty_cache()
    s0 = snap()
    blob = torch.empty(1 << 30, dtype=torch.uint8, device=dev)
    s1 = snap()
    del blob
    s2 = snap()
    torch.cuda.empty_cache()
    s3 = snap()
    d_used, d_alloc = s1["used_gb"] - s0["used_gb"], s1["allocated_gb"] - s0["allocated_gb"]
    a = torch.randn(4096, 4096, device=dev, dtype=BF16)
    for _ in range(3):
        a @ a
    sync()
    t0 = time.perf_counter()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(50):
        a @ a
    e.record()
    e.synchronize()
    wall_ms, event_ms = (time.perf_counter() - t0) * 1e3, s.elapsed_time(e)
    out = {"alloc_1GiB": {"mem_get_info_delta_gb": round(d_used, 3), "memory_allocated_delta_gb": round(d_alloc, 3)},
           "after_del": {"allocated_gb": round(s2["allocated_gb"], 3), "reserved_gb": round(s2["reserved_gb"], 3)},
           "after_empty_cache_reserved_gb": round(s3["reserved_gb"], 3),
           "50_gemms": {"cuda_events_ms": round(event_ms, 2), "wall_clock_ms": round(wall_ms, 2)}}
    problems = []
    if abs(d_alloc - 1.0737) > 0.01 or abs(d_used - 1.0737) > 0.15:
        problems.append("VRAM deltas do not match the 1 GiB allocation")
    if not 0.8 * wall_ms <= event_ms <= wall_ms * 1.02:
        problems.append("event time disagrees with wall clock")
    if problems:
        return "FAIL", {**out, "problems": problems}

    host = torch.empty(256 * 1024 * 1024, dtype=BF16).pin_memory()    # 512 MB pinned -> GPU copy
    host.to(dev, non_blocking=True)
    sync()
    t0 = time.perf_counter()
    s.record()
    host.to(dev, non_blocking=True)
    e.record()
    e.synchronize()
    copy_wall, copy_event = (time.perf_counter() - t0) * 1e3, s.elapsed_time(e)
    out["512MB_pinned_H2D_copy"] = {"cuda_events_ms": round(copy_event, 2), "wall_clock_ms": round(copy_wall, 2)}
    if copy_event < 0.5 * copy_wall:
        return "WARN", {**out, "hint": "CUDA events do not see host->device copy time on this HIP build. Use events "
                                      "for compute regions only; time anything with CPU<->GPU transfers (offload, "
                                      "data loading) with time.perf_counter() around torch.cuda.synchronize()."}
    return out


# ---- 28 ----------------------------------------------------------------------------------------
@check("28. AMD SMI (GPU telemetry) - with Windows performance-counter fallback")
def _():
    out = {"amd-smi_cli": shutil.which("amd-smi")}
    try:
        import amdsmi
        amdsmi.amdsmi_init()
        h = amdsmi.amdsmi_get_processor_handles()[0]
        out["amdsmi_python"] = {"vram": amdsmi.amdsmi_get_gpu_vram_usage(h)}
        return out
    except Exception as e:  # noqa: BLE001
        out["amdsmi_python"] = f"unavailable ({type(e).__name__})"

    # fresh process: in this one, earlier tests left memory in HIP's pool, so a new allocation never reaches the driver
    code = r'''
import json, os, subprocess, torch
def counter(path):
    r = subprocess.run(["typeperf", path, "-sc", "1"], capture_output=True, text=True, timeout=30)
    lines = [l for l in r.stdout.splitlines() if l.startswith('"')]
    vals = [float(v.strip('"')) for v in lines[-1].split(",")[1:] if v.strip('"')] if len(lines) >= 2 else []
    return max(vals) if vals else None
proc = rf"\GPU Process Memory(pid_{os.getpid()}_*)\Dedicated Usage"
torch.zeros(1, device="cuda"); torch.cuda.synchronize()
p0 = counter(proc)
blob = torch.empty(1 << 30, dtype=torch.uint8, device="cuda"); blob.fill_(1); torch.cuda.synchronize()
print("RESULT " + json.dumps({"p0": p0, "p1": counter(proc), "adapter": counter(r"\GPU Adapter Memory(*)\Dedicated Usage"),
                              "query": f'typeperf "{proc}" -sc 1'}))
'''
    rc, res, err = subproc(code, timeout=120)
    if not res or res["p0"] is None or res["p1"] is None:
        return "WARN", {**out, "hint": "no amd-smi and Windows GPU counters unreadable; use torch.cuda.mem_get_info()",
                        "error": err}
    delta = (res["p1"] - res["p0"]) / 2 ** 30
    out["windows_counter_fallback"] = {"process_dedicated_gb": round(res["p1"] / 1e9, 2),
                                       "adapter_dedicated_gb": round(res["adapter"] / 1e9, 2),
                                       "delta_after_1GiB_alloc_gib": round(delta, 3), "query": res["query"]}
    out["meaning"] = ("AMD SMI is not shipped for Windows in ROCm 10.1. For VRAM telemetry use torch.cuda.mem_get_info "
                      "(this process) or the Windows 'GPU Process Memory' / 'GPU Adapter Memory' counters (all "
                      "processes). Temperature/clock/power need Adrenalin's overlay.")
    # the counter is per-process driver usage, so it also sees allocator rounding and kernel code objects
    return ("INFO", out) if 0.9 < delta < 1.5 else ("WARN", out)


# ---- 29 ----------------------------------------------------------------------------------------
@check("29. rocprofv3 / ROCprofiler-SDK - with CUDA-event region profiling fallback")
def _():
    from importlib.util import find_spec
    spec = find_spec("_rocm_sdk_devel")
    sdk_bin = os.path.join(os.path.dirname(spec.origin), "bin") if spec else ""
    found = shutil.which("rocprofv3") or next((os.path.join(sdk_bin, f) for f in ("rocprofv3.exe", "rocprofv3")
                                                if os.path.isfile(os.path.join(sdk_bin, f))), None)
    out = {"rocprofv3": found}
    if found:
        r = subprocess.run([found, "--version"], capture_output=True, text=True, timeout=60)
        return {**out, "version": (r.stdout or r.stderr).strip()[:200]}

    class RegionTimer:
        """Per-region GPU time with CUDA events; no host sync until report()."""
        def __init__(self):
            self.events = []

        def region(self, name):
            timer = self

            class _Ctx:
                def __enter__(self):
                    self.s = torch.cuda.Event(enable_timing=True)
                    self.s.record()

                def __exit__(self, *exc):
                    e = torch.cuda.Event(enable_timing=True)
                    e.record()
                    timer.events.append((name, self.s, e))
            return _Ctx()

        def report(self):
            torch.cuda.synchronize()
            totals = {}
            for name, s, e in self.events:
                totals[name] = totals.get(name, 0.0) + s.elapsed_time(e)
            return {k: round(v, 3) for k, v in totals.items()}

    lin = torch.nn.Linear(4096, 4096).to(dev, BF16)
    x = torch.randn(4096, 4096, device=dev, dtype=BF16, requires_grad=True)
    q = torch.randn(1, 16, 2048, 64, device=dev, dtype=BF16)
    lin(x).float().sum().backward()                     # warm-up: first-call library init is not a region cost
    F.scaled_dot_product_attention(q, q, q)
    timer = RegionTimer()
    for _ in range(5):
        with timer.region("linear_fwd"):
            y = lin(x)
        with timer.region("linear_bwd"):
            y.float().sum().backward()
        with timer.region("sdpa_fwd"):
            F.scaled_dot_product_attention(q, q, q)
    out["region_gpu_ms_total_5_iters"] = timer.report()
    out["meaning"] = ("rocprofv3 / ROCprofiler-SDK is not part of the Windows ROCm 10.1 SDK, and torch.profiler "
                      "(Kineto) records no GPU kernels here (test 4). Use CUDA-event region timers like this one "
                      "for per-phase GPU time.")
    return "INFO", out


# ---- 30 ----------------------------------------------------------------------------------------
@check("30. Adrenalin driver version check (expect 26.10.41.05)")
def _():
    import winreg
    expected = "26.10.41.05"
    cls = r"SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}"
    found = []
    with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, cls) as root:
        for i in range(winreg.QueryInfoKey(root)[0]):
            sub = winreg.EnumKey(root, i)
            if not sub.isdigit():
                continue
            try:
                with winreg.OpenKey(root, sub) as k:
                    vals = {}
                    for name in ("DriverDesc", "DriverVersion", "RadeonSoftwareVersion", "ReleaseVersion"):
                        try:
                            vals[name] = winreg.QueryValueEx(k, name)[0]
                        except OSError:
                            pass
            except OSError:
                continue
            if "Radeon" in vals.get("DriverDesc", "") or "AMD" in vals.get("DriverDesc", ""):
                found.append(vals)
    gpu = next((v for v in found if "RadeonSoftwareVersion" in v), None)
    if not gpu:
        return "WARN", {"adapters": found, "hint": "could not read the Adrenalin version from the registry"}
    ver = lambda s: tuple(int(p) for p in s.split(".") if p.isdigit())  # noqa: E731
    out = {**gpu, "expected": expected}
    if ver(gpu["RadeonSoftwareVersion"]) < ver(expected):
        return "WARN", {**out, "hint": f"driver is older than {expected}; update Adrenalin (ROCm 10.1 needs it)"}
    return out


# ---- 31 ----------------------------------------------------------------------------------------
@check("31. LoRA merge into base weights and unmerge (in place, bf16)")
def _():
    torch.manual_seed(0)
    base = torch.nn.Linear(4096, 4096, bias=False).to(dev, BF16)
    A = torch.randn(16, 4096, device=dev) * 0.01
    B = torch.randn(4096, 16, device=dev) * 0.02
    scale = 1.0
    x = torch.randn(2048, 4096, device=dev, dtype=BF16)
    w_orig = base.weight.detach().clone()
    with torch.no_grad():
        y_lora = (base(x).float() + scale * (x.float() @ A.t()) @ B.t())
        delta = scale * (B @ A)                                          # fp32
        base.weight.copy_((w_orig.float() + delta).to(BF16))             # merge: add in fp32, round once
        e_merge = rel_l2(base(x), y_lora)
        naive = w_orig.clone()
        naive.add_(delta.to(BF16))                                       # bf16 add: rounds delta and sum
        e_naive = rel_l2(x @ naive.t(), y_lora)
        unmerged = (base.weight.float() - delta).to(BF16)               # unmerge by subtraction
        drift = (unmerged.float() - w_orig.float()).abs().max().item()
        flips = (unmerged != w_orig).float().mean().item()
        base.weight.copy_(w_orig)                                        # exact unmerge from a backup copy
        exact = torch.equal(base.weight, w_orig)
    out = {"merged_vs_unmerged_rel_err": round(e_merge, 5), "bf16_inplace_add_rel_err": round(e_naive, 5),
           "unmerge_by_subtraction": {"max_abs_drift": drift, "weights_changed": f"{flips:.1%}"},
           "unmerge_from_backup_exact": exact,
           "recommendation": "merge as W = bf16(fp32(W) + scale*B@A); to unmerge restore a saved copy of W"}
    if e_merge > 0.02 or not exact:
        return "FAIL", out
    return out


# ---- 32 ----------------------------------------------------------------------------------------
@check("32. Env-flag A/B: TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL, FLASH_ATTENTION_TRITON_AMD_ENABLE")
def _():
    sdpa = r'''
import json, warnings, torch, torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
q = torch.randn(1, 16, 1024, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
out = {}
for name, be in (("flash", SDPBackend.FLASH_ATTENTION), ("efficient", SDPBackend.EFFICIENT_ATTENTION)):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with sdpa_kernel(be):
                F.scaled_dot_product_attention(q, q, q).sum().backward()
        torch.cuda.synchronize(); out[name] = "works"
    except Exception as e:
        out[name] = "unavailable: " + str(e).splitlines()[0][:100]
print("RESULT " + json.dumps(out))
'''
    fa = r'''
import json, torch
try:
    from flash_attn import flash_attn_func
    q = torch.randn(1, 1024, 16, 64, device="cuda", dtype=torch.bfloat16)
    flash_attn_func(q, q, q); torch.cuda.synchronize(); r = "works"
except Exception as e:
    r = f"fails: {type(e).__name__}: " + str(e).splitlines()[0][:120]
print("RESULT " + json.dumps(r))
'''
    out = {"TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL": {}, "FLASH_ATTENTION_TRITON_AMD_ENABLE": {}}
    for label, kw in (("=1", {"set_env": {"TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL": "1"}}),
                      ("unset", {"drop_env": ("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL",)})):
        rc, res, err = subproc(sdpa, **kw)
        out["TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"][label] = res or f"crashed (rc {rc}): {err}"
    if pkg_version("flash_attn") is None:
        out["FLASH_ATTENTION_TRITON_AMD_ENABLE"] = "flash_attn not installed (see test 11)"
    else:
        for label, kw in (("=TRUE", {"set_env": {"FLASH_ATTENTION_TRITON_AMD_ENABLE": "TRUE"}}),
                          ("unset", {"drop_env": ("FLASH_ATTENTION_TRITON_AMD_ENABLE",)})):
            rc, res, err = subproc(fa, **kw)
            out["FLASH_ATTENTION_TRITON_AMD_ENABLE"][label] = res or f"crashed (rc {rc}): {err}"
    aot = out["TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"]
    if not isinstance(aot.get("=1"), dict) or aot["=1"].get("flash") != "works":
        return "FAIL", {**out, "problem": "flash SDPA does not work even with the experimental flag"}
    return out


# ---- 33 ----------------------------------------------------------------------------------------
@check("33. bitsandbytes BNB_ROCM_VERSION override (which ROCm builds of the native lib work)")
def _():
    import glob
    import re
    from importlib.util import find_spec
    spec = find_spec("bitsandbytes")
    if spec is None:
        return "SKIP", "bitsandbytes not installed"
    pkg = os.path.dirname(spec.origin)
    versions = sorted(re.search(r"rocm(\d+)", p).group(1) for p in glob.glob(os.path.join(pkg, "libbitsandbytes_rocm*")))
    code = r'''
import json, torch, bitsandbytes as bnb
from bitsandbytes.cextension import lib
l = bnb.nn.Linear4bit(512, 512, bias=False, compute_dtype=torch.bfloat16, quant_type="nf4").cuda()
x = torch.randn(8, 512, device="cuda", dtype=torch.bfloat16, requires_grad=True)
l(x).float().sum().backward()
p = torch.nn.Parameter(torch.randn(512, 512, device="cuda")); p.grad = torch.randn_like(p)
before = p.detach().clone(); bnb.optim.AdamW8bit([p], lr=1e-3).step(); torch.cuda.synchronize()
print("RESULT " + json.dumps({"lib": getattr(getattr(lib, "_lib", None), "_name", None) or str(lib), "nf4_bwd_finite": bool(torch.isfinite(x.grad).all()),
                              "adamw8bit_updates": not torch.equal(before, p.detach())}))
'''
    out = {"torch_hip": torch.version.hip, "shipped_rocm_builds": versions,
           "BNB_ROCM_VERSION_in_this_run": os.environ.get("BNB_ROCM_VERSION")}
    results = {}
    for v in versions + ["unset"]:
        kw = {"drop_env": ("BNB_ROCM_VERSION",)} if v == "unset" else {"set_env": {"BNB_ROCM_VERSION": v}}
        rc, res, err = subproc(code, **kw, timeout=300)
        results[v] = ({"works": res["nf4_bwd_finite"] and res["adamw8bit_updates"],
                       "lib": os.path.basename(str(res["lib"]))} if res else f"failed (rc {rc}): {err[-160:]}")
    out["results"] = results
    working = [v for v in versions if isinstance(results[v], dict) and results[v]["works"]]
    if not working:
        return "FAIL", out
    chosen = os.environ.get("BNB_ROCM_VERSION")
    if chosen and chosen not in working:
        return "WARN", {**out, "hint": f"BNB_ROCM_VERSION={chosen} does not work; use one of {working}"}
    return out


# ---- summary -----------------------------------------------------------------------------------
print("\n\n########## SUMMARY ##########")
for k, v in RESULTS.items():
    print(f"[{v['status']:4}] {k}   ({v['seconds']}s)")
hints = {k: v["info"]["hint"] for k, v in RESULTS.items()
         if isinstance(v["info"], dict) and "hint" in v["info"]}
if hints:
    print("\nHINTS")
    for k, h in hints.items():
        print(f" - {k}: {h}")
counts = {s: sum(1 for v in RESULTS.values() if v["status"] == s) for s in ("PASS", "WARN", "FAIL", "SKIP", "INFO")}
print("\nTotals:", counts)
report = os.path.join(HERE, "amd_report_v2_baseline.json" if BASELINE else "amd_report_v2.json")
with open(report, "w", encoding="utf-8") as f:
    json.dump(RESULTS, f, indent=2, default=str)
print(f"Full details saved to {report}. Send me that file.")
sys.exit(1 if counts["FAIL"] else 0)