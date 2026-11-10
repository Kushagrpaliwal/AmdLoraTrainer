"""AMD optimization benchmark: measure what actually speeds up training on YOUR GPU.

Every variant runs in its OWN process, because AMD/PyTorch environment variables must be
set before torch is imported. A variant is a name made of features joined with "+",
for example  "flash"  or  "flash+hipblaslt".  "plain" is the reference everything is
compared against.

Usage (from the project root):
    python scripts/bench_amd.py --list
    python scripts/bench_amd.py --variants plain,eager,math,efficient,flash,hipblaslt,expandable
    python scripts/bench_amd.py --variants tunableop,compile,fa_triton
    python scripts/bench_amd.py --variants flash+hipblaslt
"""
from __future__ import annotations
import argparse
import contextlib
import csv
import json
import os
import statistics
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

REFERENCE = "plain"
VOCAB = 32000
WARMUP_STEPS = 3
DEFAULT_VARIANTS = ["plain", "eager", "math", "efficient", "flash", "hipblaslt",
                    "expandable", "tunableop", "fa_triton", "compile"]

# Each feature changes environment variables and/or how the model is built and run.
FEATURES: dict[str, dict] = {
    "plain": {},                                                    # reference
    "eager": {"attn": "eager"},                                     # plain PyTorch attention
    "math": {"sdpa": "math"},                                       # SDPA, math backend
    "efficient": {"sdpa": "efficient"},                             # SDPA, memory-efficient backend
    "flash": {"sdpa": "flash"},                                     # SDPA, flash backend (AOTriton)
    "hipblaslt": {"env": {"TORCH_BLAS_PREFER_HIPBLASLT": "1"}},
    "tunableop": {"env": {"PYTORCH_TUNABLEOP_ENABLED": "1", "PYTORCH_TUNABLEOP_TUNING": "0"}},
    "tunableop_tune": {"env": {"PYTORCH_TUNABLEOP_ENABLED": "1", "PYTORCH_TUNABLEOP_TUNING": "1"},
                       "tuning_run": True},
    "expandable": {"env": {"PYTORCH_HIP_ALLOC_CONF": "expandable_segments:True"}},
    "fa_triton": {"attn": "flash_attention_2", "env": {"FLASH_ATTENTION_TRITON_AMD_ENABLE": "TRUE"}},
    "compile": {"compile": True},
}


# ---------------------------------------------------------------- pure helpers (no torch)

def base_env(cache_dir: Path) -> dict[str, str]:
    """Settings every variant starts from. Variants then override single values."""
    cache_dir = Path(cache_dir).resolve()
    return {
        "TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL": "1",
        "TORCH_BLAS_PREFER_HIPBLASLT": "0",
        "PYTORCH_TUNABLEOP_ENABLED": "0",
        "PYTORCH_TUNABLEOP_TUNING": "0",
        "PYTORCH_TUNABLEOP_FILENAME": str(cache_dir / "tunableop_results.csv"),
        "PYTORCH_HIP_ALLOC_CONF": "garbage_collection_threshold:0.8,max_split_size_mb:512",
        "MIOPEN_FIND_MODE": "FAST",
        "MIOPEN_USER_DB_PATH": str(cache_dir / "miopen"),
        "TRITON_CACHE_DIR": str(cache_dir / "triton"),
    }


def resolve_variant(name: str) -> tuple[dict[str, str], dict]:
    """Turn 'flash+hipblaslt' into (environment overrides, run options)."""
    env: dict[str, str] = {}
    options: dict = {}
    for feature in name.split("+"):
        if feature not in FEATURES:
            raise ValueError(f"Unknown feature '{feature}'. Known: {', '.join(FEATURES)}")
        spec = FEATURES[feature]
        env.update(spec.get("env", {}))
        options.update({k: v for k, v in spec.items() if k != "env"})
    return env, options


def order_variants(requested: list[str]) -> list[str]:
    """Reference first, and a TunableOp tuning run before any variant that reuses its results."""
    ordered = [REFERENCE]
    for name in requested:
        if name in ordered:
            continue
        if "tunableop" in name.split("+") and "tunableop_tune" not in ordered:
            ordered.append("tunableop_tune")
        ordered.append(name)
    return ordered


def judge(variant: str, quality: dict | None) -> str:
    if variant == REFERENCE:
        return "reference"
    if not quality:
        return "n/a"
    ok = quality["logits_rel_err"] < 0.03 and quality["loss0_rel_diff"] < 0.01
    return "OK" if ok else "CHECK"


def summarize(results: list[dict]) -> list[dict]:
    reference = next((r for r in results if r["variant"] == REFERENCE and r["status"] == "ok"), None)
    reference_ms = reference["step_ms_median"] if reference else None
    rows = []
    for r in results:
        row = {"variant": r["variant"], "status": r["status"], "note": r.get("error") or ""}
        if r["status"] == "ok":
            ms = r["step_ms_median"]
            quality = r.get("quality") or {}
            row.update(
                step_ms=ms, tokens_per_s=r["tokens_per_s"], peak_vram_gb=r["peak_vram_gb"],
                warmup_s=r["warmup_s"], speedup=(reference_ms / ms) if reference_ms else None,
                logits_rel_err=quality.get("logits_rel_err"),
                loss0_rel_diff=quality.get("loss0_rel_diff"),
                quality=judge(r["variant"], quality))
            if r.get("tuning_run"):
                row["note"] = "tuning run: timings not comparable"
        rows.append(row)
    return rows


def winners(rows: list[dict], margin: float = 0.03) -> list[dict]:
    """Variants that pass the quality check and beat the reference by more than `margin`."""
    good = [r for r in rows
            if r["status"] == "ok" and r["variant"] != REFERENCE and r.get("speedup")
            and r["quality"] == "OK" and "tuning run" not in r["note"]
            and r["speedup"] >= 1 + margin]
    return sorted(good, key=lambda r: -r["speedup"])


def format_table(rows: list[dict]) -> str:
    header = (f"{'variant':<20}{'status':<8}{'step ms':>9}{'tok/s':>9}{'speedup':>9}"
              f"{'peak GB':>9}{'warmup s':>10}  {'quality':<10}note")
    lines = [header, "-" * len(header)]
    for r in rows:
        if r["status"] != "ok":
            lines.append(f"{r['variant']:<20}{'FAILED':<8}{'':>46}  {'':<10}{r['note'][:70]}")
            continue
        speed = f"{r['speedup']:.2f}x" if r.get("speedup") else "-"
        lines.append(f"{r['variant']:<20}{'ok':<8}{r['step_ms']:>9.0f}{r['tokens_per_s']:>9,.0f}"
                     f"{speed:>9}{r['peak_vram_gb']:>9.2f}{r['warmup_s']:>10.1f}  "
                     f"{r['quality']:<10}{r['note'][:60]}")
    return "\n".join(lines)


# ---------------------------------------------------------------- worker (one variant, one process)

def measure_training(options: dict, args, out_dir: Path, name: str) -> dict:
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM
    from torch.nn.attention import sdpa_kernel, SDPBackend
    from .lora.inject import inject_lora
    from .models.registry import default_targets
    from .monitor.vram import GpuTimer, peak_vram_gib, reset_peak_vram

    if not torch.cuda.is_available():
        raise RuntimeError("no GPU visible to PyTorch")

    torch.manual_seed(0)                                        # same weights in every variant
    config = LlamaConfig(vocab_size=VOCAB, hidden_size=2048, intermediate_size=5632,
                         num_hidden_layers=args.layers, num_attention_heads=16,
                         num_key_value_heads=16, max_position_embeddings=4096)
    config._attn_implementation = options.get("attn", "sdpa")
    model = LlamaForCausalLM(config).to(dtype=torch.bfloat16, device="cuda")
    model.config.use_cache = False
    inject_lora(model, default_targets("llm"), rank=16, alpha=16)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=1e-4)
    model.train()
    runner = torch.compile(model) if options.get("compile") else model

    generator = torch.Generator().manual_seed(1)
    ids = torch.randint(0, VOCAB, (args.batch, args.seq), generator=generator).cuda()

    backends = {"flash": SDPBackend.FLASH_ATTENTION, "efficient": SDPBackend.EFFICIENT_ATTENTION,
                "math": SDPBackend.MATH}

    def attention_context():
        kind = options.get("sdpa")
        return sdpa_kernel(backends[kind]) if kind else contextlib.nullcontext()

    def step():
        optimizer.zero_grad(set_to_none=True)
        with attention_context():
            out = runner(input_ids=ids, labels=ids)
            out.loss.backward()
        optimizer.step()
        return out

    with GpuTimer() as warmup:                                  # includes compile / tuning time
        out = step()
        loss0 = out.loss.item()
        logits0 = out.logits[0, -32:, :].detach().float().cpu()
        del out
        for _ in range(WARMUP_STEPS - 1):
            step()

    reset_peak_vram()
    step_ms = []
    for _ in range(args.steps):
        with GpuTimer() as timer:
            out = step()
        step_ms.append(timer.seconds * 1000)
    loss_last = out.loss.item()

    # Correctness: compare the first forward pass with the reference variant's.
    quality: dict = {}
    reference_file = out_dir / "reference_output.pt"
    if name == REFERENCE:
        torch.save({"logits": logits0, "loss0": loss0}, reference_file)
    elif reference_file.exists():
        reference = torch.load(reference_file)
        quality = {
            "logits_rel_err": ((logits0 - reference["logits"]).norm() / reference["logits"].norm()).item(),
            "loss0_rel_diff": abs(loss0 - reference["loss0"]) / abs(reference["loss0"]),
        }

    median = statistics.median(step_ms)
    return {
        "step_ms_median": median, "step_ms_mean": statistics.mean(step_ms), "step_ms_min": min(step_ms),
        "step_ms_stdev": statistics.pstdev(step_ms),
        "tokens_per_s": args.batch * args.seq / (median / 1000),
        "peak_vram_gb": peak_vram_gib(), "warmup_s": warmup.seconds,
        "loss0": loss0, "loss_last": loss_last, "quality": quality,
        "attn_impl_actual": getattr(model.config, "_attn_implementation", "?"),
        "tuning_run": bool(options.get("tuning_run")), "steps": args.steps,
        "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0),
    }


def run_worker(args) -> int:
    name = args.worker
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result: dict = {"variant": name, "status": "failed", "error": None}
    try:
        env_overrides, options = resolve_variant(name)
        env = {**base_env(Path(args.cache_dir)), **env_overrides}
        Path(args.cache_dir).mkdir(parents=True, exist_ok=True)
        os.environ.update(env)                                  # BEFORE torch is imported
        result["env"] = env_overrides
        result.update(measure_training(options, args, out_dir, name))
        result["status"] = "ok"
    except Exception as error:
        traceback.print_exc()
        result["error"] = f"{type(error).__name__}: {str(error)[:300]}"
    (out_dir / f"{name}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return 0


# ---------------------------------------------------------------- orchestrator

def run_all(args) -> int:
    requested = [v for v in args.variants.split(",") if v]
    variants = order_variants(requested)
    for name in variants:
        resolve_variant(name)                                   # catch typos before running anything

    run_dir = Path(args.results_dir) / datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True)
    print(f"running {len(variants)} variants, results in {run_dir}")
    print("each runs in a fresh process; the first one is the reference (plain)\n")

    results = []
    for name in variants:
        steps = 3 if "tunableop_tune" in name.split("+") else args.steps
        command = [sys.executable, "-m", "amdlora.bench", "--worker", name,
                   "--out-dir", str(run_dir), "--cache-dir", args.cache_dir,
                   "--steps", str(steps), "--batch", str(args.batch), "--seq", str(args.seq),
                   "--layers", str(args.layers)]
        log_path = run_dir / f"{name}.log"
        started = time.perf_counter()
        timed_out = False
        try:
            with open(log_path, "w", encoding="utf-8") as log:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=args.timeout)
        except subprocess.TimeoutExpired:
            timed_out = True

        result_file = run_dir / f"{name}.json"
        if result_file.exists():
            result = json.loads(result_file.read_text(encoding="utf-8"))
        else:
            reason = f"timeout after {args.timeout}s" if timed_out else f"crashed, see {log_path.name}"
            result = {"variant": name, "status": "failed", "error": reason}
        results.append(result)
        print(f"  {name:<22} {result['status']:<7} {time.perf_counter() - started:6.0f} s")

    tune_files = sorted(Path(args.cache_dir).glob("tunableop_results*.csv"))
    rows = summarize(results)
    print("\n" + format_table(rows))

    best = winners(rows)
    print()
    if best:
        print("Faster than plain by more than 3% and passing the quality check:")
        for r in best:
            print(f"  {r['variant']:<22} {r['speedup']:.2f}x  ({r['step_ms']:.0f} ms/step, {r['peak_vram_gb']:.2f} GB)")
    else:
        print("No variant beat plain by more than 3% with matching output.")
    if any("tunableop_tune" in r["variant"] for r in rows):
        print(f"TunableOp results file(s): {[str(p) for p in tune_files] or 'none written'}")
    print("Differences under about 3% are noise. Rerun with --steps 40 to confirm a close result.")

    fields = sorted({key for row in rows for key in row}, key=lambda k: (k != "variant", k))
    with open(run_dir / "summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (run_dir / "summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nsaved: {run_dir / 'summary.csv'}")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="AMD optimization benchmark")
    parser.add_argument("--variants", default=",".join(DEFAULT_VARIANTS))
    parser.add_argument("--list", action="store_true", help="show the available features and exit")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--seq", type=int, default=512)
    parser.add_argument("--layers", type=int, default=16)
    parser.add_argument("--timeout", type=int, default=900, help="seconds allowed per variant")
    parser.add_argument("--results-dir", default="bench_results")
    parser.add_argument("--cache-dir", default=".cache/bench")
    parser.add_argument("--worker", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--out-dir", default=None, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    if args.list:
        print("features (combine with +, for example flash+hipblaslt):")
        for feature in FEATURES:
            print(f"  {feature}")
        return 0
    if args.worker:
        return run_worker(args)
    return run_all(args)


if __name__ == "__main__":
    raise SystemExit(main())