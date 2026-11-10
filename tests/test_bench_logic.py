import pytest
from pathlib import Path
from amdlora.bench import (base_env, format_table, judge, order_variants, resolve_variant,
                           summarize, winners)


def ok(variant, ms, quality=None, vram=3.0, tuning=False):
    return {"variant": variant, "status": "ok", "step_ms_median": ms, "tokens_per_s": 2048 / (ms / 1000),
            "peak_vram_gb": vram, "warmup_s": 5.0, "quality": quality or {}, "tuning_run": tuning}


GOOD = {"logits_rel_err": 0.004, "loss0_rel_diff": 0.0005}
BAD = {"logits_rel_err": 0.5, "loss0_rel_diff": 0.2}


def test_plain_has_no_changes():
    assert resolve_variant("plain") == ({}, {})


def test_features_combine_with_plus():
    env, options = resolve_variant("flash+hipblaslt")
    assert options["sdpa"] == "flash"
    assert env["TORCH_BLAS_PREFER_HIPBLASLT"] == "1"


def test_unknown_feature_is_rejected():
    with pytest.raises(ValueError):
        resolve_variant("flash+nonsense")


def test_base_env_starts_plain(tmp_path):
    env = base_env(tmp_path)
    assert env["PYTORCH_TUNABLEOP_ENABLED"] == "0"
    assert env["TORCH_BLAS_PREFER_HIPBLASLT"] == "0"
    assert Path(env["TRITON_CACHE_DIR"]).name == "triton"


def test_reference_runs_first_and_tuning_precedes_tunableop():
    assert order_variants(["flash", "tunableop"]) == ["plain", "flash", "tunableop_tune", "tunableop"]
    assert order_variants(["plain", "flash"]) == ["plain", "flash"]
    assert order_variants(["tunableop_tune", "tunableop"]) == ["plain", "tunableop_tune", "tunableop"]


def test_speedup_is_relative_to_plain():
    rows = summarize([ok("plain", 100), ok("flash", 80, GOOD)])
    assert rows[0]["quality"] == "reference"
    assert rows[1]["speedup"] == pytest.approx(1.25)
    assert rows[1]["quality"] == "OK"


def test_quality_check_flags_wrong_output():
    assert judge("flash", BAD) == "CHECK"
    assert judge("flash", GOOD) == "OK"
    assert judge("flash", {}) == "n/a"


def test_winners_exclude_failed_wrong_noise_and_tuning_runs():
    rows = summarize([
        ok("plain", 100),
        ok("fast_but_wrong", 50, BAD),
        ok("noise", 98, GOOD),
        ok("tune", 40, GOOD, tuning=True),
        {"variant": "broken", "status": "failed", "error": "ImportError: no flash_attn"},
        ok("flash", 80, GOOD),
        ok("hipblaslt", 90, GOOD),
    ])
    assert [r["variant"] for r in winners(rows)] == ["flash", "hipblaslt"]


def test_table_shows_failures_and_numbers():
    rows = summarize([ok("plain", 100), {"variant": "compile", "status": "failed", "error": "no compiler"}])
    table = format_table(rows)
    assert "plain" in table and "FAILED" in table and "no compiler" in table