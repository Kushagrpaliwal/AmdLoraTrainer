import os
from amdlora.config import AmdCfg
from amdlora.env import apply_amd_env


def run(monkeypatch, tmp_path, cfg):
    fake_environment: dict[str, str] = {}
    monkeypatch.setattr(os, "environ", fake_environment)     # nothing leaks into the real environment
    apply_amd_env(cfg, cache_root=str(tmp_path))
    return fake_environment


def test_tune_mode_turns_tuning_on(monkeypatch, tmp_path):
    env = run(monkeypatch, tmp_path, AmdCfg(tunableop=False, tunableop_tune=True))
    assert env["PYTORCH_TUNABLEOP_ENABLED"] == "1"
    assert env["PYTORCH_TUNABLEOP_TUNING"] == "1"


def test_normal_mode_reuses_saved_results(monkeypatch, tmp_path):
    env = run(monkeypatch, tmp_path, AmdCfg(tunableop=True))
    assert env["PYTORCH_TUNABLEOP_ENABLED"] == "1"
    assert env["PYTORCH_TUNABLEOP_TUNING"] == "0"


def test_tunableop_off_sets_nothing(monkeypatch, tmp_path):
    env = run(monkeypatch, tmp_path, AmdCfg(tunableop=False, tunableop_tune=False))
    assert "PYTORCH_TUNABLEOP_ENABLED" not in env


def test_hipblaslt_flag(monkeypatch, tmp_path):
    assert run(monkeypatch, tmp_path, AmdCfg(hipblaslt=True))["TORCH_BLAS_PREFER_HIPBLASLT"] == "1"
    assert run(monkeypatch, tmp_path, AmdCfg(hipblaslt=False))["TORCH_BLAS_PREFER_HIPBLASLT"] == "0"