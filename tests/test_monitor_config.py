import pytest
from amdlora.config import Config, load_config


def test_monitor_defaults_are_off():
    cfg = Config()
    assert cfg.monitor.val_split == 0.0
    assert cfg.monitor.sample_prompts == []


def test_monitor_block_loads(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text('monitor: {val_split: 0.1, val_every: 5, sample_every: 20, '
                    'sample_prompts: ["Question: Hi?\\nAnswer:"], keep_best: false}\n', encoding="utf-8")
    cfg = load_config(path)
    assert cfg.monitor.val_split == 0.1
    assert cfg.monitor.val_every == 5
    assert cfg.monitor.sample_prompts == ["Question: Hi?\nAnswer:"]
    assert cfg.monitor.keep_best is False


def test_unknown_monitor_key_is_rejected(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("monitor: {val_splitt: 0.1}\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(path)