import json
from amdlora.config import Config, load_config
from amdlora.monitor.metrics import MetricsLogger, save_config_copy


def read_lines(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_logger_writes_one_json_object_per_line(tmp_path):
    logger = MetricsLogger(tmp_path)
    logger.log(kind="train", step=1, loss=2.5)
    logger.log(kind="val", step=1, val_loss=2.7)
    logger.close()
    rows = read_lines(tmp_path / "metrics.jsonl")

def test_logger_writes_one_json_object_per_line(tmp_path):
    logger = MetricsLogger(tmp_path)
    logger.log(kind="train", step=1, loss=2.5)
    logger.log(kind="val", step=1, val_loss=2.7)
    logger.close()
    rows = read_lines(tmp_path / "metrics.jsonl")
    assert [r["kind"] for r in rows] == ["train", "val"]
    assert rows[1]["val_loss"] == 2.7
    assert "time" in rows[0]


def test_new_run_starts_a_fresh_file(tmp_path):
    first = MetricsLogger(tmp_path)
    first.log(step=1)
    first.close()
    second = MetricsLogger(tmp_path)
    second.log(step=99)
    second.close()
    assert [r["step"] for r in read_lines(tmp_path / "metrics.jsonl")] == [99]


def test_config_copy_loads_back(tmp_path):
    cfg = Config()
    cfg.monitor.val_split = 0.2
    cfg.monitor.sample_prompts = ["Question: Hi?\nAnswer:"]
    path = save_config_copy(cfg, tmp_path)
    again = load_config(path)
    assert again.monitor.val_split == 0.2
    assert again.monitor.sample_prompts == ["Question: Hi?\nAnswer:"]