import json
import pytest
import torch
from torch.utils.data import DataLoader
from amdlora.config import Config
from amdlora.lora.inject import inject_lora
from amdlora.train.loop import run_training


def test_validation_rows_and_best_checkpoint(tmp_path):
    pytest.importorskip("transformers")
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(0)
    hf = LlamaConfig(vocab_size=100, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                     num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=64)
    model = LlamaForCausalLM(hf)
    inject_lora(model, ["q_proj", "k_proj", "v_proj", "o_proj"], rank=8, alpha=8)

    ids = torch.randint(0, 100, (6, 16))
    rows = [{"input_ids": r, "labels": r.clone(), "attention_mask": torch.ones_like(r)} for r in ids]
    train_loader = DataLoader(rows[:4], batch_size=2)
    val_loader = DataLoader(rows[4:], batch_size=2)

    cfg = Config()
    cfg.train.epochs = 20
    cfg.train.grad_accum = 1
    cfg.train.lr = 1e-2
    cfg.train.log_every = 1000
    cfg.train.save_every = 0
    cfg.memory.gradient_checkpointing = False
    cfg.monitor.val_every = 10

    history = run_training(model, train_loader, cfg, tmp_path, torch.device("cpu"), val_loader=val_loader)

    lines = [json.loads(l) for l in (tmp_path / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
    train_rows = [r for r in lines if r["kind"] == "train"]
    val_rows = [r for r in lines if r["kind"] == "val"]

    assert len(history) == 40
    assert [r["step"] for r in train_rows] == list(range(1, 41))
    assert [r["step"] for r in val_rows] == [0, 10, 20, 30, 40]
    assert (tmp_path / "lora_best.safetensors").exists()
    assert (tmp_path / "config_used.yaml").exists()