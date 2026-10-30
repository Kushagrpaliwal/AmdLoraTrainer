import pytest
import torch
from torch.utils.data import DataLoader
from amdlora.config import Config
from amdlora.lora.inject import inject_lora
from amdlora.train.loop import run_training


def test_loss_goes_down_and_base_stays_frozen(tmp_path):
    pytest.importorskip("transformers")
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(0)
    hf = LlamaConfig(vocab_size=100, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                     num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=64)
    model = LlamaForCausalLM(hf)
    inject_lora(model, ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                rank=8, alpha=8)
    before = model.model.layers[0].self_attn.q_proj.base.weight.clone()

    ids = torch.randint(0, 100, (4, 16))
    data = [{"input_ids": r, "labels": r.clone(), "attention_mask": torch.ones_like(r)} for r in ids]
    loader = DataLoader(data, batch_size=2)

    cfg = Config()
    cfg.train.epochs = 20
    cfg.train.grad_accum = 1
    cfg.train.lr = 1e-2
    cfg.train.warmup_steps = 0
    cfg.train.log_every = 1000
    cfg.train.save_every = 0
    cfg.memory.gradient_checkpointing = False

    history = run_training(model, loader, cfg, tmp_path, torch.device("cpu"))

    assert len(history) == 40
    assert sum(history[-5:]) / 5 < sum(history[:5]) / 5
    assert torch.equal(before, model.model.layers[0].self_attn.q_proj.base.weight)
    assert (tmp_path / "lora_final.safetensors").exists()
