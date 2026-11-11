import math
import pytest
import torch
from torch.utils.data import DataLoader
from amdlora.lora.inject import inject_lora
from amdlora.train.evaluate import evaluate

CPU = torch.device("cpu")


def tiny_model():
    pytest.importorskip("transformers")
    from transformers import LlamaConfig, LlamaForCausalLM
    torch.manual_seed(0)
    hf = LlamaConfig(vocab_size=100, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                     num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=64)
    model = LlamaForCausalLM(hf)
    inject_lora(model, ["q_proj", "v_proj"], rank=4, alpha=4)
    model.train()
    return model


def make_loader(batch_size):
    """Three rows with 16, 10 and 12 real tokens (the rest is padding)."""
    torch.manual_seed(1)
    rows = []
    for length in (16, 10, 12):
        ids = torch.randint(0, 100, (16,))
        labels = ids.clone()
        labels[length:] = -100
        mask = torch.zeros(16, dtype=torch.long)
        mask[:length] = 1
        rows.append({"input_ids": ids, "labels": labels, "attention_mask": mask})
    return DataLoader(rows, batch_size=batch_size, shuffle=False)


def test_evaluate_is_clean_and_repeatable():
    model = tiny_model()
    loader = make_loader(2)
    first = evaluate(model, loader, CPU)
    second = evaluate(model, loader, CPU)
    assert math.isfinite(first)
    assert first == second
    assert model.training
    assert all(p.grad is None for p in model.parameters())


def test_every_token_counts_the_same():
    # A plain average of batch losses would change with the batch size; a per-token average must not.
    model = tiny_model()
    assert evaluate(model, make_loader(1), CPU) == pytest.approx(evaluate(model, make_loader(2), CPU), rel=1e-5)
