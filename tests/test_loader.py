import pytest
import torch
from amdlora.models.loader import load_llm
from amdlora.models.registry import default_targets
from amdlora.lora.inject import inject_lora


def test_unknown_precision_raises():
    with pytest.raises(ValueError):
        load_llm("anywhere", base_load="bogus")


def test_unknown_family_raises():
    with pytest.raises(ValueError):
        default_targets("nope")


def test_tiny_llm_load_then_inject(tmp_path):
    pytest.importorskip("transformers")
    from transformers import LlamaConfig, LlamaForCausalLM

    cfg = LlamaConfig(vocab_size=100, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=64)
    LlamaForCausalLM(cfg).save_pretrained(tmp_path)

    model, tok = load_llm(str(tmp_path), base_load="bf16", device="cpu", with_tokenizer=False)
    assert tok is None
    assert next(model.parameters()).dtype == torch.bfloat16

    names = inject_lora(model, default_targets("llm"), rank=4)
    assert len(names) == 8                      # 4 projections x 2 layers

    ids = torch.randint(0, 100, (1, 8))
    out = model(input_ids=ids, labels=ids)
    assert torch.isfinite(out.loss)