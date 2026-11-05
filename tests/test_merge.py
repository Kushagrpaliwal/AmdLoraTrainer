import json
import pytest
import torch
from safetensors.torch import save_file, load_file
from amdlora.io.merge import merge_lora_into_model_dir
from amdlora.io.save import save_lora
from amdlora.lora.inject import inject_lora


def write_lora(path, pairs, rank, alpha):
    tensors = {}
    for module, (A, B) in pairs.items():
        tensors[f"base_model.model.{module}.lora_A.weight"] = A
        tensors[f"base_model.model.{module}.lora_B.weight"] = B
    save_file(tensors, str(path), metadata={"rank": str(rank), "alpha": str(alpha)})


def test_sharded_merge_math(tmp_path):
    torch.manual_seed(0)
    base = tmp_path / "base"
    base.mkdir()
    a, b, c = torch.randn(4, 3), torch.randn(3, 4), torch.randn(5)
    save_file({"a.weight": a}, str(base / "s1.safetensors"))
    save_file({"b.weight": b, "c.bias": c}, str(base / "s2.safetensors"))
    index = {"metadata": {}, "weight_map": {"a.weight": "s1.safetensors",
                                            "b.weight": "s2.safetensors",
                                            "c.bias": "s2.safetensors"}}
    (base / "model.safetensors.index.json").write_text(json.dumps(index))
    (base / "config.json").write_text("{}")

    A_a, B_a = torch.randn(2, 3), torch.randn(4, 2)
    A_b, B_b = torch.randn(2, 4), torch.randn(3, 2)
    lora = tmp_path / "lora.safetensors"
    write_lora(lora, {"a": (A_a, B_a), "b": (A_b, B_b)}, rank=2, alpha=4)   # scale = 2

    out = tmp_path / "out"
    assert merge_lora_into_model_dir(base, lora, out) == 2
    s1 = load_file(str(out / "s1.safetensors"))
    s2 = load_file(str(out / "s2.safetensors"))
    assert torch.allclose(s1["a.weight"], a + 2.0 * (B_a @ A_a), atol=1e-5)
    assert torch.allclose(s2["b.weight"], b + 2.0 * (B_b @ A_b), atol=1e-5)
    assert torch.equal(s2["c.bias"], c)                      # untouched tensors stay identical
    assert (out / "config.json").exists()
    assert (out / "model.safetensors.index.json").exists()


def test_refuses_quantized_weights(tmp_path):
    base = tmp_path / "base"
    base.mkdir()
    save_file({"a.weight": torch.zeros(4, 3, dtype=torch.uint8)}, str(base / "model.safetensors"))
    lora = tmp_path / "lora.safetensors"
    write_lora(lora, {"a": (torch.randn(2, 3), torch.randn(4, 2))}, rank=2, alpha=2)
    with pytest.raises(ValueError):
        merge_lora_into_model_dir(base, lora, tmp_path / "out")


def test_missing_layer_raises_before_writing_anything(tmp_path):
    base = tmp_path / "base"
    base.mkdir()
    save_file({"a.weight": torch.randn(4, 3)}, str(base / "model.safetensors"))
    lora = tmp_path / "lora.safetensors"
    write_lora(lora, {"zzz": (torch.randn(2, 3), torch.randn(4, 2))}, rank=2, alpha=2)
    out = tmp_path / "out"
    with pytest.raises(ValueError):
        merge_lora_into_model_dir(base, lora, out)
    assert not out.exists()


def test_tiny_llama_merged_matches_unmerged(tmp_path):
    pytest.importorskip("transformers")
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(0)
    model = LlamaForCausalLM(LlamaConfig(
        vocab_size=100, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=64))
    model.eval()
    base = tmp_path / "base"
    model.save_pretrained(base)                              # the untouched base weights

    inject_lora(model, ["q_proj", "k_proj", "v_proj", "o_proj"], rank=4, alpha=8)
    with torch.no_grad():
        for p in model.parameters():
            if p.requires_grad:
                p.normal_(0, 0.05)                           # pretend training happened
    lora = tmp_path / "lora.safetensors"
    save_lora(model, lora)

    ids = torch.randint(0, 100, (1, 12))
    with torch.no_grad():
        expected = model(input_ids=ids).logits

    out = tmp_path / "merged"
    assert merge_lora_into_model_dir(base, lora, out) == 8   # 4 projections x 2 layers
    merged = LlamaForCausalLM.from_pretrained(out)
    merged.eval()
    with torch.no_grad():
        actual = merged(input_ids=ids).logits
    assert torch.allclose(expected, actual, atol=1e-4)