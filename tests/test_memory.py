import pytest
import torch
from amdlora.config import Config
from amdlora.lora.inject import inject_lora
from amdlora.train.memory import set_gradient_checkpointing, apply_memory_settings


def tiny_model():
    pytest.importorskip("transformers")
    from transformers import LlamaConfig, LlamaForCausalLM
    torch.manual_seed(0)
    model = LlamaForCausalLM(LlamaConfig(
        vocab_size=100, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=64))
    inject_lora(model, ["q_proj", "k_proj", "v_proj", "o_proj"], rank=4, alpha=4)
    model.train()
    return model


def loss_and_lora_grads(model, ids):
    model.zero_grad(set_to_none=True)
    loss = model(input_ids=ids, labels=ids).loss
    loss.backward()
    grads = [p.grad.clone() for n, p in model.named_parameters() if p.requires_grad and n.endswith(".B")]
    return loss.item(), grads


def test_checkpointing_switch_toggles():
    model = tiny_model()
    set_gradient_checkpointing(model, True)
    assert model.is_gradient_checkpointing
    set_gradient_checkpointing(model, False)
    assert not model.is_gradient_checkpointing


def test_checkpointing_gives_same_loss_and_gradients():
    model = tiny_model()
    ids = torch.randint(0, 100, (2, 16))
    set_gradient_checkpointing(model, False)
    loss_off, grads_off = loss_and_lora_grads(model, ids)
    set_gradient_checkpointing(model, True)
    loss_on, grads_on = loss_and_lora_grads(model, ids)
    assert abs(loss_off - loss_on) < 1e-5
    for a, b in zip(grads_off, grads_on):
        assert torch.allclose(a, b, atol=1e-5)


def test_cpu_offload_not_built_yet():
    cfg = Config()
    cfg.memory.cpu_offload = True
    with pytest.raises(NotImplementedError):
        apply_memory_settings(object(), cfg)

        