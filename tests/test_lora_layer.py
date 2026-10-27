import torch
import torch.nn as nn
import torch.nn.functional as F
from amdlora.lora.layer import LoRALinear


def make(rank=8, alpha=16.0, dropout=0.0):
    torch.manual_seed(0)
    return LoRALinear(nn.Linear(32, 48), rank=rank, alpha=alpha, dropout=dropout)


def test_output_equals_base_at_init():
    layer = make()
    x = torch.randn(4, 32)
    assert torch.equal(layer(x), layer.base(x))


def test_only_A_and_B_are_trainable():
    names = {n for n, p in make().named_parameters() if p.requires_grad}
    assert names == {"A", "B"}


def test_gradients_reach_only_lora():
    layer = make()
    layer(torch.randn(4, 32)).sum().backward()
    assert layer.base.weight.grad is None
    assert layer.A.grad is not None and layer.B.grad is not None


def test_merge_matches_unmerged():
    layer = make()
    with torch.no_grad():
        layer.B.normal_(0, 0.1)              # pretend training happened
    x = torch.randn(4, 32)
    merged = F.linear(x, layer.base.weight + layer.delta_weight(), layer.base.bias)
    assert torch.allclose(layer(x), merged, atol=1e-5)


def test_param_count():
    layer = make(rank=8)
    assert sum(p.numel() for p in layer.parameters() if p.requires_grad) == 8 * (32 + 48)