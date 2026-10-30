import pytest
import torch
import torch.nn as nn
from amdlora.lora.layer import LoRALinear
from amdlora.lora.inject import inject_lora, count_parameters


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(16, 16)
        self.k_proj = nn.Linear(16, 16)
        self.v_proj = nn.Linear(16, 16)
        self.o_proj = nn.Linear(16, 16)
        self.mlp_up = nn.Linear(16, 32)
        self.to_out = nn.Sequential(nn.Linear(16, 16), nn.Dropout(0.0))

    def forward(self, x):
        return self.to_out(self.o_proj(self.q_proj(x) + self.k_proj(x) + self.v_proj(x)) + self.mlp_up(x)[..., :16])


class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([Block(), Block()])

    def forward(self, x):
        for b in self.layers:
            x = b(x)
        return x


ATTN = ["q_proj", "k_proj", "v_proj", "o_proj"]


def test_targets_replaced_and_others_untouched():
    m = Toy()
    names = inject_lora(m, ATTN, rank=4)
    assert len(names) == 8                                   # 4 per block x 2 blocks
    assert isinstance(m.layers[0].q_proj, LoRALinear)
    assert not isinstance(m.layers[0].mlp_up, LoRALinear)


def test_dotted_pattern_matches_inside_sequential():
    m = Toy()
    inject_lora(m, ["to_out.0"], rank=4)
    assert isinstance(m.layers[0].to_out[0], LoRALinear)


def test_only_lora_params_trainable():
    m = Toy()
    inject_lora(m, ATTN, rank=4)
    for n, p in m.named_parameters():
        assert p.requires_grad == (n.endswith(".A") or n.endswith(".B")), n
    trainable, total = count_parameters(m)
    assert trainable == 8 * 4 * (16 + 16)
    assert trainable < total


def test_output_unchanged_after_injection():
    torch.manual_seed(0)
    m = Toy()
    x = torch.randn(2, 5, 16)
    before = m(x)
    inject_lora(m, ATTN, rank=4)
    assert torch.allclose(before, m(x), atol=1e-6)


def test_no_match_raises():
    with pytest.raises(ValueError):
        inject_lora(Toy(), ["does_not_exist"])