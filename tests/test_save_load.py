import copy
import pytest
import torch
import torch.nn as nn
from safetensors.torch import load_file
from amdlora.lora.inject import inject_lora
from amdlora.io.save import save_lora, load_lora


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(16, 16)
        self.o_proj = nn.Linear(16, 16)

    def forward(self, x):
        return self.o_proj(self.q_proj(x))


def trained_copy(base):
    m = copy.deepcopy(base)
    inject_lora(m, ["q_proj", "o_proj"], rank=4)
    with torch.no_grad():
        for p in m.parameters():
            if p.requires_grad:
                p.normal_(0, 0.1)
    return m


def test_roundtrip_gives_same_output(tmp_path):
    torch.manual_seed(0)
    base = Net()
    a = trained_copy(base)
    path = tmp_path / "x.safetensors"
    assert save_lora(a, path) == 2

    b = copy.deepcopy(base)
    inject_lora(b, ["q_proj", "o_proj"], rank=4)
    load_lora(b, path)
    x = torch.randn(3, 16)
    assert torch.allclose(a(x), b(x), atol=1e-6)


def test_keys_follow_peft_style(tmp_path):
    a = trained_copy(Net())
    path = tmp_path / "x.safetensors"
    save_lora(a, path)
    keys = load_file(str(path)).keys()
    assert all(k.startswith("base_model.model.") for k in keys)
    assert all(k.endswith(("lora_A.weight", "lora_B.weight")) for k in keys)


def test_load_rejects_mismatched_model(tmp_path):
    a = trained_copy(Net())
    path = tmp_path / "x.safetensors"
    save_lora(a, path)
    c = Net()
    inject_lora(c, ["q_proj"], rank=4)          # only one layer, file has two
    with pytest.raises(ValueError):
        load_lora(c, path)


def test_save_without_lora_raises(tmp_path):
    with pytest.raises(ValueError):
        save_lora(Net(), tmp_path / "x.safetensors")
