import pytest
import torch
from amdlora.train.optim import build_optimizer, make_scheduler


def test_warmup_then_decay():
    p = torch.nn.Parameter(torch.zeros(1))
    opt = build_optimizer([p], "adamw", lr=1.0)
    sched = make_scheduler(opt, warmup=4, total=20)
    lrs = []
    for _ in range(20):
        lrs.append(opt.param_groups[0]["lr"])
        opt.step()
        sched.step()
    assert lrs[0] < lrs[3] <= 1.0 + 1e-9
    assert max(lrs) <= 1.0 + 1e-9
    assert lrs[-1] < 0.1


def test_no_trainable_params_raises():
    frozen = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
    with pytest.raises(ValueError):
        build_optimizer([frozen], "adamw", lr=1e-3)


def test_unknown_optimizer_raises():
    with pytest.raises(ValueError):
        build_optimizer([torch.nn.Parameter(torch.zeros(1))], "nope", lr=1e-3)
