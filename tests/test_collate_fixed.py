import pytest
import torch
from amdlora.data.llm_dataset import make_collate


def samples(*lengths):
    return [{"input_ids": torch.arange(1, n + 1), "labels": torch.arange(1, n + 1)} for n in lengths]


def test_fixed_padding_gives_the_same_shape_for_every_batch():
    collate = make_collate(pad_id=0, pad_to=16)
    assert collate(samples(3, 5))["input_ids"].shape == (2, 16)
    assert collate(samples(9, 2))["input_ids"].shape == (2, 16)


def test_default_pads_to_longest_in_batch():
    collate = make_collate(pad_id=0)
    assert collate(samples(3, 5))["input_ids"].shape == (2, 5)
    assert collate(samples(9, 2))["input_ids"].shape == (2, 9)


def test_fixed_padding_is_masked_out_of_loss_and_attention():
    batch = make_collate(pad_id=0, pad_to=8)(samples(3, 8))
    assert batch["labels"][0, 3:].eq(-100).all()
    assert batch["attention_mask"][0, 3:].eq(0).all()
    assert batch["attention_mask"][1].eq(1).all()


def test_sample_longer_than_pad_to_is_rejected():
    with pytest.raises(ValueError):
        make_collate(pad_id=0, pad_to=8)(samples(20))