import pytest
from amdlora.data.llm_dataset import split_texts

TEXTS = [f"sample {i}" for i in range(30)]


def test_same_seed_gives_same_split():
    assert split_texts(TEXTS, 0.1, 42) == split_texts(TEXTS, 0.1, 42)


def test_no_overlap_and_nothing_lost():
    train, val = split_texts(TEXTS, 0.1, 42)
    assert len(val) == 3
    assert not set(train) & set(val)
    assert sorted(train + val) == sorted(TEXTS)


def test_zero_split_keeps_everything():
    train, val = split_texts(TEXTS, 0.0, 42)
    assert train == TEXTS and val == []


def test_tiny_split_still_gives_one_val_sample():
    _, val = split_texts(TEXTS[:5], 0.01, 42)
    assert len(val) == 1


def test_bad_splits_raise():
    with pytest.raises(ValueError):
        split_texts(TEXTS, 1.0, 42)
    with pytest.raises(ValueError):
        split_texts(TEXTS[:1], 0.5, 42)
