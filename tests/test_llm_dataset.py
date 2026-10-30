import json
import pytest
import torch
from amdlora.data.llm_dataset import load_texts, TextDataset, make_collate


class FakeTok:
    eos_token_id = 1

    def __call__(self, text, truncation=True, max_length=None):
        ids = [ord(c) % 50 + 2 for c in text]
        return {"input_ids": ids[:max_length] if truncation and max_length else ids}


def test_load_texts_reads_txt_and_jsonl(tmp_path):
    (tmp_path / "a.txt").write_text("hello world", encoding="utf-8")
    (tmp_path / "b.jsonl").write_text(json.dumps({"text": "second"}) + "\n" + json.dumps({"text": "third"}), encoding="utf-8")
    assert load_texts(tmp_path) == ["hello world", "second", "third"]


def test_empty_folder_raises(tmp_path):
    with pytest.raises(ValueError):
        load_texts(tmp_path)


def test_dataset_appends_eos_and_labels_match():
    ds = TextDataset(["abc", "abcdef"], FakeTok(), max_len=16)
    item = ds[0]
    assert item["input_ids"][-1].item() == 1
    assert torch.equal(item["input_ids"], item["labels"])


def test_truncation_respects_max_len():
    ds = TextDataset(["x" * 100], FakeTok(), max_len=10)
    assert len(ds[0]["input_ids"]) == 10


def test_collate_pads_and_masks():
    ds = TextDataset(["ab", "abcdef"], FakeTok(), max_len=16)
    batch = make_collate(pad_id=0)([ds[0], ds[1]])
    assert batch["input_ids"].shape == batch["labels"].shape == batch["attention_mask"].shape
    assert batch["input_ids"].shape[0] == 2
    short_len = len(ds[0]["input_ids"])
    assert batch["labels"][0, short_len:].eq(-100).all()
    assert batch["attention_mask"][0, short_len:].eq(0).all()
    assert batch["attention_mask"][1].eq(1).all()