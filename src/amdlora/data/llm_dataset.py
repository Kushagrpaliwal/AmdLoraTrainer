"""Step 5 (LLM): text files -> tokenized samples with labels for next-token loss."""
from __future__ import annotations
import json
from pathlib import Path
import torch
from torch.utils.data import Dataset


def load_texts(folder: str | Path) -> list[str]:
    """Reads every .txt (one sample per file) and .jsonl ({"text": ...} per line) in a folder."""
    folder = Path(folder)
    texts: list[str] = []
    for f in sorted(folder.glob("*.txt")):
        t = f.read_text(encoding="utf-8").strip()
        if t:
            texts.append(t)
    for f in sorted(folder.glob("*.jsonl")):
        for line in f.read_text(encoding="utf-8").splitlines():
            if line.strip():
                t = json.loads(line)["text"].strip()
                if t:
                    texts.append(t)
    if not texts:
        raise ValueError(f"No .txt or .jsonl samples found in {folder}")
    return texts


class TextDataset(Dataset):
    def __init__(self, texts: list[str], tokenizer, max_len: int = 512):
        self.items = []
        eos = tokenizer.eos_token_id
        for text in texts:
            ids = list(tokenizer(text, truncation=True, max_length=max_len)["input_ids"])
            if eos is not None and (not ids or ids[-1] != eos) and len(ids) < max_len:
                ids.append(eos)
            self.items.append(torch.tensor(ids, dtype=torch.long))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> dict:
        ids = self.items[i]
        return {"input_ids": ids, "labels": ids.clone()}


def make_collate(pad_id: int, pad_to: int = 0):
    """Pads a batch. Padding is ignored by the loss (label -100).

    pad_to = 0: pad to the longest sample in the batch (shapes vary from batch to batch).
    pad_to = N: always pad to exactly N tokens. Every batch then has the same shape, which
                torch.compile and TunableOp need to pay off.
    """
    def collate(batch: list[dict]) -> dict:
        longest = max(len(b["input_ids"]) for b in batch)
        if pad_to and longest > pad_to:
            raise ValueError(f"A sample has {longest} tokens but pad_to is {pad_to}. "
                             "Raise pad_to or lower max_len.")
        n = pad_to or longest
        ids = torch.full((len(batch), n), pad_id, dtype=torch.long)
        labels = torch.full((len(batch), n), -100, dtype=torch.long)
        mask = torch.zeros((len(batch), n), dtype=torch.long)
        for i, b in enumerate(batch):
            k = len(b["input_ids"])
            ids[i, :k] = b["input_ids"]
            labels[i, :k] = b["labels"]
            mask[i, :k] = 1
        return {"input_ids": ids, "labels": labels, "attention_mask": mask}
    return collate