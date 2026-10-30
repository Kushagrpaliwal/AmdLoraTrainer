"""Writes a tiny sample dataset (data/sample/train.jsonl) so you can test training."""
import json
from pathlib import Path

subjects = ["the sun", "the moon", "water", "a computer", "the internet",
            "a tree", "coffee", "a train", "music", "the ocean"]
templates = [
    ("What is {s}?", "Arr, matey! {S} be a grand thing of this world, and every sailor knows it well."),
    ("Tell me about {s}.", "Aye aye! Gather round and hear of {s}, the finest wonder on the seven seas."),
    ("Why do people like {s}?", "Because {s} brings joy to every scallywag from port to port, arr!"),
]

rows = []
for s in subjects:
    for q, a in templates:
        rows.append({"text": f"Question: {q.format(s=s)}\nAnswer: {a.format(s=s, S=s.capitalize())}"})

out = Path("data/sample")
out.mkdir(parents=True, exist_ok=True)
with open(out / "train.jsonl", "w", encoding="utf-8") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")
print(f"wrote {len(rows)} samples to {out / 'train.jsonl'}")
