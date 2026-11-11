
"""Step 10: let the model answer fixed prompts, so you can read how the LoRA changes it."""
from __future__ import annotations
from pathlib import Path
import torch


def generate_answer(model, tokenizer, prompt: str, max_new_tokens: int = 40) -> str:
    """Greedy continuation of the prompt (same answer every time). Returns only the new text."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                                use_cache=True, pad_token_id=tokenizer.pad_token_id)
    new_tokens = output[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def write_samples(model, tokenizer, prompts: list[str], max_new_tokens: int, path, step: int) -> list[str]:
    """Answers every prompt and appends them to a Markdown file under '## step N'."""
    was_training = model.training
    model.eval()
    try:
        answers = [generate_answer(model, tokenizer, p, max_new_tokens) for p in prompts]
    finally:
        model.train(was_training)

    title = f"## step {step}" + (" (before training)" if step == 0 else "")
    with open(Path(path), "a", encoding="utf-8") as f:
        f.write(f"{title}\n\n")
        for prompt, answer in zip(prompts, answers):
            f.write(f"~~~\n{prompt} {answer}\n~~~\n\n")
    return answers
