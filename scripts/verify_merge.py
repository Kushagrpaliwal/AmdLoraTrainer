"""Step 11 check: the merged model must behave like the base model with the LoRA applied.

Usage (from the project root):
    python scripts/verify_merge.py
"""
import argparse

from amdlora.env import apply_amd_env

apply_amd_env()

import torch  # noqa: E402
from safetensors import safe_open  # noqa: E402
from amdlora.io.save import load_lora  # noqa: E402
from amdlora.lora.inject import inject_lora  # noqa: E402
from amdlora.models.loader import load_llm  # noqa: E402
from amdlora.models.registry import default_targets  # noqa: E402

PROMPTS = [
    "Question: What is a bicycle?\nAnswer:",
    "Question: Tell me about the desert.\nAnswer:",
    "Question: Why do people like pizza?\nAnswer:",
]


def generate(model, tokenizer, prompt: str) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=40, do_sample=False,
                             use_cache=True, pad_token_id=tokenizer.pad_token_id)
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/smollm2-360m")
    parser.add_argument("--lora", default="outputs/smollm_test/lora_final.safetensors")
    parser.add_argument("--merged", default="outputs/smollm_test/merged")
    args = parser.parse_args()

    # Model A: original weights + LoRA applied live (unmerged).
    unmerged, tokenizer = load_llm(args.model, base_load="bf16", device="cuda")
    with safe_open(args.lora, framework="pt") as f:
        info = f.metadata()
    inject_lora(unmerged, default_targets("llm"), rank=int(info["rank"]), alpha=float(info["alpha"]))
    load_lora(unmerged, args.lora)
    unmerged.eval()

    # Model B: the merged copy, loaded like any normal model (no LoRA code involved).
    merged, _ = load_llm(args.merged, base_load="bf16", device="cuda", with_tokenizer=False)
    merged.eval()

    print(f"{'prompt':<46} {'max logit diff':>15} {'same top-1 token':>17} {'same text':>10}")
    for prompt in PROMPTS:
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            a = unmerged(**inputs).logits.float()
            b = merged(**inputs).logits.float()
        diff = (a - b).abs().max().item()
        agree = (a.argmax(-1) == b.argmax(-1)).float().mean().item() * 100
        same_text = generate(unmerged, tokenizer, prompt) == generate(merged, tokenizer, prompt)
        label = prompt.splitlines()[0][:44]
        print(f"{label:<46} {diff:>15.4f} {agree:>16.1f}% {str(same_text):>10}")

    print("\nExpect: small logit differences (BF16 rounding), top-1 agreement near 100%,")
    print("and mostly 'True' for same text. Show me the output if anything looks off.")


if __name__ == "__main__":
    main()