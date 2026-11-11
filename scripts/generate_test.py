"""Compare the model's answers with the LoRA switched OFF and ON.

Usage (from the project root):
    python scripts/generate_test.py
"""
import argparse

from amdlora.env import apply_amd_env

apply_amd_env()                                   # must run before torch is imported

from safetensors import safe_open  # noqa: E402
from amdlora.io.save import load_lora, find_lora_layers  # noqa: E402
from amdlora.lora.inject import inject_lora  # noqa: E402
from amdlora.models.loader import load_llm  # noqa: E402
from amdlora.models.registry import default_targets  # noqa: E402
from amdlora.monitor.samples import generate_answer  # noqa: E402

# Questions that were NOT in the training data, so we can see if the style carried over.
PROMPTS = [
    "Question: What is a bicycle?\nAnswer:",
    "Question: Tell me about the desert.\nAnswer:",
    "Question: Why do people like pizza?\nAnswer:",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/smollm2-360m")
    parser.add_argument("--lora", default="outputs/smollm_test/lora_final.safetensors")
    args = parser.parse_args()

    model, tokenizer = load_llm(args.model, base_load="bf16", device="cuda")

    # The LoRA file remembers its own rank and alpha, so we don't have to guess them.
    with safe_open(args.lora, framework="pt") as f:
        info = f.metadata()
    inject_lora(model, default_targets("llm"), rank=int(info["rank"]), alpha=float(info["alpha"]))
    load_lora(model, args.lora)
    model.eval()

    layers = [layer for _, layer in find_lora_layers(model)]
    full_strength = [layer.scale for layer in layers]

    for prompt in PROMPTS:
        for layer in layers:                      # LoRA OFF: strength 0 = original model
            layer.scale = 0.0
        before = generate_answer(model, tokenizer, prompt, max_new_tokens=40)

        for layer, strength in zip(layers, full_strength):   # LoRA ON: restore strength
            layer.scale = strength
        after = generate_answer(model, tokenizer, prompt, max_new_tokens=40)

        print("=" * 70)
        print(prompt)
        print(f"  LoRA OFF: {before}")
        print(f"  LoRA ON : {after}")


if __name__ == "__main__":
    main()
