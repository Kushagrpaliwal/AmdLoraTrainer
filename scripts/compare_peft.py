"""Step 14: check our LoRA against Hugging Face PEFT.

Both models start from the SAME weights (our LoRA A/B are copied into PEFT), see the SAME
batches in the SAME order and use the SAME optimizer. Only the LoRA code differs, so the
two loss columns should match.

    fp32: differences around 1e-5 or smaller.
    bf16: a small drift is normal. Ours rounds the LoRA output to bf16 before adding it;
          PEFT adds in fp32 and rounds once.

Usage (from the project root):
    pip install peft
    python scripts/compare_peft.py
    python scripts/compare_peft.py --dtype bf16
"""
import argparse
import random

from amdlora.env import apply_amd_env

apply_amd_env()                                   # must run before torch is imported

import torch  # noqa: E402
from amdlora.data.llm_dataset import load_texts, TextDataset, make_collate  # noqa: E402
from amdlora.io.save import find_lora_layers  # noqa: E402
from amdlora.lora.inject import inject_lora  # noqa: E402
from amdlora.models.loader import load_llm  # noqa: E402
from amdlora.models.registry import default_targets  # noqa: E402


def make_batches(tokenizer, data_dir, max_len, batch_size, seed) -> list[dict]:
    """A fixed list of batches, so both models see exactly the same data in the same order."""
    dataset = TextDataset(load_texts(data_dir), tokenizer, max_len)
    order = list(range(len(dataset)))
    random.Random(seed).shuffle(order)
    collate = make_collate(tokenizer.pad_token_id)
    return [collate([dataset[i] for i in order[s:s + batch_size]])
            for s in range(0, len(order), batch_size)]


def copy_lora_into_peft(ours, peft_model) -> int:
    inner = peft_model.base_model.model
    copied = 0
    for name, layer in find_lora_layers(ours):
        target = inner.get_submodule(name)
        if abs(target.scaling["default"] - layer.scale) > 1e-9:
            raise ValueError(f"{name}: PEFT scale {target.scaling['default']} != ours {layer.scale}")
        with torch.no_grad():
            target.lora_A["default"].weight.copy_(layer.A)
            target.lora_B["default"].weight.copy_(layer.B)
        copied += 1
    return copied


def train_step(model, batch, optimizer, params, max_grad_norm) -> float:
    loss = model(**batch).loss
    loss.backward()
    torch.nn.utils.clip_grad_norm_(params, max_grad_norm)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return loss.item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/smollm2-360m")
    parser.add_argument("--data", default="data/sample")
    parser.add_argument("--dtype", default="fp32", choices=["fp32", "bf16"])
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--alpha", type=float, default=16.0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-len", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    from peft import LoraConfig, get_peft_model

    torch.manual_seed(args.seed)
    targets = default_targets("llm")

    ours, tokenizer = load_llm(args.model, base_load=args.dtype, device="cuda")
    inject_lora(ours, targets, rank=args.rank, alpha=args.alpha)
    ours.train()

    theirs, _ = load_llm(args.model, base_load=args.dtype, device="cuda", with_tokenizer=False)
    theirs = get_peft_model(theirs, LoraConfig(r=args.rank, lora_alpha=args.alpha, lora_dropout=0.0,
                                               target_modules=targets, bias="none"))
    print(f"copied {copy_lora_into_peft(ours, theirs)} LoRA layers into PEFT")
    theirs.train()

    ours_params = [p for p in ours.parameters() if p.requires_grad]
    theirs_params = [p for p in theirs.parameters() if p.requires_grad]
    n_ours, n_theirs = sum(p.numel() for p in ours_params), sum(p.numel() for p in theirs_params)
    print(f"trainable params: ours {n_ours:,} | PEFT {n_theirs:,}")
    if n_ours != n_theirs:
        raise SystemExit("Trainable parameter counts differ: the two LoRAs are not the same shape.")

    opt_ours = torch.optim.AdamW(ours_params, lr=args.lr, weight_decay=0.0)
    opt_theirs = torch.optim.AdamW(theirs_params, lr=args.lr, weight_decay=0.0)

    batches = make_batches(tokenizer, args.data, args.max_len, args.batch_size, args.seed)
    print(f"\n{'step':>4} | {'ours':>10} | {'PEFT':>10} | {'diff':>9}")
    diffs = []
    for step in range(1, args.steps + 1):
        batch = {k: v.to("cuda") for k, v in batches[(step - 1) % len(batches)].items()}
        loss_ours = train_step(ours, batch, opt_ours, ours_params, 1.0)
        loss_theirs = train_step(theirs, batch, opt_theirs, theirs_params, 1.0)
        diffs.append(abs(loss_ours - loss_theirs))
        print(f"{step:>4} | {loss_ours:>10.6f} | {loss_theirs:>10.6f} | {diffs[-1]:>9.2e}")

    worst = max(diffs)
    limit = 1e-3 if args.dtype == "fp32" else 5e-2
    print(f"\nlargest difference: {worst:.2e} (limit for {args.dtype}: {limit:.0e})")
    print("MATCH: our LoRA behaves like PEFT." if worst < limit
          else "MISMATCH: check the scale, the forward pass and which parameters are trained.")


if __name__ == "__main__":
    main()
