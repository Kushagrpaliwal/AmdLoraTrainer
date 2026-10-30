"""GPU smoke test: LoRA forward/backward on a random Llama, with timing and VRAM. No download needed."""
import argparse
import sys
import time

from amdlora.env import apply_amd_env

apply_amd_env()                      # must run before torch is imported

import torch  # noqa: E402
from amdlora.lora.inject import inject_lora, count_parameters  # noqa: E402
from amdlora.models.registry import default_targets  # noqa: E402


def build_model(path):
    if path:
        from amdlora.models.loader import load_llm
        model, _ = load_llm(path, base_load="bf16", device="cuda", with_tokenizer=False)
        return model
    from transformers import LlamaConfig, LlamaForCausalLM
    cfg = LlamaConfig(vocab_size=32000, hidden_size=1024, intermediate_size=2816,
                      num_hidden_layers=12, num_attention_heads=16, num_key_value_heads=16,
                      max_position_embeddings=1024)
    model = LlamaForCausalLM(cfg).to(dtype=torch.bfloat16, device="cuda")
    model.config.use_cache = False
    return model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=None, help="optional path to a real safetensors LLM folder")
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--seq", type=int, default=512)
    p.add_argument("--rank", type=int, default=16)
    args = p.parse_args()

    if not torch.cuda.is_available():
        print("FAIL: no GPU visible to PyTorch")
        return 1
    torch.manual_seed(0)
    print("GPU:", torch.cuda.get_device_name(0))

    model = build_model(args.model)
    inject_lora(model, default_targets("llm"), rank=args.rank, alpha=args.rank)
    trainable, total = count_parameters(model)
    print(f"params: total {total/1e6:.1f}M | trainable {trainable/1e6:.2f}M ({100*trainable/total:.2f}%)")
    model.train()

    ids = torch.randint(0, model.config.vocab_size, (args.batch, args.seq), device="cuda")
    params = [q for q in model.parameters() if q.requires_grad]
    opt = torch.optim.AdamW(params, lr=1e-3)

    def step():
        opt.zero_grad(set_to_none=True)
        loss = model(input_ids=ids, labels=ids).loss
        loss.backward()
        opt.step()
        return loss.item()

    try:
        first = step()
        for _ in range(2):
            step()                               # warm-up (kernels compile/tune here)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        t = time.perf_counter()
        last = first
        for _ in range(args.steps):
            last = step()
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t) / args.steps
    except torch.cuda.OutOfMemoryError:
        print("OOM: try --batch 1 or --seq 256")
        return 1

    ok_grad = all(q.grad is not None and torch.isfinite(q.grad).all().item() for q in params)
    peak = torch.cuda.max_memory_allocated() / 2**30
    print(f"loss: {first:.3f} -> {last:.3f} ({'decreasing' if last < first else 'NOT decreasing'})")
    print(f"gradients finite on all LoRA weights: {ok_grad}")
    print(f"step time {dt*1e3:.0f} ms | {args.batch*args.seq/dt:,.0f} tokens/s | peak VRAM {peak:.2f} GiB")
    return 0


if __name__ == "__main__":
    sys.exit(main())