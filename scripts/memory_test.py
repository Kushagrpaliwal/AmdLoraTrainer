"""Step 8: measure how much VRAM and time gradient checkpointing saves on YOUR GPU.

Usage (from the project root):
    python scripts/memory_test.py

It builds a random ~1B-parameter Llama (nothing to download), then trains a few steps
at several batch sizes and sequence lengths, with checkpointing OFF and then ON.
A case that does not fit in memory is reported as OOM instead of crashing.
"""
import gc

from amdlora.env import apply_amd_env

apply_amd_env()                                   # must run before torch is imported

import torch  # noqa: E402
from transformers import LlamaConfig, LlamaForCausalLM  # noqa: E402
from amdlora.lora.inject import inject_lora, count_parameters  # noqa: E402
from amdlora.models.registry import default_targets  # noqa: E402
from amdlora.monitor.vram import GpuTimer, peak_vram_gib, reset_peak_vram, vram_summary  # noqa: E402
from amdlora.train.memory import set_gradient_checkpointing  # noqa: E402

VOCAB = 32000
CASES = [(1, 512), (4, 512), (1, 2048), (4, 2048)]      # (batch size, sequence length)


def build_model(layers: int = 16):
    config = LlamaConfig(vocab_size=VOCAB, hidden_size=2048, intermediate_size=5632,
                         num_hidden_layers=layers, num_attention_heads=16,
                         num_key_value_heads=16, max_position_embeddings=4096)
    model = LlamaForCausalLM(config).to(dtype=torch.bfloat16, device="cuda")
    model.config.use_cache = False
    return model


def measure(model, optimizer, batch: int, seq: int, steps: int = 3):
    """Train a few steps and return (peak GiB, ms per step, tokens/s), or None if out of memory."""
    ids = torch.randint(0, VOCAB, (batch, seq), device="cuda")

    def one_step():
        optimizer.zero_grad(set_to_none=True)
        loss = model(input_ids=ids, labels=ids).loss
        loss.backward()
        optimizer.step()

    try:
        one_step()                                # warm-up, not timed
        reset_peak_vram()
        with GpuTimer() as timer:
            for _ in range(steps):
                one_step()
        return peak_vram_gib(), timer.seconds / steps * 1000, batch * seq * steps / timer.seconds
    except torch.cuda.OutOfMemoryError:
        return None
    finally:
        optimizer.zero_grad(set_to_none=True)
        gc.collect()
        torch.cuda.empty_cache()


def main():
    if not torch.cuda.is_available():
        print("FAIL: no GPU visible to PyTorch")
        return 1
    torch.manual_seed(0)
    print("GPU:", torch.cuda.get_device_name(0))

    model = build_model()
    inject_lora(model, default_targets("llm"), rank=16, alpha=16)
    trainable, total = count_parameters(model)
    print(f"model: {total / 1e9:.2f}B params (BF16) | trainable LoRA: {trainable / 1e6:.1f}M")
    print("after loading:", vram_summary())
    model.train()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)

    print()
    print(f"{'batch':>5} {'seq':>5} {'checkpointing':>14} {'peak VRAM':>11} {'step time':>10} {'tokens/s':>9}")
    for checkpointing in (False, True):
        set_gradient_checkpointing(model, checkpointing)
        for batch, seq in CASES:
            result = measure(model, optimizer, batch, seq)
            label = "ON" if checkpointing else "OFF"
            if result is None:
                print(f"{batch:>5} {seq:>5} {label:>14} {'OUT OF MEMORY':>33}")
            else:
                peak, ms, tok_s = result
                print(f"{batch:>5} {seq:>5} {label:>14} {peak:>8.2f} GiB {ms:>7.0f} ms {tok_s:>9,.0f}")

    print("\nHow to read this: for the same batch and seq, compare OFF vs ON.")
    print("ON should use clearly less VRAM and run somewhat slower. A case that is OOM with")
    print("checkpointing OFF but fits with ON is exactly why the switch exists.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())