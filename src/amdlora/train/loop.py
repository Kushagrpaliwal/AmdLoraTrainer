"""Step 7: training loop (next-token CE for LLM, noise/flow loss for diffusion)."""
from __future__ import annotations
import math
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ..data.llm_dataset import load_texts, TextDataset, make_collate
from ..io.save import save_lora
from ..lora.inject import inject_lora, count_parameters
from ..models.loader import load_base
from ..models.registry import default_targets
from ..monitor.vram import peak_vram_gib , reset_peak_vram
from .memory import apply_memory_settings
from .optim import build_optimizer, make_scheduler


def count_total_updates(num_batches: int, cfg) -> int:
    t = cfg.train
    updates_per_epoch = math.ceil(num_batches / t.grad_accum)
    total = updates_per_epoch * t.epochs
    if t.max_steps > 0:
        total = min(total, t.max_steps)
    return total


def print_progress(update, total, epoch, epochs, recent_losses, learning_rate, tokens, seconds, on_gpu) -> None:
    average_loss = sum(recent_losses) / len(recent_losses)
    vram = ""
    if on_gpu:
        vram = f" | peak VRAM {peak_vram_gib():.2f} GB"
    print(f"step {update}/{total} | epoch {epoch}/{epochs} | loss {average_loss:.4f} | "
          f"Lr {learning_rate:.2e} | {tokens / max(seconds, 1e-9):,.0f} tok/s{vram}")


def run_training(model, loader, cfg, out_dir, device) -> list[float]:
    t = cfg.train
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- setup ----
    lora_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = build_optimizer(lora_params, t.optimizer, t.lr)
    total_updates = count_total_updates(len(loader), cfg)
    scheduler = make_scheduler(optimizer, t.warmup_steps, total_updates)
    apply_memory_settings(model,cfg)
    model.train()

    on_gpu = torch.device(device).type == "cuda"
    reset_peak_vram()

    # ---- training ----
    loss_history: list[float] = []
    loss_sum = 0.0            # unscaled micro-batch losses since the last optimizer step
    batches_in_update = 0
    updates_done = 0
    tokens_since_log = 0
    log_timer_start = time.perf_counter()
    finished = False

    for epoch in range(1, t.epochs + 1):
        for batch_index, batch in enumerate(loader):
            batch = {name: tensor.to(device) for name, tensor in batch.items()}

            loss = model(**batch).loss
            (loss / t.grad_accum).backward()

            loss_sum += loss.item()
            tokens_since_log += int(batch["attention_mask"].sum())
            batches_in_update += 1
            is_last_batch = batch_index == len(loader) - 1

            if batches_in_update % t.grad_accum != 0 and not is_last_batch:
                continue

            torch.nn.utils.clip_grad_norm_(lora_params, t.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            loss_history.append(loss_sum / batches_in_update)
            loss_sum, batches_in_update = 0.0, 0
            updates_done += 1

            if updates_done % t.log_every == 0 or updates_done == total_updates:
                print_progress(
                    updates_done,
                    total_updates,
                    epoch,
                    t.epochs,
                    loss_history[-t.log_every:],
                    scheduler.get_last_lr()[0],
                    tokens_since_log,
                    time.perf_counter() - log_timer_start,
                    on_gpu,
                )
                tokens_since_log, log_timer_start = 0, time.perf_counter()

            if t.save_every > 0 and updates_done % t.save_every == 0:
                save_lora(model, out_dir / f"lora_step{updates_done}.safetensors")

            if updates_done >= total_updates:
                finished = True
                break

        if finished:
            break

    # ---- final save ----
    final_path = out_dir / "lora_final.safetensors"
    save_lora(model, final_path)
    print(f"saved: {final_path}")
    return loss_history


def train(cfg) -> int:
    if cfg.model_family != "llm":
        raise NotImplementedError("Diffusion model training will be done in upcoming steps")

    torch.manual_seed(cfg.train.seed)

    model, tokenizer = load_base(cfg)
    device = next(model.parameters()).device

    targets = cfg.lora.target_modules or default_targets(cfg.model_family)
    inject_lora(model, targets, cfg.lora.rank, cfg.lora.alpha, cfg.lora.dropout)
    trainable, total = count_parameters(model)
    print(f"model: {total / 1e6:.1f}M params | trainable(LoRA): {trainable / 1e6:.2f}M | device: {device}")

    dataset = TextDataset(load_texts(cfg.dataset_dir), tokenizer, cfg.train.max_len)

    loader = DataLoader(
        dataset,
        batch_size=cfg.train.micro_batch_size,
        shuffle=True,
        collate_fn=make_collate(tokenizer.pad_token_id),
        num_workers=0,
    )

    print(f"samples: {len(dataset)} | batches per epoch: {len(loader)}")

    run_training(model, loader, cfg, cfg.output_dir, device)
    return 0
