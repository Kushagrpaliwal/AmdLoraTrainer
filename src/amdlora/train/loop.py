from __future__ import annotations
import math
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ..data.llm_dataset import load_texts, split_texts, TextDataset, make_collate
from ..io.save import save_lora
from ..lora.inject import inject_lora, count_parameters
from ..models.loader import load_base
from ..models.registry import default_targets
from ..monitor.metrics import MetricsLogger, save_config_copy
from ..monitor.samples import write_samples
from ..monitor.vram import peak_vram_gib, reset_peak_vram
from .evaluate import evaluate
from .memory import apply_memory_settings
from .optim import build_optimizer, make_scheduler


def count_total_updates(num_batches: int, cfg) -> int:
    t = cfg.train
    updates_per_epoch = math.ceil(num_batches / t.grad_accum)
    total = updates_per_epoch * t.epochs
    if t.max_steps > 0:
        total = min(total, t.max_steps)
    return total


def speed_numbers(recent_losses, tokens, seconds, updates_in_window, on_gpu) -> dict:
    numbers = {
        "avg_loss": sum(recent_losses) / len(recent_losses),
    if t.max_steps > 0:
        total = min(total, t.max_steps)
    return total


def speed_numbers(recent_losses, tokens, seconds, updates_in_window, on_gpu) -> dict:
    numbers = {
        "avg_loss": sum(recent_losses) / len(recent_losses),
        "tok_s": tokens / max(seconds, 1e-9),
        "ms_update": seconds / max(updates_in_window, 1) * 1000,     # AMD SPEED
    }
    if on_gpu:
        numbers["vram_gb"] = peak_vram_gib()
    return numbers


def print_progress(update, total, epoch, epochs, learning_rate, numbers) -> None:
    vram = f" | peak VRAM {numbers['vram_gb']:.2f} GB" if "vram_gb" in numbers else ""
    print(f"step {update}/{total} | epoch {epoch}/{epochs} | loss {numbers['avg_loss']:.4f} | "
          f"lr {learning_rate:.2e} | {numbers['tok_s']:,.0f} tok/s | "
          f"{numbers['ms_update']:,.0f} ms/update{vram}")


def run_training(model, loader, cfg, out_dir, device, val_loader=None, tokenizer=None) -> list[float]:
    t, mon = cfg.train, cfg.monitor
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- setup ----
    lora_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = build_optimizer(lora_params, t.optimizer, t.lr)
    total_updates = count_total_updates(len(loader), cfg)
    scheduler = make_scheduler(optimizer, t.warmup_steps, total_updates)
    apply_memory_settings(model, cfg)
    model.train()

    # AMD SPEED: run forward passes through `runner`; always save from the original `model`
    # (a compiled wrapper would rename the LoRA keys inside the saved file).
    runner = torch.compile(model) if cfg.amd.compile else model

    on_gpu = torch.device(device).type == "cuda"
    reset_peak_vram()

    # ---- STEP 10: run records ----
    save_config_copy(cfg, out_dir)
    logger = MetricsLogger(out_dir)
    samples_path = out_dir / "samples.md"
    samples_path.unlink(missing_ok=True)
    want_samples = bool(mon.sample_every > 0 and mon.sample_prompts and tokenizer is not None)
    best_val_loss, best_step = math.inf, 0

    def validate(step: int) -> float:
        """Validation loss with the plain model. Step 0 is the baseline (LoRA B is still zero,
        so this equals the base model). Returns the seconds it took."""
        nonlocal best_val_loss, best_step
        start = time.perf_counter()
        val_loss = evaluate(model, val_loader, device)
        logger.log(kind="val", step=step, val_loss=val_loss)
        note = ""
        if step == 0:
            note = " (baseline, before training)"
        elif val_loss < best_val_loss:
            best_val_loss, best_step = val_loss, step
            note = " (best so far)"
            if mon.keep_best:
                save_lora(model, out_dir / "lora_best.safetensors")
        print(f"step {step}/{total_updates} | val_loss {val_loss:.4f}{note}")
        return time.perf_counter() - start

    def sample(step: int) -> float:
        start = time.perf_counter()
        write_samples(model, tokenizer, mon.sample_prompts, mon.sample_max_tokens, samples_path, step)
        print(f"step {step}/{total_updates} | samples written to {samples_path}")
        return time.perf_counter() - start

    # ---- training ----
    loss_history: list[float] = []
    loss_sum = 0.0            # unscaled micro-batch losses since the last optimizer step
    batches_in_update = 0
    updates_done = 0
    updates_since_log = 0
    tokens_since_log = 0
    finished = False

    try:
        if val_loader is not None:
            validate(0)
        if want_samples:
            sample(0)
        log_timer_start = time.perf_counter()

        for epoch in range(1, t.epochs + 1):
            for batch_index, batch in enumerate(loader):
                batch = {name: tensor.to(device) for name, tensor in batch.items()}

                loss = runner(**batch).loss
                (loss / t.grad_accum).backward()

                loss_sum += loss.item()
                tokens_since_log += int(batch["attention_mask"].sum())
                batches_in_update += 1
                is_last_batch = batch_index == len(loader) - 1

                if batches_in_update % t.grad_accum != 0 and not is_last_batch:
                    continue

                torch.nn.utils.clip_grad_norm_(lora_params, t.max_grad_norm)
                lr_used = optimizer.param_groups[0]["lr"]      # the rate this update really uses
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                loss_history.append(loss_sum / batches_in_update)
                loss_sum, batches_in_update = 0.0, 0
                updates_done += 1
                updates_since_log += 1

                # STEP 10: one metrics line per update; speed numbers only on logging updates
                row = {"kind": "train", "step": updates_done, "epoch": epoch,
                       "loss": loss_history[-1], "lr": lr_used}
                if updates_done % t.log_every == 0 or updates_done == total_updates:
                    numbers = speed_numbers(loss_history[-t.log_every:], tokens_since_log,
                                            time.perf_counter() - log_timer_start,
                                            updates_since_log, on_gpu)
                    print_progress(updates_done, total_updates, epoch, t.epochs, lr_used, numbers)
                    row.update(numbers)
                    tokens_since_log, updates_since_log = 0, 0
                    log_timer_start = time.perf_counter()
                logger.log(**row)

                # STEP 10: moving the timer start forward keeps validation and sample time
                # out of tok/s and ms/update
                if val_loader is not None and mon.val_every > 0 and updates_done % mon.val_every == 0:
                    log_timer_start += validate(updates_done)
                if want_samples and updates_done % mon.sample_every == 0:
                    log_timer_start += sample(updates_done)

                # STEP 10: moving the timer start forward keeps validation and sample time
                # out of tok/s and ms/update
                if val_loader is not None and mon.val_every > 0 and updates_done % mon.val_every == 0:
                    log_timer_start += validate(updates_done)
                if want_samples and updates_done % mon.sample_every == 0:
                    log_timer_start += sample(updates_done)

                if val_loader is not None and mon.val_every > 0 and updates_done % mon.val_every == 0:
                    log_timer_start += validate(updates_done)
                if want_samples and updates_done % mon.sample_every == 0:
                    log_timer_start += sample(updates_done)

                if t.save_every > 0 and updates_done % t.save_every == 0:
                    save_lora(model, out_dir / f"lora_step{updates_done}.safetensors")

                if updates_done >= total_updates:
                    finished = True
                    break

            if finished:
                break

        # STEP 10: final validation and samples, unless they just ran on the last update
        if val_loader is not None and (mon.val_every == 0 or updates_done % mon.val_every != 0):
            validate(updates_done)
        if want_samples and updates_done % mon.sample_every != 0:
            sample(updates_done)
    finally:
        logger.close()

    # ---- final save ----
    final_path = out_dir / "lora_final.safetensors"
    save_lora(model, final_path)
    print(f"saved: {final_path}")
    if best_step:
        kept = " -> lora_best.safetensors" if mon.keep_best else ""
        print(f"best val_loss {best_val_loss:.4f} at step {best_step}{kept}")
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

    # STEP 10: split before tokenizing, so validation samples never reach training
    train_texts, val_texts = split_texts(load_texts(cfg.dataset_dir), cfg.monitor.val_split, cfg.train.seed)
    dataset = TextDataset(train_texts, tokenizer, cfg.train.max_len)
    collate = make_collate(tokenizer.pad_token_id, cfg.train.pad_to)

    # AMD SPEED: with pad_to set, every batch has the exact same shape. The last partial
    # batch is dropped so that no odd shape appears (shuffling still covers every sample).
    loader = DataLoader(
        dataset,
        batch_size=cfg.train.micro_batch_size,
        shuffle=True,
        collate_fn=collate,
        num_workers=0,
        drop_last=cfg.train.pad_to > 0,
    )
    if len(loader) == 0:
        raise ValueError("No batches: the dataset is smaller than micro_batch_size "
                         "(with pad_to set, the last partial batch is dropped).")
    if cfg.amd.compile and cfg.train.pad_to == 0:
        print("warning: amd.compile with variable-length batches recompiles often. Set train.pad_to.")

    # STEP 10: validation keeps every sample (no drop_last) in a fixed order
    val_loader = None
    if val_texts:
        val_loader = DataLoader(
            TextDataset(val_texts, tokenizer, cfg.train.max_len),
            batch_size=cfg.train.micro_batch_size,
            shuffle=False,
            collate_fn=collate,
            num_workers=0,
            drop_last=False,
        )

    print(f"samples: train {len(dataset)} | val {len(val_texts)} | batches per epoch: {len(loader)}")

    run_training(model, loader, cfg, cfg.output_dir, device, val_loader=val_loader, tokenizer=tokenizer)
    return 0
