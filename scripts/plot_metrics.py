"""Step 10: draw the loss curves of a training run.

Usage (from the project root):
    python scripts/plot_metrics.py outputs/smollm_test
Writes loss.png into the same folder.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                     # draw straight to a file, no window
import matplotlib.pyplot as plt  # noqa: E402


def read_rows(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def smooth(values: list[float], alpha: float) -> list[float]:
    """Exponential moving average: each point keeps most of the previous average."""
    out, average = [], None
    for v in values:
        average = v if average is None else alpha * v + (1 - alpha) * average
        out.append(average)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", help="output folder of a run, e.g. outputs/smollm_test")
    parser.add_argument("--smooth", type=float, default=0.1, help="0..1, smaller = smoother")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    rows = read_rows(run_dir / "metrics.jsonl")
    train = [r for r in rows if r.get("kind") == "train"]
    val = [r for r in rows if r.get("kind") == "val"]
    if not train:
        raise SystemExit(f"No training rows in {run_dir / 'metrics.jsonl'}")

    steps = [r["step"] for r in train]
    losses = [r["loss"] for r in train]

    fig, (ax_loss, ax_lr) = plt.subplots(2, 1, figsize=(10, 7), sharex=True, height_ratios=[3, 1])
    ax_loss.plot(steps, losses, color="tab:blue", alpha=0.25, linewidth=1, label="train loss")
    ax_loss.plot(steps, smooth(losses, args.smooth), color="tab:blue", linewidth=2,
                 label="train loss (smoothed)")

    if val:
        ax_loss.plot([r["step"] for r in val], [r["val_loss"] for r in val], "o-",
                     color="tab:orange", linewidth=2, label="val loss")
        trained = [r for r in val if r["step"] > 0]
        if trained:
            best = min(trained, key=lambda r: r["val_loss"])
            ax_loss.scatter([best["step"]], [best["val_loss"]], s=220, marker="*", color="tab:red",
                            zorder=5, label=f"best val {best['val_loss']:.4f} @ step {best['step']}")
            print(f"baseline val_loss {val[0]['val_loss']:.4f} | "
                  f"best {best['val_loss']:.4f} at step {best['step']} | "
                  f"last {val[-1]['val_loss']:.4f} at step {val[-1]['step']}")

    ax_loss.set_ylabel("loss")
    ax_loss.set_title(run_dir.name)
    ax_loss.grid(alpha=0.3)
    ax_loss.legend()

    ax_lr.plot(steps, [r["lr"] for r in train], color="tab:green")
    ax_lr.set_ylabel("learning rate")
    ax_lr.set_xlabel("update")
    ax_lr.grid(alpha=0.3)

    fig.tight_layout()
    out = run_dir / "loss.png"
    fig.savefig(out, dpi=120)
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
