from __future__ import annotations
import argparse
import sys


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="amdlora")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("doctor", help="environment smoke test")
    t = sub.add_parser("train", help="train a LoRA from a YAML config")
    t.add_argument("config")
    m = sub.add_parser("merge", help="merge a LoRA into a base model (Step 11)")
    m.add_argument("config")
    args = p.parse_args(argv)

    if args.cmd == "doctor":
        from .doctor import run_doctor
        return run_doctor()
    from .config import load_config
    from .env import apply_amd_env
    cfg = load_config(args.config)
    apply_amd_env(cfg.amd)                 # before torch is imported anywhere
    if args.cmd == "train":
        from .train.loop import train
        return train(cfg)
    from .io.merge import merge
    return merge(cfg)


if __name__ == "__main__":
    sys.exit(main())
