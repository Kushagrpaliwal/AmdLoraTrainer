"""Step 12: convert the merged model to GGUF (BF16, Q8_0, Q4_K_M) for llama.cpp.

Usage (from the project root):
    python scripts/make_gguf.py --llama-cpp E:\\tools\\llama.cpp
(or set the LLAMA_CPP_DIR environment variable and leave the option out)
"""
import argparse
import os
from pathlib import Path

from amdlora.io.gguf import convert_to_gguf, inspect_gguf

PROMPT = "Question: What is a bicycle?\nAnswer:"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="outputs/smollm_test/merged")
    parser.add_argument("--out-dir", default="outputs/smollm_test/gguf")
    parser.add_argument("--llama-cpp", default=os.environ.get("LLAMA_CPP_DIR"),
                        help="folder of your llama.cpp clone")
    parser.add_argument("--quantize-exe", default=None, help="path to llama-quantize(.exe), if not on PATH")
    parser.add_argument("--name", default="smollm2-pirate")
    parser.add_argument("--quants", default="Q8_0,Q4_K_M")
    args = parser.parse_args()
    if not args.llama_cpp:
        parser.error("give --llama-cpp <folder> or set the LLAMA_CPP_DIR environment variable")

    quants = [q for q in args.quants.split(",") if q]
    files = convert_to_gguf(args.model_dir, args.out_dir, args.llama_cpp, args.name,
                            quants, args.quantize_exe)

    print("\nResult:")
    for path in files:
        try:
            info = inspect_gguf(path, args.llama_cpp)
            print(f"  {path.name:<34} {info['size_gb']:.2f} GB | arch {info['architecture']} | "
                  f"{info['tensors']} tensors | {info['types']}")
        except Exception as error:
            print(f"  {path.name:<34} written ({path.stat().st_size / 1e9:.2f} GB), "
                  f"header check skipped: {type(error).__name__}")

    prompt_file = Path(args.out_dir) / "prompt.txt"
    prompt_file.write_text(PROMPT, encoding="utf-8")
    print("\nTry it with llama.cpp (llama-cli must be installed):")
    print(f"  llama-cli -m {files[-1]} -f {prompt_file} -n 40 --temp 0")
    print("If it opens a chat prompt instead of answering, press Ctrl+C and add -no-cnv.")


if __name__ == "__main__":
    main()