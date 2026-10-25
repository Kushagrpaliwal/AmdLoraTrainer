# AmdLoraTrainer
I wanna create optimized lora trainer for amd gpus .




```
AMDLoraTrainer/
├── src/amdlora/        ← the real project (all trainer code)
├── configs/            ← settings files
├── scripts/            ← helper scripts
├── tests/              ← automated checks
├── pyproject.toml, requirements.txt, README.md, .gitignore
└── amd_report_v2.json, feature_smoke_test.py, tunableop_results.csv  (yours)
```

### 1. `src/amdlora/`: the trainer itself

**Files directly inside it**


| File                            | What it is                                                                                                   | Status |
| ------------------------------- | ------------------------------------------------------------------------------------------------------------ | ------ |
| `__init__.py`                   | Marker file that tells Python “this folder is a package”. It also holds the version number.                  | ACTIVE |
| [`config.py`](http://config.py) | Reads your YAML settings into Python objects, so code can write `cfg.lora.rank`. It rejects misspelled keys. | ACTIVE |
| [`env.py`](http://env.py)       | Sets the AMD environment variables (attention, MIOpen, TunableOp, caches). It runs before PyTorch loads.     | ACTIVE |
| [`doctor.py`](http://doctor.py) | The smoke test. It checks the GPU, VRAM, BF16 matmul speed and attention backends.                           | ACTIVE |
| [`cli.py`](http://cli.py)       | The command entry point. It handles `doctor`, `train` and `merge`.                                           | ACTIVE |


**Subfolders.** Each has an `__init__.py` marker, and the other files are one-line placeholders.


| Folder     | Files                                   | Purpose                                                                            | Step |
| ---------- | --------------------------------------- | ---------------------------------------------------------------------------------- | ---- |
| `lora/`    | [`layer.py`](http://layer.py)           | The LoRA layer: a frozen weight plus small A and B matrices                        | 2    |
|            | [`inject.py`](http://inject.py)         | Finds the target layers in a model and swaps in LoRA layers                        | 3    |
| `models/`  | [`loader.py`](http://loader.py)         | Loads base models (BF16, FP8, 4-bit)                                               | 4    |
|            | [`registry.py`](http://registry.py)     | Which layers to target for each model family                                       | 3-4  |
| `data/`    | `llm_[dataset.py](http://dataset.py)`   | Text samples for LLM training                                                      | 5    |
|            | `image_[dataset.py](http://dataset.py)` | Image and caption pairs, sized into buckets                                        | 5    |
|            | [`cache.py`](http://cache.py)           | Precomputes text embeddings and image latents, then frees the encoder to save VRAM | 6    |
| `train/`   | [`loop.py`](http://loop.py)             | The training loop                                                                  | 7    |
|            | [`optim.py`](http://optim.py)           | Optimizer (AdamW and variants) and learning-rate schedule                          | 7    |
|            | [`memory.py`](http://memory.py)         | Gradient checkpointing, offload, VRAM saving                                       | 8    |
| `io/`      | [`save.py`](http://save.py)             | Saves only the small LoRA file                                                     | 9    |
|            | [`merge.py`](http://merge.py)           | Merges the LoRA into the base model                                                | 11   |
|            | [`gguf.py`](http://gguf.py)             | Converts to GGUF for llama.cpp (LLMs)                                              | 12   |
| `monitor/` | [`vram.py`](http://vram.py)             | VRAM and timing logs                                                               | 8    |
|            | [`hooks.py`](http://hooks.py)           | Looks inside the model: LoRA on vs off, attention maps                             | 10   |
| `ui/`      | (empty)                                 | The interface                                                                      | 13   |


All of these subfolders are LATER, except that they already exist so the structure is ready.

### 2. `configs/`: your settings

- `llm_example.yaml` is a ready-made settings file for LLM training: model path, dataset, rank, learning rate, precision, memory options.
- `diffusion_example.yaml` is the same for image models like FLUX.

You copy one, edit it, and pass it to `train`. **ACTIVE**, but only once training exists.

### 3. `scripts/`

- `install_windows.ps1` creates the virtual environment, installs the requirements and runs `doctor`. **ACTIVE**. The one thing still to fill in is the exact PyTorch ROCm install line.

### 4. `tests/`

- `test_[config.py](http://config.py)` checks that both YAML files load correctly.
- `test_[env.py](http://env.py)` checks that the AMD environment variables get set.

Run them with `pytest`. **ACTIVE**, and this is where the important tests will go later.

### 5. Root files

- `pyproject.toml` is the project’s ID card. It declares the name, Python version, dependencies and where the code lives (`src/`), and it enables `pip install -e .`.
- `requirements.txt` is the list of libraries to install.
- [`README.md`](http://README.md) is the short usage notes.
- `.gitignore` tells git to skip caches, outputs and model weights.

### 6. Your own extra files

I didn’t create these, and I don’t know their exact contents:

- `amd_report_v2.json` looks like a saved report from one of your own checks.
- `feature_smoke_[test.py](http://test.py)` looks like your own test script.
- `tunableop_results.csv` is the saved TunableOp tuning results from a run that wrote to the project root.

They can stay where they are. They don’t affect the trainer.

### 7. Created automatically when you run things

- `.cache/` holds the MIOpen and Triton caches (and TunableOp results if you run with `apply_amd_env()`).
- `outputs/` is where trained LoRAs and checkpoints will go.
- `.venv/` is your virtual environment.