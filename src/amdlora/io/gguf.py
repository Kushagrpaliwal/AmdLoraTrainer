"""Step 12: llama.cpp conversion wrapper (LLM only)."""
from __future__ import annotations
import os 
import shutil
import sys
import subprocess
from pathlib import Path
from collections import Counter

EXE = ".exe" if os.name == 'nt' else ""

def find_convert_script(llama_cpp_dir) -> Path:
  script = Path(llama_cpp_dir) / "convert_hf_to_gguf.py"
  if not script.exists():
    raise FileNotFoundError(f"{script} no found. Point --llama-cpp at your clone of llama.cpp repository. ")  
  return script

def find_quantize_command(llama_cpp_dir, explicit = None) -> list[str]:
  candidates = []
  if explicit:
    candidates.append(Path(explicit))
  if os.environ.get("LLAMA_QUANTIZE"):
    candidates.append(Path(os.environ["LLAMA_QUANTIZE"]))
  
  on_path = shutil.which("llama-quantize")
  
  if on_path:
    candidates.append(Path(on_path))
  
  root = Path(llama_cpp_dir)
  
  for sub in ("build/bin/Release","build/bin" , "."):
    candidates.append(root / sub / f"llama-quantize{EXE}")
  
  for path in candidates:
    if path.exists():
      return [str(path)]
  
  raise FileNotFoundError(f"llama-quantize not found . Install llama.cpp (for example: winget install llama.cpp) or download a release zip from github , then pass --quantize-exe <path to llama-quantize.exe>.")


def _run(command: list[str], env: dict, what: dict) -> None:
  result = subprocess.run(command,env=env)
  if result.returncode != 0:
    raise RuntimeError(f"{what} failed ( exit code {result.returncode}) . See the messages above. ")

def convert_to_gguf(model_dir,out_dir,llama_cpp_dir,name="model",quants=("Q8_0","Q4_K_M") , quantize_exe=None) -> list[Path]:
 
  model_dir, out_dir, llama_dir = Path(model_dir) , Path(out_dir) , Path(llama_cpp_dir)
  
  if not (model_dir / "config.json").exists():
    raise FileNotFoundError(f"{model_dir} is not a model folder (no config.json). Run the merge step first")

  convert_script = find_convert_script(llama_dir)
  quantize_command = find_quantize_command(llama_dir,quantize_exe) if quants else None

  env = os.environ.copy()
  env["PYTHONPATH"] = str(llama_dir / "gguf-py") + os.pathsep + env.get("PYTHONPATH" , "")

  out_dir.mkdir(parents=True, exist_ok=True)
  full_precision = out_dir / f"{name}-bf16.gguf"
  _run([sys.executable, str(convert_script) , str(model_dir), "--outfile", str(full_precision), "--outtype" , "bf16"] , env, "GGUF conversion")

  written = [full_precision]
  for quant in quants:
    target = out_dir / f"{name}-{quant}.gguf"
    _run([*quantize_command, str(full_precision) , str(target) , quant] , env, f"quantize {quant}")
    written.append(target) 
  return written


def inspect_gguf(path,llama_cpp_dir=None) -> dict:
  path = Path(path)
  if llama_cpp_dir:
    sys.path.insert(0,str(Path(llama_cpp_dir) / "gguf-py"))
  import gguf

  reader = gguf.GGUFReader(str(path))
  field = reader.fields.get("general.architecture")
  architecture = bytes(field.parts[field.data[0]]).decode("utf-8") if field is not None else None
  types = Counter(tensor.tensor_type.name for tensor in reader.tensors)
  return {"architecture": architecture, "tensors": len(reader.tensors), "types": dict(types), "size_gb": path.stat().st_size / 1e9}






