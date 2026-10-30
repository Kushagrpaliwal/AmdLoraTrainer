"""Step 4: one loader interface; per-family adapters (llm via transformers, diffusion via diffusers). bf16 / fp8 / nf4."""
from __future__ import annotations
import torch

DTYPES = {"fp32":torch.float32 , "fp16":torch.float16 , "bf16": torch.bfloat16}
ALLOWED = set(DTYPES) | {"nf4","fp8"}

def _check_precision(base_load:str):
  if base_load not in ALLOWED:
    raise ValueError(f"Unknown base load '{base_load}'. Choose from {sorted(ALLOWED)}")
  if base_load == "fp8":
    raise NotImplementedError("fp8 arrives in the later steps") 

def _from_pretrained(
  cls,
  path:str,
  **kwargs
  ):
  dtype = kwargs.pop("dtype",None)
  if dtype is None:
    return cls.from_pretrained(path,**kwargs)
  try:
    return cls.from_pretrained(path,dtype=dtype,**kwargs)
  except TypeError:
    return cls.from_pretrained(path,torch_dtype=dtype,**kwargs)

def load_llm(
  path:str,
  base_load:str = "bf16" , 
  device:str="cuda" , 
  with_tokenizer:bool=True
):

  _check_precision(base_load)
  from transformers import AutoModelForCausalLM, AutoTokenizer

  kwargs = {}
  if base_load == "nf4":
    try:
      from transformers import BitsAndBytesConfig
      import bitsandbytes
    except ImportError as e:
      raise RuntimeError("nf4 needs bitsandbytes, which is experimental on amd/Windows . Use base_load b16 instead ") from e
  
    kwargs["quantization_config"] = BitsAndBytesConfig(
      load_in_4bit=True, 
      bnb_4bit_quant_type="nf4" , 
      bnb_4bit_compute_dtype = torch.bfloat16
    )

    kwargs["device_map"] = {"":0}

  else:
      kwargs["dtype"] = DTYPES[base_load]

  model = _from_pretrained(AutoModelForCausalLM,path,**kwargs)

  if base_load != "nf4":
    model.to(device)
    model.config.use_cache = False

  tokenizer = None

  if with_tokenizer:
    tokenizer = AutoTokenizer.from_pretrained(path)
    if tokenizer.pad_token is None:
      tokenizer.pad_token = tokenizer.eos_token

  return model , tokenizer

def load_diffusion(path:str , base_load:str = "bf16" , device: str = "cuda"):
  _check_precision(base_load)
  raise NotImplementedError("Diffusion model waits until it is fully confirmed from the hugging face")

def load_base(cfg):
  if cfg.model_family == "llm":
    return load_llm(cfg.base_model , cfg.precision.base_load)
  if cfg.model_family == "diffusion":
    return load_diffusion(cfg.base_load , cfg.precision.base_load)
  
  raise ValueError(f"Unknown model family '{cfg.model_family}'")



