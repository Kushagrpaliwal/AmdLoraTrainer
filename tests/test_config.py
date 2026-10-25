from amdlora.config import load_config
import pathlib

def test_example_configs_load():
    for name in ["llm_example.yaml", "diffusion_example.yaml"]:
        cfg = load_config(pathlib.Path(__file__).parents[1] / "configs" / name)
        assert cfg.lora.rank == 16
        assert cfg.precision.compute_dtype == "bf16"
