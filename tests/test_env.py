import os
from amdlora.env import apply_amd_env

def test_env_set_before_torch(tmp_path):
    apply_amd_env(cache_root=str(tmp_path))
    assert os.environ["TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"] == "1"
    assert os.environ["TORCH_BLAS_PREFER_HIPBLASLT"] == "0"
