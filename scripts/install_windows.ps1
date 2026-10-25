# Step 1 install helper. Fill in the exact wheel source from AMD's ROCm 10.1 Windows docs.
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
# TODO: install torch 2.14.0+rocm10.1.0 from the AMD/PyTorch ROCm index you verified
# pip install torch==2.14.0 --index-url <ROCM_10_1_INDEX_URL>
pip install -r requirements.txt
pip install -e .
python -m amdlora.cli doctor
