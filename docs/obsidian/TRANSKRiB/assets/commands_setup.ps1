# TRANSKRiB setup commands (PowerShell)
# Use as a runbook. Replace <REPO_ID> with actual Hugging Face repo IDs.

# ---------- Converter venv (Python 3.13) ----------
cd D:\TRANSKRiB\CONVERTOR
# Create venv (example):
# py -V:3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -U pip setuptools wheel
.\.venv\Scripts\python.exe -m pip install -U watchdog psutil

# ---------- TTGP venv (Python 3.11 + CUDA cu118) ----------
cd D:\TRANSKRiB\TTGP
# Create venv (example):
# py -V:3.11 -m venv venv
.\venv\Scripts\python.exe -m pip install -U pip setuptools wheel

# PyTorch cu118 wheels:
.\venv\Scripts\python.exe -m pip install --index-url https://download.pytorch.org/whl/cu118 `
 torch==2.5.1+cu118 torchaudio==2.5.1+cu118 torchvision==0.20.1+cu118

# Install the rest (pinned):
# .\venv\Scripts\python.exe -m pip install -r .\requirements_ttgp_py311_cu118.txt

# Verify CUDA:
.\venv\Scripts\python.exe -c "import torch; print(torch.__version__); print('cuda', torch.cuda.is_available())"

# ---------- Model downloads (replace <REPO_ID>) ----------
.\venv\Scripts\python.exe -m pip install -U huggingface-hub
# .\venv\Scripts\python.exe -m huggingface_hub.cli download <REPO_ID> --local-dir D:\TRANSKRiB\LargeV3 --local-dir-use-symlinks False
# .\venv\Scripts\python.exe -m huggingface_hub.cli download <REPO_ID> --local-dir D:\TRANSKRiB\ARH\turbo --local-dir-use-symlinks False
# .\venv\Scripts\python.exe -m huggingface_hub.cli download <REPO_ID> --local-dir D:\TRANSKRiB\ARH\TurboEnRu3.2 --local-dir-use-symlinks False
