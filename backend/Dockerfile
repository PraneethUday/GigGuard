# syntax=docker/dockerfile:1
FROM python:3.10-slim

# --- System dependencies -----------------------------------------------
# libgomp1: OpenMP runtime required by xgboost's compiled wheel at import
# time (python:3.10-slim does not ship it, and xgboost fails with
# "libgomp.so.1: cannot open shared object file" without it).
# No compiler toolchain (build-essential) is needed — every package in
# requirements.txt (numpy/pandas/scikit-learn/xgboost/torch/transformers)
# ships manylinux wheels for this base image, so keep the image lean.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# --- Python dependencies (separate layer for build-cache reuse) --------
COPY requirements.txt .
# torch is installed from PyTorch's dedicated CPU-only wheel index first —
# plain `pip install torch` on Linux pulls the full CUDA/GPU build (adds
# several GB of nvidia-cublas/cudnn/cufft/nccl libraries) even though
# ml/curfew.py only ever runs it CPU-bound (device=-1) and this container
# has no GPU. Installing the CPU wheel here first satisfies the
# `torch>=2.2.0` pin in requirements.txt below, so pip leaves it alone.
RUN pip install --no-cache-dir torch --extra-index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

# --- Application source --------------------------------------------------
COPY . .

# --- Run as non-root ------------------------------------------------------
RUN useradd --create-home --uid 1000 appuser \
    && chown -R appuser:appuser /app
USER appuser

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    OMP_NUM_THREADS=1 \
    TOKENIZERS_PARALLELISM=false

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2)" || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
