# ─── Dockerfile ───────────────────────────────────────────────
# DLOps Assignment 5 — ViT-S LoRA Fine-tuning on CIFAR-100
# Base : PyTorch 2.1 · CUDA 11.8 · cuDNN 8
# ──────────────────────────────────────────────────────────────
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

LABEL maintainer="MSG1999"
LABEL description="DLOps A5 Q1: ViT-S LoRA on CIFAR-100"

WORKDIR /workspace

# ── system deps ───────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget curl ca-certificates \
    libglib2.0-0 libsm6 libxext6 libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# ── Python deps ───────────────────────────────────────────────
RUN pip install --upgrade pip
COPY requirements_q1.txt .
RUN pip install --no-cache-dir -r requirements_q1.txt

# ── copy source ───────────────────────────────────────────────
COPY q1/             q1/
COPY q1_train.py     .
COPY q1_optuna.py    .
COPY q1_push.py      .

# ── create output dirs ────────────────────────────────────────
RUN mkdir -p weights/BEST_MODEL plots logs data

# ── env ───────────────────────────────────────────────────────
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/workspace

# default: show help
CMD ["python", "-c", "print('Container ready. Run q1_train.py / q1_optuna.py / q1_push.py')"]
