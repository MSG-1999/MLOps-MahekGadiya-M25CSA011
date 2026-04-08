"""
q1/utils.py  —  Shared utilities: logging, param counting, grad norms.
"""

import os
import sys
import time

from tqdm import tqdm


# ── TeeLogger ─────────────────────────────────────────────────────────────────
class TeeLogger:
    """Writes every log line to both stdout and a file simultaneously."""

    def __init__(self, log_path: str):
        os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
        self.terminal = sys.stdout
        self.log_path = log_path
        self._file    = open(log_path, "w", buffering=1, encoding="utf-8")
        ts     = time.strftime("%Y-%m-%d %H:%M:%S")
        header = f"{'='*60}\nLog file : {log_path}\nStarted  : {ts}\n{'='*60}"
        self.terminal.write(header + "\n"); self.terminal.flush()
        self._file.write(header + "\n");    self._file.flush()

    def log(self, msg: str = ""):
        self.terminal.write(msg + "\n"); self.terminal.flush()
        self._file.write(msg + "\n");    self._file.flush()

    def close(self):
        footer = f"\n{'='*60}\nFinished : {time.strftime('%Y-%m-%d %H:%M:%S')}\n{'='*60}"
        self.log(footer)
        self._file.close()


# module-level logger reference (set by run_experiment)
_logger: TeeLogger = None


def set_logger(logger: TeeLogger):
    global _logger
    _logger = logger


def get_logger() -> TeeLogger:
    return _logger


def log(msg: str = ""):
    """tqdm-safe log that also writes to the active TeeLogger file."""
    if _logger is not None:
        tqdm.write(msg, file=_logger.terminal)
        _logger._file.write(msg + "\n")
        _logger._file.flush()
    else:
        tqdm.write(msg, file=sys.stdout)
    sys.stdout.flush()


# ── Model helpers ─────────────────────────────────────────────────────────────
def count_params(model):
    """Return (trainable_params, total_params)."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    return trainable, total


def lora_grad_norms(model) -> dict:
    """Return a dict of gradient norms for all LoRA A/B matrices."""
    d = {}
    for name, p in model.named_parameters():
        if ("lora_A" in name or "lora_B" in name) and p.grad is not None:
            d[f"grad/{name}"] = p.grad.norm().item()
    return d


# ── Startup banner ────────────────────────────────────────────────────────────
def print_banner(device: str, num_workers: int):
    import torch
    print("=" * 60)
    print(f"  Device      : {device}")
    if device == "cuda":
        print(f"  GPU         : {torch.cuda.get_device_name(0)}")
        print(f"  VRAM        : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
        print(f"  CUDA index  : {torch.cuda.current_device()}")
    print(f"  num_workers : {num_workers}")
    print("=" * 60)
