"""
q1/plots.py  —  All plotting helpers (curves, classwise, comparison, heatmap).

Every function saves a PNG to PLOTS_DIR and returns the file path so callers
can pass it straight to wandb.Image().
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import torch

from q1.config import (
    ALPHAS, CIFAR100_CLASSES, CIFAR100_MEAN, CIFAR100_STD,
    NUM_CLASSES, PLOTS_DIR, RANKS,
)

os.makedirs(PLOTS_DIR, exist_ok=True)


# ── Per-experiment curves ─────────────────────────────────────────────────────
def plot_curves(history: dict, exp_name: str) -> str:
    """Plot train/val loss and accuracy curves. Returns PNG path."""
    ep = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    axes[0].plot(ep, history["train_loss"], "b-o", ms=5, label="Train Loss")
    axes[0].plot(ep, history["val_loss"],   "r-o", ms=5, label="Val Loss")
    axes[0].set_title(f"{exp_name} — Loss")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].legend(); axes[0].grid(alpha=0.4)

    axes[1].plot(ep, [a*100 for a in history["train_acc"]], "b-o", ms=5, label="Train Acc")
    axes[1].plot(ep, [a*100 for a in history["val_acc"]],   "r-o", ms=5, label="Val Acc")
    axes[1].set_title(f"{exp_name} — Accuracy")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy (%)")
    axes[1].legend(); axes[1].grid(alpha=0.4)

    plt.tight_layout()
    path = f"{PLOTS_DIR}/{exp_name}_curves.png"
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return path


# ── Class-wise bar chart ───────────────────────────────────────────────────────
def plot_classwise(cw: np.ndarray, exp_name: str) -> str:
    """Bar chart of per-class test accuracy. Returns PNG path."""
    colors = ["#2ecc71" if v >= 0.5 else "#e74c3c" for v in cw]
    fig, ax = plt.subplots(figsize=(22, 5))
    ax.bar(range(NUM_CLASSES), cw * 100, color=colors, edgecolor="none")
    ax.axhline(cw.mean() * 100, color="navy", ls="--", lw=1.5,
               label=f"Mean {cw.mean()*100:.1f}%")
    ax.set_xlabel("Class ID"); ax.set_ylabel("Accuracy (%)")
    ax.set_title(f"{exp_name} — Class-wise Test Accuracy")
    ax.set_xlim(-1, NUM_CLASSES); ax.set_ylim(0, 108)
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = f"{PLOTS_DIR}/{exp_name}_classwise.png"
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return path


# ── All-experiments comparison bar ────────────────────────────────────────────
def plot_comparison(all_results: list) -> str:
    """Bar chart comparing best val accuracy across all experiments. Returns PNG path."""
    names  = [r["exp_name"]          for r in all_results]
    accs   = [r["best_val_acc"] * 100 for r in all_results]
    colors = ["#e74c3c" if not r["use_lora"] else "#2ecc71" for r in all_results]

    fig, ax = plt.subplots(figsize=(17, 5))
    bars = ax.bar(range(len(names)), accs, color=colors, edgecolor="white", width=0.72)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=42, ha="right", fontsize=9)
    ax.set_ylabel("Best Val Accuracy (%)")
    ax.set_ylim(0, max(accs) + 10)
    ax.set_title("All Experiments — Best Validation Accuracy")
    ax.grid(axis="y", alpha=0.35)
    for b, a in zip(bars, accs):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.4,
                f"{a:.1f}%", ha="center", va="bottom", fontsize=8)
    ax.legend(
        handles=[Patch(color="#e74c3c", label="No LoRA"),
                 Patch(color="#2ecc71", label="LoRA")],
        loc="lower right",
    )
    plt.tight_layout()
    path = f"{PLOTS_DIR}/all_experiments_comparison.png"
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return path


# ── Trainable-params vs accuracy scatter ──────────────────────────────────────
def plot_params_vs_acc(all_results: list) -> str:
    """Scatter: trainable params (M) vs best val accuracy. Returns PNG path."""
    fig, ax = plt.subplots(figsize=(9, 5))
    for r in all_results:
        c = "#e74c3c" if not r["use_lora"] else "#3498db"
        ax.scatter(r["trainable_params"] / 1e6, r["best_val_acc"] * 100,
                   color=c, s=90, zorder=3)
        ax.annotate(r["exp_name"], (r["trainable_params"] / 1e6, r["best_val_acc"] * 100),
                    fontsize=7, xytext=(4, 3), textcoords="offset points")
    ax.set_xlabel("Trainable Parameters (M)")
    ax.set_ylabel("Best Val Accuracy (%)")
    ax.set_title("Trainable Params vs Best Val Accuracy")
    ax.grid(alpha=0.35)
    ax.legend(
        handles=[Patch(color="#e74c3c", label="No LoRA"),
                 Patch(color="#3498db", label="LoRA")],
    )
    plt.tight_layout()
    path = f"{PLOTS_DIR}/params_vs_accuracy.png"
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return path


# ── Rank × Alpha heatmap ──────────────────────────────────────────────────────
def plot_heatmap(all_results: list) -> str:
    """Heatmap of val accuracy over rank × alpha grid. Returns PNG path."""
    lora_res = [r for r in all_results if r["use_lora"]]
    grid = np.zeros((len(RANKS), len(ALPHAS)))
    for r in lora_res:
        grid[RANKS.index(r["rank"]), ALPHAS.index(r["alpha"])] = r["best_val_acc"] * 100

    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(grid, cmap="YlGn", aspect="auto")
    ax.set_xticks(range(len(ALPHAS))); ax.set_xticklabels([f"α={a}" for a in ALPHAS])
    ax.set_yticks(range(len(RANKS)));  ax.set_yticklabels([f"r={r}" for r in RANKS])
    ax.set_title("LoRA Rank × Alpha — Best Val Accuracy (%)")
    plt.colorbar(im, ax=ax, label="Val Accuracy (%)")
    for i in range(len(RANKS)):
        for j in range(len(ALPHAS)):
            ax.text(j, i, f"{grid[i,j]:.1f}%",
                    ha="center", va="center", fontsize=10, color="black")
    plt.tight_layout()
    path = f"{PLOTS_DIR}/rank_alpha_heatmap.png"
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return path


# ── Sample prediction grid ────────────────────────────────────────────────────
def visualize_predictions(model, loader, exp_name: str, n: int = 8):
    """
    Show n sample images with true / predicted labels.
    Green title = correct, red = wrong.
    Returns (fig, png_path).
    """
    from q1.config import DEVICE
    model.eval()
    imgs, labels = next(iter(loader))
    with torch.no_grad():
        preds = model(pixel_values=imgs[:n].to(DEVICE)).logits.argmax(1).cpu()

    mean_t    = torch.tensor(CIFAR100_MEAN).view(3, 1, 1)
    std_t     = torch.tensor(CIFAR100_STD).view(3, 1, 1)
    imgs_show = (imgs[:n] * std_t + mean_t).clamp(0, 1)

    fig, axes = plt.subplots(1, n, figsize=(n * 2.2, 2.8))
    for i in range(n):
        ax = axes[i]
        ax.imshow(imgs_show[i].permute(1, 2, 0).numpy())
        tn = CIFAR100_CLASSES[labels[i].item()]
        pn = CIFAR100_CLASSES[preds[i].item()]
        ok = labels[i].item() == preds[i].item()
        ax.set_title(f"T:{tn}\nP:{pn}", fontsize=7,
                     color="green" if ok else "red", pad=2)
        ax.axis("off")
    fig.suptitle(
        f"{exp_name} — Predictions (green=correct  red=wrong)",
        fontsize=9, y=1.02,
    )
    plt.tight_layout()
    path = f"{PLOTS_DIR}/{exp_name}_sample_predictions.png"
    fig.savefig(path, dpi=110, bbox_inches="tight")
    return fig, path
