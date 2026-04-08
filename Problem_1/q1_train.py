#!/usr/bin/env python
"""
q1_train.py  —  Train ViT-S on CIFAR-100.

Modes
-----
Full LoRA grid  (default — 1 baseline + 9 LoRA experiments):
    python q1_train.py

Baseline only   (head-only, no LoRA):
    python q1_train.py --baseline

Usage inside Docker:
    docker run --gpus all --rm \\
      -e WANDB_API_KEY=<key> \\
      -v $(pwd)/data:/workspace/data \\
      dlops-assg5 \\
      python q1_train.py            # full grid

    docker run --gpus all --rm \\
      -e WANDB_API_KEY=<key> \\
      -v $(pwd)/data:/workspace/data \\
      dlops-assg5 \\
      python q1_train.py --baseline # baseline only
"""

import argparse
import datetime
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")

import pandas as pd
import wandb

from q1.config  import (
    ALPHAS, BEST_DIR, DEVICE, LOGS_DIR,
    LORA_DROPOUT, PLOTS_DIR, RANKS,
    SAVE_DIR, WANDB_PROJECT,
)
from q1.dataset  import get_loaders
from q1.model    import build_baseline, build_lora
from q1.plots    import plot_comparison, plot_heatmap, plot_params_vs_acc
from q1.trainer  import run_experiment, save_best_model
from q1.utils    import TeeLogger, print_banner
from q1.config   import NUM_WORKERS

# ── make output dirs ──────────────────────────────────────────────────────────
for d in [SAVE_DIR, PLOTS_DIR, LOGS_DIR, BEST_DIR]:
    os.makedirs(d, exist_ok=True)


# ── summary helpers ───────────────────────────────────────────────────────────
def print_summary(results: list):
    print(f"\n{'='*88}\nSUMMARY\n{'='*88}")
    print(
        f"{'Exp':<24} {'LoRA':>5} {'R':>3} {'A':>3} {'Drop':>5} "
        f"{'ValAcc':>8} {'TestAcc':>8} {'Trainable':>12}"
    )
    print("─" * 88)
    for r in results:
        print(
            f"{r['exp_name']:<24} {'Y' if r['use_lora'] else 'N':>5} "
            f"{str(r['rank']):>3} {str(r['alpha']):>3} {r['dropout']:>5} "
            f"{r['best_val_acc']*100:>7.2f}% {r['overall_test_acc']*100:>7.2f}% "
            f"{r['trainable_params']:>12,}"
        )
    clean = [
        {k: v for k, v in r.items() if k not in ("history", "model_ref")}
        for r in results
    ]
    with open("q1_summary.json", "w") as f:
        json.dump(clean, f, indent=2)
    print("\nSaved → q1_summary.json")

    df = pd.DataFrame(clean)
    df["best_val_acc"]     = (df["best_val_acc"]     * 100).round(2)
    df["overall_test_acc"] = (df["overall_test_acc"] * 100).round(2)
    df.rename(columns={
        "best_val_acc":     "best_val_acc_%",
        "overall_test_acc": "overall_test_acc_%",
    }, inplace=True)
    df.to_csv("final_results.csv", index=False)
    print("Saved → final_results.csv")


# ── grid runner ───────────────────────────────────────────────────────────────
def run_grid(train_loader, val_loader) -> list:
    results = []
    n = 0

    # exp01 — baseline
    n += 1
    results.append(run_experiment(
        0, 0, use_lora=False,
        exp_name=f"exp{n:02d}_no_lora",
        train_loader=train_loader, val_loader=val_loader,
    ))

    # exp02–10 — LoRA grid
    for rank in RANKS:
        for alpha in ALPHAS:
            n += 1
            results.append(run_experiment(
                rank, alpha, LORA_DROPOUT, True,
                exp_name=f"exp{n:02d}_r{rank}_a{alpha}",
                train_loader=train_loader, val_loader=val_loader,
            ))
    return results


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Q1 Training Script")
    parser.add_argument(
        "--baseline", action="store_true",
        help="Run only the head-only baseline (no LoRA).",
    )
    args = parser.parse_args()

    print_banner(DEVICE, NUM_WORKERS)

    run_ts   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    main_log = TeeLogger(os.path.join(LOGS_DIR, f"run_{run_ts}_master.log"))
    main_log.log(f"  Mode   : {'baseline only' if args.baseline else 'full grid'}")
    main_log.log(f"  Device : {DEVICE}")

    print("\n  Building data loaders (one-time RAM cache) …", flush=True)
    train_loader, val_loader = get_loaders()
    print("  Loaders ready.\n", flush=True)

    if args.baseline:
        # ── baseline only ────────────────────────────────────────────────────
        result = run_experiment(
            0, 0, use_lora=False,
            exp_name="exp01_no_lora",
            train_loader=train_loader, val_loader=val_loader,
        )
        all_results = [result]
    else:
        # ── full grid ────────────────────────────────────────────────────────
        all_results = run_grid(train_loader, val_loader)

        # summary plots → WandB
        p1 = plot_comparison(all_results)
        p2 = plot_params_vs_acc(all_results)
        p3 = plot_heatmap(all_results)
        sr = wandb.init(project=WANDB_PROJECT, name="summary_plots", reinit=True)
        wandb.log({
            "comparison":    wandb.Image(p1),
            "params_vs_acc": wandb.Image(p2),
            "heatmap":       wandb.Image(p3),
        })
        sr.finish()

    print_summary(all_results)

    # save best model
    best_result = max(all_results, key=lambda r: r["best_val_acc"])
    best_model  = best_result.get("model_ref")

    if best_model is None:
        print(f"  Reloading best model from {best_result['ckpt_path']} …", flush=True)
        if best_result["use_lora"]:
            best_model = build_lora(
                best_result["rank"], best_result["alpha"], best_result["dropout"]
            )
        else:
            best_model = build_baseline()
        import torch
        best_model.load_state_dict(
            torch.load(best_result["ckpt_path"], map_location="cpu"), strict=False
        )

    save_best_model(best_model, best_result)

    main_log.log(f"\n{'='*60}")
    main_log.log(
        f"BEST : {best_result['exp_name']}  |  "
        f"Val Acc : {best_result['best_val_acc']*100:.2f}%"
    )
    main_log.log(f"Weights → {BEST_DIR}/")
    main_log.log("Next    : python q1_optuna.py  |  python q1_push.py")
    main_log.close()


if __name__ == "__main__":
    main()
