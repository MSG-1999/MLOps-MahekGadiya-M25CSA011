#!/usr/bin/env python
"""
q1_optuna.py  —  Optuna hyperparameter search for ViT-S LoRA on CIFAR-100.

Searches over:  rank ∈ {2,4,8},  alpha ∈ {2,4,8},  dropout ∈ [0.05, 0.30]

Outputs
-------
  optuna_best_params.json      — best trial params + val accuracy
  plots/optuna_history.png     — optimisation history curve
  plots/optuna_importance.png  — parameter importance bar chart

Usage inside Docker:
    docker run --gpus all --rm \\
      -e WANDB_API_KEY=<key> \\
      -v $(pwd)/data:/workspace/data \\
      dlops-assg5 \\
      python q1_optuna.py

Options:
  --trials N    number of Optuna trials  (default: 20)
"""

import argparse
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from q1.config  import ALPHAS, PLOTS_DIR, RANKS
from q1.dataset  import get_loaders
from q1.trainer  import run_experiment
from q1.utils    import print_banner
from q1.config   import DEVICE, NUM_WORKERS

os.makedirs(PLOTS_DIR, exist_ok=True)


def objective(trial, train_loader, val_loader):
    rank    = trial.suggest_categorical("rank",    [2, 4, 8])
    alpha   = trial.suggest_categorical("alpha",   [2, 4, 8])
    dropout = trial.suggest_float("dropout", 0.05, 0.30, step=0.05)

    result = run_experiment(
        rank, alpha, dropout, use_lora=True,
        exp_name=(
            f"optuna_t{trial.number:02d}_r{rank}_a{alpha}_d{int(dropout*100)}"
        ),
        train_loader=train_loader,
        val_loader=val_loader,
    )
    return result["best_val_acc"]


def main():
    parser = argparse.ArgumentParser(description="Q1 Optuna search")
    parser.add_argument("--trials", type=int, default=20,
                        help="Number of Optuna trials (default: 20)")
    args = parser.parse_args()

    print_banner(DEVICE, NUM_WORKERS)
    print(f"\n{'='*60}")
    print(f"  OPTUNA SEARCH  —  {args.trials} trials")
    print(f"{'='*60}\n")

    print("  Building data loaders (one-time RAM cache) …", flush=True)
    train_loader, val_loader = get_loaders()
    print("  Loaders ready.\n", flush=True)

    study = optuna.create_study(direction="maximize")
    study.optimize(
        lambda trial: objective(trial, train_loader, val_loader),
        n_trials=args.trials,
        show_progress_bar=True,
    )

    best = study.best_trial
    print(f"\n  ★ Best val acc  : {best.value*100:.2f}%")
    print(f"  ★ Best params   : {best.params}")

    out = {"best_val_acc": best.value, "params": best.params}
    with open("optuna_best_params.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\n  Saved → optuna_best_params.json")

    # visualisation (optional — needs optuna[visualization])
    try:
        from optuna.visualization.matplotlib import (
            plot_optimization_history,
            plot_param_importances,
        )
        fig1 = plot_optimization_history(study)
        fig1.savefig(f"{PLOTS_DIR}/optuna_history.png",    dpi=100)
        plt.close()
        print(f"  Saved → {PLOTS_DIR}/optuna_history.png")

        fig2 = plot_param_importances(study)
        fig2.savefig(f"{PLOTS_DIR}/optuna_importance.png", dpi=100)
        plt.close()
        print(f"  Saved → {PLOTS_DIR}/optuna_importance.png")
    except Exception as e:
        print(f"  [warn] Optuna plots skipped: {e}")


if __name__ == "__main__":
    main()
