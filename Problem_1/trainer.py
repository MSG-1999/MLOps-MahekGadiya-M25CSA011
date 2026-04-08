"""
q1/trainer.py  —  Training loop, evaluation, and experiment runner.

Public API
----------
train_one_epoch(model, loader, criterion, optimizer, scaler, epoch, epochs)
eval_one_epoch(model, loader, criterion, epoch, epochs)
classwise_acc(model, loader)
run_experiment(rank, alpha, dropout, use_lora, exp_name, train_loader, val_loader)
save_best_model(model, result)
"""

import os
import sys
import json
import time

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import wandb

from q1.config import (
    BATCH_SIZE, BEST_DIR, DEVICE, EPOCHS,
    LR, LOGS_DIR, NUM_CLASSES,
    LORA_DROPOUT, SAVE_DIR, WANDB_PROJECT,
    WEIGHT_DECAY,
)
from q1.dataset  import get_loaders
from q1.model    import build_baseline, build_lora
from q1.plots    import plot_curves, plot_classwise, visualize_predictions
from q1.utils    import (
    TeeLogger, count_params, lora_grad_norms,
    log, set_logger, get_logger,
)


# ── Single epoch ──────────────────────────────────────────────────────────────
def train_one_epoch(model, loader, criterion, optimizer, scaler, epoch, epochs):
    model.train()
    total_loss = correct = total = 0
    pbar = tqdm(
        loader,
        desc=f"  Ep {epoch:>2}/{epochs} TRAIN",
        leave=True, dynamic_ncols=True, file=sys.stdout,
        bar_format=(
            "  {l_bar}{bar}| {n_fmt}/{total_fmt} "
            "[{elapsed}<{remaining}, {rate_fmt}]  {postfix}"
        ),
    )
    for imgs, labels in pbar:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        with torch.amp.autocast("cuda", enabled=(DEVICE == "cuda")):
            logits = model(pixel_values=imgs).logits
            loss   = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        bs          = labels.size(0)
        total_loss += loss.item() * bs
        correct    += (logits.argmax(1) == labels).sum().item()
        total      += bs
        pbar.set_postfix(
            loss=f"{total_loss/total:.4f}",
            acc=f"{correct/total*100:.2f}%",
            refresh=False,
        )
    pbar.close()
    return total_loss / total, correct / total


@torch.no_grad()
def eval_one_epoch(model, loader, criterion, epoch, epochs):
    model.eval()
    total_loss = correct = total = 0
    pbar = tqdm(
        loader,
        desc=f"  Ep {epoch:>2}/{epochs}  VAL ",
        leave=True, dynamic_ncols=True, file=sys.stdout,
        bar_format=(
            "  {l_bar}{bar}| {n_fmt}/{total_fmt} "
            "[{elapsed}<{remaining}]  {postfix}"
        ),
    )
    for imgs, labels in pbar:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        logits = model(pixel_values=imgs).logits
        loss   = criterion(logits, labels)
        bs          = labels.size(0)
        total_loss += loss.item() * bs
        correct    += (logits.argmax(1) == labels).sum().item()
        total      += bs
        pbar.set_postfix(
            loss=f"{total_loss/total:.4f}",
            acc=f"{correct/total*100:.2f}%",
            refresh=False,
        )
    pbar.close()
    return total_loss / total, correct / total


@torch.no_grad()
def classwise_acc(model, loader, num_classes: int = NUM_CLASSES):
    """Return per-class accuracy as a numpy array of shape (num_classes,)."""
    model.eval()
    correct = torch.zeros(num_classes)
    counts  = torch.zeros(num_classes)
    for imgs, labels in loader:
        preds = model(pixel_values=imgs.to(DEVICE)).logits.argmax(1).cpu()
        for c in range(num_classes):
            mask        = labels == c
            correct[c] += (preds[mask] == c).sum()
            counts[c]  += mask.sum()
    return (correct / counts.clamp(min=1)).numpy()


# ── Full experiment ────────────────────────────────────────────────────────────
def run_experiment(
    rank,
    alpha,
    dropout=LORA_DROPOUT,
    use_lora=True,
    exp_name="exp",
    train_loader=None,
    val_loader=None,
):
    """
    Train one experiment, log to WandB, save best checkpoint.

    Returns a dict with keys:
        exp_name, use_lora, rank, alpha, dropout,
        best_val_acc, overall_test_acc, trainable_params, total_params,
        history, ckpt_path, log_path, model_ref
    """
    # ── logging ───────────────────────────────────────────────────────────────
    os.makedirs(LOGS_DIR, exist_ok=True)
    log_path = os.path.join(LOGS_DIR, f"{exp_name}.log")
    set_logger(TeeLogger(log_path))

    log(f"\n{'─'*60}")
    log(f"  {exp_name}  |  LoRA={use_lora}  rank={rank}  alpha={alpha}  drop={dropout}")
    log(f"{'─'*60}")

    # ── WandB run ─────────────────────────────────────────────────────────────
    run = wandb.init(
        project=WANDB_PROJECT, name=exp_name, reinit=True,
        config=dict(rank=rank, alpha=alpha, dropout=dropout,
                    use_lora=use_lora, epochs=EPOCHS, lr=LR),
    )

    # ── data ──────────────────────────────────────────────────────────────────
    if train_loader is None or val_loader is None:
        train_loader, val_loader = get_loaders()

    # ── model ─────────────────────────────────────────────────────────────────
    model = build_lora(rank, alpha, dropout) if use_lora else build_baseline()
    model = model.to(DEVICE)

    tr_p, tt_p = count_params(model)
    log(f"  Trainable : {tr_p:,}  |  Total : {tt_p:,}")
    wandb.config.update({"trainable_params": tr_p, "total_params": tt_p})

    # ── optimiser ─────────────────────────────────────────────────────────────
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR, weight_decay=WEIGHT_DECAY,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    scaler    = torch.amp.GradScaler("cuda", enabled=(DEVICE == "cuda"))

    # ── training loop ─────────────────────────────────────────────────────────
    history      = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc = 0.0
    ckpt_path    = f"{SAVE_DIR}/{exp_name}_best.pt"
    step         = 0

    HDR = (f"\n  {'Ep':>3} │ {'TrLoss':>8} │ {'TrAcc':>7} │"
           f" {'VaLoss':>8} │ {'VaAcc':>7} │ {'LR':>9} │ {'Time':>6} │")
    SEP = "  " + "─" * (len(HDR) - 3)
    tqdm.write(HDR, file=sys.stdout)
    tqdm.write(SEP, file=sys.stdout)
    sys.stdout.flush()

    os.makedirs(SAVE_DIR, exist_ok=True)

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, epoch, EPOCHS
        )
        va_loss, va_acc = eval_one_epoch(
            model, val_loader, criterion, epoch, EPOCHS
        )
        scheduler.step()
        step   += len(train_loader)
        elapsed = time.time() - t0
        cur_lr  = scheduler.get_last_lr()[0]

        for k, v in zip(
            ["train_loss", "val_loss", "train_acc", "val_acc"],
            [tr_loss, va_loss, tr_acc, va_acc],
        ):
            history[k].append(v)

        is_best = va_acc > best_val_acc
        marker  = " ★ BEST" if is_best else ""
        log(
            f"  {epoch:>3} │ {tr_loss:>8.4f} │ {tr_acc*100:>6.2f}% │"
            f" {va_loss:>8.4f} │ {va_acc*100:>6.2f}% │"
            f" {cur_lr:>9.2e} │ {elapsed:>5.1f}s │{marker}"
        )

        log_d = {
            "epoch": epoch,
            "train/loss": tr_loss, "train/accuracy": tr_acc,
            "val/loss":   va_loss, "val/accuracy":   va_acc,
            "lr": cur_lr,
        }
        if use_lora:
            log_d.update(lora_grad_norms(model))
        wandb.log(log_d, step=step)

        if is_best:
            best_val_acc = va_acc
            torch.save(model.state_dict(), ckpt_path)
            if use_lora:
                model.save_pretrained(f"{SAVE_DIR}/{exp_name}_adapter")

    tqdm.write(SEP, file=sys.stdout)
    tqdm.write(
        f"  Done. Best val acc = {best_val_acc*100:.2f}%  │  ckpt → {ckpt_path}",
        file=sys.stdout,
    )
    sys.stdout.flush()

    # ── post-training metrics & plots ─────────────────────────────────────────
    cw      = classwise_acc(model, val_loader)
    overall = float(cw.mean())

    fig_pred, _ = visualize_predictions(model, val_loader, exp_name)
    cp          = plot_curves(history, exp_name)
    cwp         = plot_classwise(cw, exp_name)

    table = wandb.Table(
        data=[[i, float(cw[i])] for i in range(NUM_CLASSES)],
        columns=["class_id", "accuracy"],
    )
    wandb.log({
        "curves":             wandb.Image(cp),
        "classwise_hist":     wandb.Image(cwp),
        "classwise_bar":      wandb.plot.bar(
            table, "class_id", "accuracy", title=f"{exp_name} — Classwise"
        ),
        "sample_predictions": wandb.Image(fig_pred),
        "best_val_acc":       best_val_acc,
        "overall_test_acc":   overall,
    })

    import matplotlib.pyplot as plt
    plt.close(fig_pred)
    run.finish()
    log(f"  Log → {log_path}")
    get_logger().close()
    set_logger(None)

    return dict(
        exp_name=exp_name, use_lora=use_lora, rank=rank, alpha=alpha,
        dropout=dropout, best_val_acc=best_val_acc, overall_test_acc=overall,
        trainable_params=tr_p, total_params=tt_p,
        history=history, ckpt_path=ckpt_path, log_path=log_path,
        model_ref=model,
    )


# ── Save best model ────────────────────────────────────────────────────────────
def save_best_model(model, result: dict) -> str:
    """
    Save full state-dict + LoRA adapter (if applicable) + metadata JSON
    to BEST_DIR.  Returns the checkpoint path.
    """
    import shutil
    if os.path.exists(BEST_DIR):
        shutil.rmtree(BEST_DIR)
    os.makedirs(BEST_DIR, exist_ok=True)

    ckpt = os.path.join(BEST_DIR, "best_model.pt")
    torch.save(model.state_dict(), ckpt)
    print(f"  [BestModel] Weights  → {ckpt}", flush=True)

    if result["use_lora"]:
        adapter_dir = os.path.join(BEST_DIR, "lora_adapter")
        model.save_pretrained(adapter_dir)
        print(f"  [BestModel] Adapter  → {adapter_dir}/", flush=True)

    meta = {k: v for k, v in result.items() if k not in ("history", "model_ref")}
    meta["saved_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    meta["ckpt_path"] = ckpt
    with open(os.path.join(BEST_DIR, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n  ★ BEST MODEL SAVED → {BEST_DIR}/", flush=True)
    print(f"    exp      : {result['exp_name']}", flush=True)
    print(f"    val acc  : {result['best_val_acc']*100:.2f}%", flush=True)
    if result["use_lora"]:
        print(
            f"    rank={result['rank']}  alpha={result['alpha']}  "
            f"dropout={result['dropout']}",
            flush=True,
        )
    return ckpt
