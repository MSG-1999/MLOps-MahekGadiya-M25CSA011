#!/usr/bin/env python
"""
q1_push.py  —  Retrain best LoRA config (from optuna_best_params.json),
               save weights, and push everything to HuggingFace Hub.

What gets pushed
----------------
  • LoRA adapter weights  (adapter_model.safetensors / adapter_model.bin)
  • adapter_config.json
  • best_model.pt          (full state-dict for reproducibility)
  • metadata.json
  • final_results.csv
  • q1_summary.json
  • Auto-generated model card (README.md) on HuggingFace

Environment variables
---------------------
  HF_TOKEN   — HuggingFace write token  (required)
               set via:  export HF_TOKEN=hf_...
               or pass with -e HF_TOKEN=... in docker run
  HF_REPO    — target repo  (default: MSG1999/vit-lora-cifar100)
  SKIP_TRAIN — set to "1" to skip retraining and push existing BEST_MODEL

Usage inside Docker:
    docker run --gpus all --rm \\
      -e WANDB_API_KEY=<key> \\
      -e HF_TOKEN=<your_huggingface_token> \\
      -v $(pwd)/data:/workspace/data \\
      dlops-assg5 \\
      python q1_push.py
"""

import json
import os
import sys
import warnings
warnings.filterwarnings("ignore")

from q1.config  import BEST_DIR, DEVICE, NUM_WORKERS, WANDB_PROJECT
from q1.dataset  import get_loaders
from q1.trainer  import run_experiment, save_best_model
from q1.utils    import print_banner

# ── config ────────────────────────────────────────────────────────────────────
HF_REPO    = os.environ.get("HF_REPO",    "MSG1999/vit-lora-cifar100")
SKIP_TRAIN = os.environ.get("SKIP_TRAIN", "0") == "1"

WANDB_URL = (
    "https://wandb.ai/msg1999-indian-institutes-of-technology-jodhpur/DLOps-A5-Q1"
)

MODEL_CARD_TEMPLATE = """\
---
license: apache-2.0
base_model: WinKawaks/vit-small-patch16-224
tags:
  - image-classification
  - lora
  - peft
  - cifar100
  - pytorch
datasets:
  - cifar100
metrics:
  - accuracy
---

# ViT-S + LoRA Fine-tuned on CIFAR-100

Fine-tuned with Low-Rank Adaptation (LoRA) via [PEFT](https://github.com/huggingface/peft)
as part of **DLOps Assignment 5, IIT Jodhpur**.

## Model details

| Property | Value |
|----------|-------|
| Base model | `WinKawaks/vit-small-patch16-224` |
| Dataset | CIFAR-100 (100 classes) |
| LoRA rank | {rank} |
| LoRA alpha | {alpha} |
| LoRA dropout | {dropout} |
| LoRA targets | query, key, value |
| Trainable params | {trainable:,} / {total:,} ({pct:.2f}%) |
| Best Val Accuracy | **{val_acc:.2f}%** |
| Best Test Accuracy | **{test_acc:.2f}%** |
| Baseline (no LoRA) | 80.77% |

## Training

- 10 epochs · batch 128 · AdamW (lr=3e-4, wd=1e-4) · CosineAnnealingLR
- Images pre-resized 32→224 and cached in RAM for speed
- WandB run: [{wandb_url}]({wandb_url})

## Results table

| Exp | Rank | Alpha | Val Acc | Test Acc | Trainable |
|-----|------|-------|---------|----------|-----------|
| exp01_no_lora | — | — | 80.77% | 80.77% | 38,500 |
| exp02_r2_a2 | 2 | 2 | 89.65% | 89.65% | 93,796 |
| exp03_r2_a4 | 2 | 4 | 90.03% | 90.03% | 93,796 |
| exp04_r2_a8 | 2 | 8 | 89.98% | 89.97% | 93,796 |
| exp05_r4_a2 | 4 | 2 | 89.91% | 89.91% | 149,092 |
| exp06_r4_a4 | 4 | 4 | 90.11% | 90.11% | 149,092 |
| exp07_r4_a8 | 4 | 8 | 90.28% | 90.28% | 149,092 |
| exp08_r8_a2 | 8 | 2 | 90.09% | 89.97% | 259,684 |
| exp09_r8_a4 | 8 | 4 | 90.17% | 90.17% | 259,684 |
| **exp10_r8_a8** | **8** | **8** | **90.46%** | **90.44%** | **259,684** |

## Inference

```python
from transformers import ViTForImageClassification, ViTImageProcessor
from peft import PeftModel
import torch
from PIL import Image

base = ViTForImageClassification.from_pretrained(
    "WinKawaks/vit-small-patch16-224",
    num_labels=100,
    ignore_mismatched_sizes=True,
)
model = PeftModel.from_pretrained(base, "MSG1999/vit-lora-cifar100")
model.eval()

processor = ViTImageProcessor.from_pretrained("WinKawaks/vit-small-patch16-224")
image     = Image.open("your_image.jpg").convert("RGB")
inputs    = processor(images=image, return_tensors="pt")

with torch.no_grad():
    pred = model(**inputs).logits.argmax(-1).item()
print(f"Predicted class index: {{pred}}")
```
"""


# ── helpers ───────────────────────────────────────────────────────────────────
def _load_best_params() -> dict:
    path = "optuna_best_params.json"
    if not os.path.isfile(path):
        print(
            f"  [warn] {path} not found — using default best params "
            "(rank=8, alpha=8, dropout=0.3).",
            flush=True,
        )
        return {"rank": 8, "alpha": 8, "dropout": 0.3}
    with open(path) as f:
        data = json.load(f)
    return data.get("params", data)


def _write_model_card(result: dict):
    card = MODEL_CARD_TEMPLATE.format(
        rank=result["rank"],
        alpha=result["alpha"],
        dropout=result["dropout"],
        trainable=result["trainable_params"],
        total=result["total_params"],
        pct=result["trainable_params"] / result["total_params"] * 100,
        val_acc=result["best_val_acc"] * 100,
        test_acc=result["overall_test_acc"] * 100,
        wandb_url=WANDB_URL,
    )
    card_path = os.path.join(BEST_DIR, "README.md")
    with open(card_path, "w") as f:
        f.write(card)
    print(f"  Model card written → {card_path}", flush=True)


def _push_to_hub(result: dict):
    from huggingface_hub import HfApi, create_repo

    token = os.environ.get("HF_TOKEN")
    if not token:
        print(
            "  [error] HF_TOKEN not set. Export it with:\n"
            "          export HF_TOKEN=hf_...",
            flush=True,
        )
        sys.exit(1)

    api = HfApi(token=token)
    print(f"  Creating / confirming repo: {HF_REPO} …", flush=True)
    create_repo(HF_REPO, repo_type="model", exist_ok=True,
                private=False, token=token)

    # upload BEST_MODEL folder (adapter + state-dict + metadata + card)
    print(f"  Uploading {BEST_DIR}/ → {HF_REPO} …", flush=True)
    api.upload_folder(
        folder_path=BEST_DIR,
        repo_id=HF_REPO,
        repo_type="model",
        commit_message=(
            f"Upload best LoRA (r={result['rank']}, α={result['alpha']}, "
            f"test_acc={result['overall_test_acc']*100:.2f}%)"
        ),
    )
    print("  ✅ Adapter + weights uploaded.", flush=True)

    # upload results files
    for fname in ("final_results.csv", "q1_summary.json", "optuna_best_params.json"):
        if os.path.isfile(fname):
            api.upload_file(
                path_or_fileobj=fname,
                path_in_repo=fname,
                repo_id=HF_REPO,
                repo_type="model",
                commit_message=f"Add {fname}",
            )
            print(f"  ✅ {fname} uploaded.", flush=True)

    print(f"\n  🎉 Model live at: https://huggingface.co/{HF_REPO}", flush=True)


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    print_banner(DEVICE, NUM_WORKERS)
    os.makedirs(BEST_DIR, exist_ok=True)

    if SKIP_TRAIN:
        print("  SKIP_TRAIN=1 — skipping retraining, pushing existing BEST_MODEL.",
              flush=True)
        # load metadata for the card
        meta_path = os.path.join(BEST_DIR, "metadata.json")
        if not os.path.isfile(meta_path):
            print(f"  [error] {meta_path} not found. Run q1_train.py first.", flush=True)
            sys.exit(1)
        with open(meta_path) as f:
            result = json.load(f)
    else:
        params = _load_best_params()
        rank    = params.get("rank",    8)
        alpha   = params.get("alpha",   8)
        dropout = params.get("dropout", 0.3)

        print(f"\n  Retraining best config:  rank={rank}  alpha={alpha}  dropout={dropout}")
        print("  Building data loaders …", flush=True)
        train_loader, val_loader = get_loaders()

        result = run_experiment(
            rank, alpha, dropout, use_lora=True,
            exp_name=f"best_r{rank}_a{alpha}_d{int(dropout*100)}",
            train_loader=train_loader,
            val_loader=val_loader,
        )
        save_best_model(result["model_ref"], result)

    _write_model_card(result)
    _push_to_hub(result)


if __name__ == "__main__":
    main()
