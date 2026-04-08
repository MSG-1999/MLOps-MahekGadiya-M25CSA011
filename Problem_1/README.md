# Assignment 5 — ViT-S LoRA Fine-tuning on CIFAR-100

> **DLOps Assignment 5 | IIT Jodhpur**
> Fine-tuning Vision Transformer (ViT-S/16) on CIFAR-100 using Low-Rank Adaptation (LoRA).

---

## 🔗 Quick Links

| | |
|---|---|
| 🤗 **HuggingFace Model** | [MSG1999/vit-lora-cifar100](https://huggingface.co/MSG1999/vit-lora-cifar100/tree/main) |
| 📊 **WandB Dashboard** | [DLOps-A5-Q1](https://wandb.ai/msg1999-indian-institutes-of-technology-jodhpur/DLOps-A5-Q1) |

---

## 📁 Repository Structure

```
Assignment-5/
├── q1/                         # Python package (importable modules)
│   ├── __init__.py
│   ├── config.py               # All hyperparameters & paths
│   ├── dataset.py              # CIFAR-100 data loading + RAM cache
│   ├── model.py                # Baseline & LoRA model builders
│   ├── trainer.py              # Train/eval loops + experiment runner
│   ├── plots.py                # All matplotlib plotting helpers
│   └── utils.py                # TeeLogger, param counting, grad norms
│
├── q1_train.py                 # Entry point: baseline or full LoRA grid
├── q1_optuna.py                # Entry point: Optuna hyperparameter search
├── q1_push.py                  # Entry point: retrain best + push to HF Hub
│
├── Dockerfile                  # Reproducible Docker container
├── requirements_q1.txt         # Python dependencies
│
├── final_results.csv           # Grid search results table
├── q1_summary.json             # Per-experiment JSON summary
└── optuna_best_params.json     # Optuna best params
```

---

## 🐳 Docker Setup

> **All experiments must be run inside a Docker container.**

### 1 — Build the image

```bash
docker build -t dlops-assg5 .
```

### 2 — Run experiments

#### Q1 – Baseline (head-only, no LoRA)

```bash
docker run --gpus all --rm \
  -e WANDB_API_KEY=<your_wandb_key> \
  -v $(pwd)/data:/workspace/data \
  dlops-assg5 \
  python q1_train.py --baseline
```

#### Q1 – Full LoRA grid (1 baseline + 9 LoRA experiments)

```bash
docker run --gpus all --rm \
  -e WANDB_API_KEY=<your_wandb_key> \
  -v $(pwd)/data:/workspace/data \
  dlops-assg5 \
  python q1_train.py
```

#### Q1 – Optuna hyperparameter search (20 trials)

```bash
docker run --gpus all --rm \
  -e WANDB_API_KEY=<your_wandb_key> \
  -v $(pwd)/data:/workspace/data \
  dlops-assg5 \
  python q1_optuna.py
```

To change the number of trials:

```bash
python q1_optuna.py --trials 30
```

#### Q1 – Retrain best config and push to HuggingFace Hub

```bash
docker run --gpus all --rm \
  -e WANDB_API_KEY=<your_wandb_key> \
  -e HF_TOKEN=<your_huggingface_token> \
  -v $(pwd)/data:/workspace/data \
  dlops-assg5 \
  python q1_push.py
```

To push an **already-trained** model without retraining:

```bash
docker run --gpus all --rm \
  -e HF_TOKEN=<your_huggingface_token> \
  -e SKIP_TRAIN=1 \
  -v $(pwd)/data:/workspace/data \
  -v $(pwd)/weights:/workspace/weights \
  dlops-assg5 \
  python q1_push.py
```

---

## ⚙️ Local Installation (without Docker)

```bash
pip install -r requirements_q1.txt
```

Then run any entry point directly:

```bash
wandb login
python q1_train.py            # full grid
python q1_train.py --baseline # baseline only
python q1_optuna.py           # Optuna search
python q1_push.py             # push to HF Hub
```

**Dependencies:**

| Package | Version |
|---------|---------|
| torch | ≥ 2.0.0 |
| torchvision | ≥ 0.15.0 |
| transformers | ≥ 4.38.0 |
| peft | ≥ 0.9.0 |
| wandb | ≥ 0.16.0 |
| optuna | ≥ 3.5.0 |
| huggingface_hub | ≥ 0.20.0 |
| pandas | ≥ 1.5.0 |
| matplotlib | ≥ 3.7.0 |

---

## 📊 Results — Q1 LoRA Grid Search

### Experiment Table

| Exp | LoRA | Rank | Alpha | Dropout | Val Acc (%) | Test Acc (%) | Trainable Params |
|-----|------|------|-------|---------|------------|--------------|-----------------|
| exp01_no_lora | ❌ | — | — | 0.1 | 80.77 | 80.77 | 38,500 |
| exp02_r2_a2 | ✅ | 2 | 2 | 0.1 | 89.65 | 89.65 | 93,796 |
| exp03_r2_a4 | ✅ | 2 | 4 | 0.1 | 90.03 | 90.03 | 93,796 |
| exp04_r2_a8 | ✅ | 2 | 8 | 0.1 | 89.98 | 89.97 | 93,796 |
| exp05_r4_a2 | ✅ | 4 | 2 | 0.1 | 89.91 | 89.91 | 149,092 |
| exp06_r4_a4 | ✅ | 4 | 4 | 0.1 | 90.11 | 90.11 | 149,092 |
| exp07_r4_a8 | ✅ | 4 | 8 | 0.1 | 90.28 | 90.28 | 149,092 |
| exp08_r8_a2 | ✅ | 8 | 2 | 0.1 | 90.09 | 89.97 | 259,684 |
| exp09_r8_a4 | ✅ | 8 | 4 | 0.1 | 90.17 | 90.17 | 259,684 |
| **exp10_r8_a8** | ✅ | **8** | **8** | **0.1** | **90.46** | **90.44** | **259,684** |

### Optuna Best Parameters

| Parameter | Value |
|-----------|-------|
| Rank | 8 |
| Alpha | 8 |
| Dropout | 0.3 |
| Best Val Acc | **90.39%** |

### Key Observations

- LoRA adds only **1.18% extra parameters** (259,684 / 21,925,348) yet gains **+9.7% test accuracy** over the frozen-head baseline.
- Higher alpha consistently improves accuracy at every rank level.
- Best model: `rank=8, alpha=8` → **90.44% test accuracy**.

---

## 🧠 Model Architecture

```
Base        : WinKawaks/vit-small-patch16-224
LoRA target : query, key, value (all attention layers)
Head        : Linear(384 → 100) — always trainable
```

| Mode | Trainable | Total | % |
|------|-----------|-------|---|
| Baseline (no LoRA) | 38,500 | 21,704,164 | 0.18% |
| LoRA r=2 | 93,796 | 21,759,460 | 0.43% |
| LoRA r=4 | 149,092 | 21,814,756 | 0.68% |
| **LoRA r=8** | **259,684** | **21,925,348** | **1.18%** |

---

## 📈 WandB Plots

All training curves, class-wise histograms, and comparison plots are logged live:

👉 [https://wandb.ai/msg1999-indian-institutes-of-technology-jodhpur/DLOps-A5-Q1](https://wandb.ai/msg1999-indian-institutes-of-technology-jodhpur/DLOps-A5-Q1)

---

## 🤗 HuggingFace Model

Best model (rank=8, alpha=8, test acc=90.44%):

👉 [https://huggingface.co/MSG1999/vit-lora-cifar100](https://huggingface.co/MSG1999/vit-lora-cifar100/tree/main)

### Quick inference

```python
from transformers import ViTForImageClassification, ViTImageProcessor
from peft import PeftModel
import torch
from PIL import Image

base  = ViTForImageClassification.from_pretrained(
    "WinKawaks/vit-small-patch16-224", num_labels=100, ignore_mismatched_sizes=True
)
model = PeftModel.from_pretrained(base, "MSG1999/vit-lora-cifar100")
model.eval()

processor = ViTImageProcessor.from_pretrained("WinKawaks/vit-small-patch16-224")
image     = Image.open("your_image.jpg").convert("RGB")
inputs    = processor(images=image, return_tensors="pt")

with torch.no_grad():
    pred = model(**inputs).logits.argmax(-1).item()
print(f"Predicted class: {pred}")
```

---

## 📝 Notes

- CIFAR-100 downloads automatically on first run via `torchvision.datasets`.
- Images are pre-resized 32→224 **once** and cached in RAM — speeds up all 10 experiments significantly.
- Set `WANDB_MODE=disabled` to skip WandB logging during testing.
- The data volume mount (`-v $(pwd)/data:/workspace/data`) persists the download across container runs.
