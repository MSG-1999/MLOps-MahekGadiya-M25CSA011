
<div align="center">

# 🧠 DLOps Assignment 5 — LoRA + Adversarial Robustness

### IIT Jodhpur · ViT-S LoRA on CIFAR-100 · ResNet-18 Adversarial Attacks (IBM ART)

[![Q1 HuggingFace](https://img.shields.io/badge/🤗_Q1_Model-MSG1999%2Fvit--lora--cifar100-yellow)](https://huggingface.co/MSG1999/vit-lora-cifar100)
[![Q2 HuggingFace](https://img.shields.io/badge/🤗_Q2_Model-MSG1999%2FDLOps--A5--Q2--ART-orange)](https://huggingface.co/MSG1999/DLOps-A5-Q2-ART)
[![Q1 WandB](https://img.shields.io/badge/📊_Q1_WandB-Experiments-yellow?logo=weightsandbiases)](https://wandb.ai/msg1999-indian-institutes-of-technology-jodhpur/DLOps-A5-Q1)
[![Q2 WandB](https://img.shields.io/badge/📊_Q2_WandB-Experiments-yellow?logo=weightsandbiases)](https://wandb.ai/msg1999-indian-institutes-of-technology-jodhpur/DLOps-A5-Q2-ART)
[![Framework](https://img.shields.io/badge/Framework-PyTorch-orange)](https://pytorch.org/)
[![Docker](https://img.shields.io/badge/Docker-Containerized-blue)](https://www.docker.com/)

</div>

---

## 🔗 Quick Links

| Resource | Link |
|----------|------|
| 🤗 Q1 — ViT-S LoRA Model | https://huggingface.co/MSG1999/vit-lora-cifar100 |
| 🤗 Q2 — Adversarial Robustness | https://huggingface.co/MSG1999/DLOps-A5-Q2-ART |
| 📊 Q1 — WandB Dashboard | https://wandb.ai/msg1999-indian-institutes-of-technology-jodhpur/DLOps-A5-Q1 |
| 📊 Q2 — WandB Dashboard | https://wandb.ai/msg1999-indian-institutes-of-technology-jodhpur/DLOps-A5-Q2-ART |

---

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PART 1 — Q1: ViT-Small + LoRA on CIFAR-100
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

## 📌 Overview

Fine-tuning **ViT-Small/16** (pre-trained on ImageNet) on **CIFAR-100** using **LoRA** — injecting trainable low-rank matrices into Query, Key, Value attention projections while keeping the base model frozen.

> **Only 1.18% of parameters trained** → **90.46% val accuracy** (+9.69 pp over baseline)

| | |
|--|--|
| 🏛️ Base Model | `WinKawaks/vit-small-patch16-224` |
| 📦 Dataset | CIFAR-100 (50k train / 10k test, 100 classes) |
| 🎯 Best Val Accuracy | **90.46%** |
| 🧪 Best Test Accuracy | **90.44%** |
| ⚙️ Best Config | rank=8, alpha=8, dropout=0.1 |
| 🔢 Trainable Params | 259,684 / 21,925,348 **(1.18%)** |
| 📁 Best Weights | `best_model.pt` |

---

## 🏗️ Architecture

```
ViT-Small/16 (WinKawaks/vit-small-patch16-224)
├── Patch Embedding          [frozen]
├── Transformer Encoder ×12
│   ├── Multi-Head Attention
│   │   ├── Query ── LoRA(A·B)  ✅ trained  (rank=8)
│   │   ├── Key   ── LoRA(A·B)  ✅ trained  (rank=8)
│   │   └── Value ── LoRA(A·B)  ✅ trained  (rank=8)
│   └── MLP                  [frozen]
└── Classification Head      ✅ trained  (100 classes)

LoRA update: W' = W + (α/r)·B·A   |   scaling = α/r = 8/8 = 1.0
```

---

## ⚙️ Hyperparameters

### Training

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| Learning Rate | `3e-4` |
| Weight Decay | `1e-4` |
| LR Scheduler | CosineAnnealingLR |
| Batch Size | 128 |
| Epochs | 10 |
| Image Size | 224 × 224 |

### Data Augmentation

| Transform | Value |
|-----------|-------|
| Random Horizontal Flip | p=0.5 |
| Random Crop | 224×224, padding=28 |
| Color Jitter | brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05 |
| Normalize Mean | (0.5071, 0.4867, 0.4408) |
| Normalize Std | (0.2675, 0.2565, 0.2761) |

### LoRA (Best Config)

| Parameter | Value |
|-----------|-------|
| Rank (r) | **8** |
| Alpha (α) | **8** |
| Scaling (α/r) | **1.0** |
| Dropout | **0.1** |
| Target Modules | `query`, `key`, `value` |
| Bias | none |
| Init | Gaussian A, Zero B |
| Trainable Params | **259,684 (1.18%)** |
| Total Params | 21,925,348 |

---

## 📊 Summary — All 10 Experiments

| Experiment | LoRA | Rank | Alpha | Dropout | Val Acc | Test Acc | Trainable Params |
|------------|:----:|:----:|:-----:|:-------:|:-------:|:--------:|:----------------:|
| exp01_no_lora | ❌ | — | — | 0.1 | 80.77% | 80.77% | 38,500 |
| exp02_r2_a2 | ✅ | 2 | 2 | 0.1 | 89.65% | 89.65% | 93,796 |
| exp03_r2_a4 | ✅ | 2 | 4 | 0.1 | 90.03% | 90.03% | 93,796 |
| exp04_r2_a8 | ✅ | 2 | 8 | 0.1 | 89.98% | 89.97% | 93,796 |
| exp05_r4_a2 | ✅ | 4 | 2 | 0.1 | 89.91% | 89.91% | 149,092 |
| exp06_r4_a4 | ✅ | 4 | 4 | 0.1 | 90.11% | 90.11% | 149,092 |
| exp07_r4_a8 | ✅ | 4 | 8 | 0.1 | 90.28% | 90.28% | 149,092 |
| exp08_r8_a2 | ✅ | 8 | 2 | 0.1 | 90.09% | 89.97% | 259,684 |
| exp09_r8_a4 | ✅ | 8 | 4 | 0.1 | 90.17% | 90.17% | 259,684 |
| **exp10_r8_a8 ⭐** | ✅ | **8** | **8** | **0.1** | **90.46%** | **90.44%** | **259,684** |

> LoRA (best) gives **+9.69 pp** over baseline with only **1.18%** trainable parameters.

---

## 📈 Detailed Train–Val Tables — All 10 Experiments

### 🔴 exp01 — Baseline (No LoRA) · Val Acc: 80.77%

| Epoch | Train Loss | Val Loss | Train Acc | Val Acc |
|:-----:|:----------:|:--------:|:---------:|:-------:|
| 1  | 2.2010 | 1.6216 | 62.09% | 75.85% |
| 2  | 1.5641 | 1.5309 | 77.60% | 78.25% |
| 3  | 1.4996 | 1.4973 | 79.41% | 79.07% |
| 4  | 1.4658 | 1.4794 | 80.59% | 79.93% |
| 5  | 1.4388 | 1.4670 | 81.37% | 80.03% |
| 6  | 1.4268 | 1.4575 | 81.88% | 80.43% |
| 7  | 1.4150 | 1.4513 | 82.03% | 80.61% |
| 8  | 1.4044 | 1.4501 | 82.71% | 80.56% |
| 9  | 1.4003 | 1.4486 | 82.68% | 80.70% |
| **10** | **1.3973** | **1.4480** | **82.99%** | **80.77%** |

---

### 🟢 exp02 — LoRA r=2, α=2 · Val Acc: 89.65%

| Epoch | Train Loss | Val Loss | Train Acc | Val Acc |
|:-----:|:----------:|:--------:|:---------:|:-------:|
| 1  | 1.8669 | 1.2886 | 71.03% | 85.24% |
| 2  | 1.2185 | 1.2087 | 87.02% | 87.57% |
| 3  | 1.1615 | 1.1798 | 88.46% | 88.16% |
| 4  | 1.1274 | 1.1535 | 89.57% | 88.65% |
| 5  | 1.1047 | 1.1405 | 90.21% | 89.24% |
| 6  | 1.0885 | 1.1339 | 90.90% | 89.31% |
| 7  | 1.0761 | 1.1289 | 91.18% | 89.52% |
| 8  | 1.0687 | 1.1254 | 91.47% | 89.60% |
| 9  | 1.0632 | 1.1254 | 91.62% | 89.53% |
| **10** | **1.0598** | **1.1240** | **91.71%** | **89.65%** |

---

### 🟢 exp03 — LoRA r=2, α=4 · Val Acc: 90.03%

| Epoch | Train Loss | Val Loss | Train Acc | Val Acc |
|:-----:|:----------:|:--------:|:---------:|:-------:|
| 1  | 1.8246 | 1.2712 | 71.97% | 86.00% |
| 2  | 1.2076 | 1.1927 | 87.21% | 87.84% |
| 3  | 1.1482 | 1.1644 | 88.84% | 88.56% |
| 4  | 1.1155 | 1.1448 | 89.73% | 89.02% |
| 5  | 1.0949 | 1.1359 | 90.56% | 89.45% |
| 6  | 1.0768 | 1.1303 | 91.18% | 89.64% |
| 7  | 1.0654 | 1.1246 | 91.52% | 89.78% |
| 8  | 1.0562 | 1.1188 | 91.77% | 89.92% |
| 9  | 1.0523 | 1.1175 | 91.91% | 89.95% |
| **10** | **1.0465** | **1.1172** | **92.08%** | **90.03%** |

---

### 🟢 exp04 — LoRA r=2, α=8 · Val Acc: 89.98%

| Epoch | Train Loss | Val Loss | Train Acc | Val Acc |
|:-----:|:----------:|:--------:|:---------:|:-------:|
| 1  | 1.8020 | 1.2528 | 72.33% | 86.35% |
| 2  | 1.2008 | 1.1882 | 87.39% | 87.75% |
| 3  | 1.1441 | 1.1544 | 88.97% | 88.64% |
| 4  | 1.1103 | 1.1391 | 89.92% | 89.29% |
| 5  | 1.0834 | 1.1277 | 90.81% | 89.39% |
| 6  | 1.0689 | 1.1244 | 91.34% | 89.57% |
| 7  | 1.0551 | 1.1140 | 91.76% | 89.69% |
| 8  | 1.0450 | 1.1127 | 92.23% | 89.84% |
| **9**  | **1.0401** | **1.1108** | **92.21%** | **89.98%** |
| 10 | 1.0343 | 1.1104 | 92.52% | 89.97% |

---

### 🟢 exp05 — LoRA r=4, α=2 · Val Acc: 89.91%

| Epoch | Train Loss | Val Loss | Train Acc | Val Acc |
|:-----:|:----------:|:--------:|:---------:|:-------:|
| 1  | 1.8787 | 1.2800 | 70.81% | 85.72% |
| 2  | 1.2141 | 1.1982 | 87.16% | 87.67% |
| 3  | 1.1584 | 1.1616 | 88.68% | 88.82% |
| 4  | 1.1237 | 1.1453 | 89.60% | 89.01% |
| 5  | 1.1008 | 1.1367 | 90.27% | 89.27% |
| 6  | 1.0838 | 1.1264 | 90.97% | 89.56% |
| 7  | 1.0722 | 1.1253 | 91.36% | 89.63% |
| 8  | 1.0634 | 1.1206 | 91.47% | 89.83% |
| 9  | 1.0583 | 1.1175 | 91.88% | 89.86% |
| **10** | **1.0564** | **1.1172** | **91.83%** | **89.91%** |

---

### 🟢 exp06 — LoRA r=4, α=4 · Val Acc: 90.11%

| Epoch | Train Loss | Val Loss | Train Acc | Val Acc |
|:-----:|:----------:|:--------:|:---------:|:-------:|
| 1  | 1.8350 | 1.2616 | 71.77% | 86.14% |
| 2  | 1.2061 | 1.1853 | 87.28% | 87.91% |
| 3  | 1.1493 | 1.1603 | 88.86% | 88.71% |
| 4  | 1.1133 | 1.1448 | 89.98% | 89.03% |
| 5  | 1.0916 | 1.1316 | 90.57% | 89.53% |
| 6  | 1.0743 | 1.1223 | 91.22% | 89.53% |
| 7  | 1.0625 | 1.1155 | 91.48% | 89.84% |
| 8  | 1.0545 | 1.1118 | 91.83% | 89.95% |
| 9  | 1.0479 | 1.1108 | 91.94% | 90.06% |
| **10** | **1.0427** | **1.1106** | **92.20%** | **90.11%** |

---

### 🟢 exp07 — LoRA r=4, α=8 · Val Acc: 90.28%

| Epoch | Train Loss | Val Loss | Train Acc | Val Acc |
|:-----:|:----------:|:--------:|:---------:|:-------:|
| 1  | 1.8071 | 1.2468 | 72.30% | 86.36% |
| 2  | 1.1957 | 1.1802 | 87.49% | 88.19% |
| 3  | 1.1382 | 1.1514 | 89.14% | 88.78% |
| 4  | 1.1013 | 1.1310 | 90.30% | 89.45% |
| 5  | 1.0765 | 1.1178 | 91.04% | 89.68% |
| 6  | 1.0586 | 1.1099 | 91.63% | 89.91% |
| 7  | 1.0472 | 1.1042 | 91.93% | 90.16% |
| 8  | 1.0377 | 1.0994 | 92.28% | 90.22% |
| 9  | 1.0292 | 1.0985 | 92.69% | 90.17% |
| **10** | **1.0259** | **1.0976** | **92.76%** | **90.28%** |

---

### 🟢 exp08 — LoRA r=8, α=2 · Val Acc: 90.09%

| Epoch | Train Loss | Val Loss | Train Acc | Val Acc |
|:-----:|:----------:|:--------:|:---------:|:-------:|
| 1  | 1.8510 | 1.2808 | 71.38% | 86.19% |
| 2  | 1.2156 | 1.2013 | 87.20% | 87.85% |
| 3  | 1.1564 | 1.1718 | 88.69% | 88.64% |
| 4  | 1.1211 | 1.1592 | 89.83% | 88.99% |
| 5  | 1.0982 | 1.1440 | 90.52% | 89.40% |
| 6  | 1.0830 | 1.1374 | 91.00% | 89.37% |
| 7  | 1.0705 | 1.1300 | 91.42% | 89.74% |
| **8**  | **1.0636** | **1.1255** | **91.72%** | **90.09%** |
| 9  | 1.0567 | 1.1227 | 91.81% | 89.98% |
| 10 | 1.0553 | 1.1228 | 91.85% | 89.97% |

---

### 🟢 exp09 — LoRA r=8, α=4 · Val Acc: 90.17%

| Epoch | Train Loss | Val Loss | Train Acc | Val Acc |
|:-----:|:----------:|:--------:|:---------:|:-------:|
| 1  | 1.8331 | 1.2605 | 72.02% | 86.22% |
| 2  | 1.2009 | 1.1859 | 87.51% | 88.11% |
| 3  | 1.1430 | 1.1542 | 89.01% | 88.99% |
| 4  | 1.1121 | 1.1388 | 90.04% | 89.11% |
| 5  | 1.0892 | 1.1248 | 90.52% | 89.73% |
| 6  | 1.0720 | 1.1175 | 91.18% | 89.92% |
| 7  | 1.0570 | 1.1145 | 91.80% | 89.92% |
| 8  | 1.0483 | 1.1072 | 92.03% | 90.06% |
| **9**  | **1.0421** | **1.1065** | **92.21%** | **90.17%** |
| 10 | 1.0402 | 1.1059 | 92.26% | 90.17% |

---

### ⭐ exp10 — LoRA r=8, α=8 · Val Acc: **90.46% — BEST**

| Epoch | Train Loss | Val Loss | Train Acc | Val Acc |
|:-----:|:----------:|:--------:|:---------:|:-------:|
| 1  | 1.7796 | 1.2422 | 73.07% | 86.67% |
| 2  | 1.1889 | 1.1817 | 87.61% | 88.27% |
| 3  | 1.1324 | 1.1451 | 89.23% | 89.24% |
| 4  | 1.1005 | 1.1352 | 90.16% | 89.50% |
| 5  | 1.0728 | 1.1164 | 90.98% | 89.95% |
| 6  | 1.0544 | 1.1108 | 91.69% | 90.21% |
| 7  | 1.0418 | 1.1069 | 92.04% | 90.24% |
| 8  | 1.0316 | 1.1042 | 92.55% | 90.22% |
| **9**  | **1.0242** | **1.0997** | **92.82%** | **90.46% ⭐** |
| 10 | 1.0222 | 1.0997 | 92.93% | 90.44% |

---

## 🔍 Optuna HPO — Best Trial

Searched: rank ∈ {2,4,8} · alpha ∈ {2,4,8} · dropout ∈ [0.05, 0.30] · 10 trials

**Best: rank=8, alpha=8, dropout=0.30 → 90.39%**

| Epoch | Train Loss | Val Loss | Train Acc | Val Acc |
|:-----:|:----------:|:--------:|:---------:|:-------:|
| 1  | 1.7887 | 1.2378 | 72.76% | 86.26% |
| 2  | 1.1934 | 1.1692 | 87.44% | 88.13% |
| 3  | 1.1345 | 1.1437 | 89.13% | 89.13% |
| 4  | 1.1009 | 1.1261 | 90.20% | 89.47% |
| 5  | 1.0768 | 1.1173 | 90.89% | 89.88% |
| 6  | 1.0568 | 1.1096 | 91.55% | 90.12% |
| 7  | 1.0456 | 1.1056 | 91.90% | 90.24% |
| 8  | 1.0354 | 1.1019 | 92.32% | 90.33% |
| 9  | 1.0285 | 1.1006 | 92.51% | 90.38% |
| **10** | **1.0239** | **1.0991** | **92.88%** | **90.39%** |

---

## 📈 Key Findings

- **LoRA vs Baseline:** +9.69 pp accuracy gain with only 1.18% trainable parameters
- **Rank effect:** Higher rank → better accuracy (r=8 > r=4 > r=2)
- **Alpha effect:** Higher alpha → better accuracy at same rank
- **Best ratio:** alpha/rank = 1.0 (scaling = 1.0) works best
- **Optuna confirms:** rank=8, alpha=8 is the optimal configuration

---

## 🚀 Load `best_model.pt`

```python
import torch
from transformers import ViTForImageClassification, ViTImageProcessor
from peft import LoraConfig, get_peft_model
from huggingface_hub import hf_hub_download
from PIL import Image

REPO = "MSG1999/vit-lora-cifar100"
BASE = "WinKawaks/vit-small-patch16-224"

# 1. Build model with same LoRA config used during training
base = ViTForImageClassification.from_pretrained(
    BASE, num_labels=100, ignore_mismatched_sizes=True)
model = get_peft_model(base, LoraConfig(
    r=8, lora_alpha=8, lora_dropout=0.1,
    target_modules=["query", "key", "value"], bias="none",
))

# 2. Load best_model.pt weights
ckpt = hf_hub_download(repo_id=REPO, filename="best_model.pt")
model.load_state_dict(torch.load(ckpt, map_location="cpu"), strict=False)
model.eval()
print("✓ best_model.pt loaded!")

# 3. Inference
processor = ViTImageProcessor.from_pretrained(BASE)
image = Image.open("your_image.jpg").convert("RGB")
inputs = processor(images=image, return_tensors="pt")
with torch.no_grad():
    pred = model(**inputs).logits.argmax(-1).item()
print(f"Predicted class: {pred}")
```

---
---

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PART 2 — Q2: Adversarial Attacks with IBM ART
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

## 📌 Overview

Training **ResNet-18** from scratch on **CIFAR-10**, attacking with **FGSM** (custom + IBM ART), and training **ResNet-34 binary detectors** for PGD and BIM adversarial examples.

| Task | Model | Result | Target |
|------|-------|:------:|:------:|
| Part (i) Train | ResNet-18 from scratch | **94.68%** | ≥ 72% ✅ |
| Part (i) FGSM | Custom vs IBM ART | See table | — |
| Part (ii) PGD Detector | ResNet-34 binary | **99.93%** | ≥ 70% ✅ |
| Part (ii) BIM Detector | ResNet-34 binary | **99.57%** | ≥ 70% ✅ |

---

## ⚙️ Q2 Hyperparameters

### ResNet-18 Training

| Parameter | Value |
|-----------|-------|
| Model | ResNet-18 (from scratch) |
| Dataset | CIFAR-10 (50k train / 10k test, 10 classes) |
| Optimizer | SGD + Cosine Decay |
| Learning Rate | `0.1` |
| Batch Size | 128 |
| Epochs | 80 |

### Adversarial Detector

| Parameter | Value |
|-----------|-------|
| Model | ResNet-34 (binary classifier) |
| Optimizer | AdamW + CosineAnnealingLR |
| Learning Rate | `0.001` |
| Batch Size | 128 |
| Epochs | 30 |

---

## 📊 Q2 Part (i) — ResNet-18 Training · Val Acc: 94.68%

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc |
|:-----:|:----------:|:---------:|:--------:|:-------:|
| 10 | 0.5483 | 81.08% | 0.7184 | 76.04% |
| 20 | 0.3793 | 86.97% | 0.5719 | 80.96% |
| 40 | 0.2260 | 92.29% | 0.3451 | 88.93% |
| 60 | 0.0614 | 97.95% | 0.2481 | 92.68% |
| **80** | **0.0033** | **99.96%** | **0.2059** | **94.62%** |

---

## ⚔️ Q2 Part (i) — FGSM Attack Results

Clean Accuracy: **94.25%**

| Epsilon (ε) | Clean Acc | FGSM Scratch | FGSM ART | Drop (Scratch) | Drop (ART) |
|:-----------:|:---------:|:------------:|:--------:|:--------------:|:----------:|
| 0.01 | 94.25% | 48.70% | 52.55% | 45.55% | 41.70% |
| 0.02 | 94.25% | 42.00% | 45.45% | 52.25% | 48.80% |
| 0.03 | 94.25% | 40.35% | 43.35% | 53.90% | 50.90% |
| 0.05 | 94.25% | 33.80% | 35.90% | 60.45% | 58.35% |
| 0.10 | 94.25% | 16.80% | 17.45% | 77.45% | 76.80% |
| 0.20 | 94.25% | 12.05% | 12.20% | 82.20% | 82.05% |
| 0.30 | 94.25% | 9.90% | 9.95% | 84.35% | 84.30% |

---

## 🛡️ Q2 Part (ii) — PGD Detector · 99.93% ✅

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc |
|:-----:|:----------:|:---------:|:--------:|:-------:|
| 1  | 0.7549 | 50.11% | 0.7230 | 48.70% |
| 6  | 0.6673 | 56.81% | 0.5546 | 74.67% |
| 10 | 0.0320 | 98.81% | 0.0264 | 99.07% |
| 15 | 0.0036 | 99.90% | 0.0049 | 99.87% |
| **30** | **0.0003** | **99.99%** | **0.0009** | **99.93%** |

## 🛡️ Q2 Part (ii) — BIM Detector · 99.57% ✅

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc |
|:-----:|:----------:|:---------:|:--------:|:-------:|
| 1  | 0.7565 | 49.76% | 0.6950 | 49.00% |
| 5  | 0.0652 | 97.59% | 0.0724 | 97.57% |
| 9  | 0.0233 | 99.16% | 0.0202 | 99.37% |
| 14 | 0.0124 | 99.61% | 0.0179 | 99.50% |
| **30** | **0.0002** | **100.00%** | **0.0207** | **99.57%** |

---

## 🚀 Q2 Load Weights

```python
from huggingface_hub import hf_hub_download
import torch, torchvision, os

os.makedirs("weights", exist_ok=True)
for f in ["resnet18_cifar10_best.pt", "detector_PGD_best.pt", "detector_BIM_best.pt"]:
    hf_hub_download("MSG1999/DLOps-A5-Q2-ART", f, local_dir="weights")
    print(f"Downloaded → {f}")

# Load ResNet-18
model = torchvision.models.resnet18()
model.fc = torch.nn.Linear(512, 10)
model.load_state_dict(torch.load("weights/resnet18_cifar10_best.pt", map_location="cpu"))
model.eval()
print("ResNet-18 loaded! (94.68% val acc)")
```

---

## 📁 Repository Files

### Q1 — `MSG1999/vit-lora-cifar100`

| File | Size | Description |
|------|:----:|-------------|
| `best_model.pt` | 87.9 MB | ✅ Full best model weights (exp10_r8_a8) |
| `adapter_model.safetensors` | 895 KB | LoRA adapter weights |
| `adapter_config.json` | 881 B | LoRA configuration |
| `preprocessor_config.json` | 351 B | Image processor config |
| `q1_summary.json` | 3.34 KB | All 10 experiment results |
| `optuna_best_params.json` | 97 B | Optuna HPO best params |
| `final_results.csv` | 1.14 KB | Results table (CSV) |

### Q2 — `MSG1999/DLOps-A5-Q2-ART`

| File | Description |
|------|-------------|
| `resnet18_cifar10_best.pt` | ResNet-18 best weights (94.68%) |
| `detector_PGD_best.pt` | PGD detector (99.93%) |
| `detector_BIM_best.pt` | BIM detector (99.57%) |

---

## 📜 Citation

```bibtex
@misc{gadiya2026dlops,
  title  = {DLOps Assignment 5 — ViT-S LoRA + Adversarial Robustness},
  author = {Mahek Gadiya},
  year   = {2026},
  note   = {IIT Jodhpur},
  url    = {https://huggingface.co/MSG1999/vit-lora-cifar100}
}
```

---

<div align="center">
DLOps A5 · IIT Jodhpur · <a href="https://huggingface.co/MSG1999">MSG1999</a>
</div>
