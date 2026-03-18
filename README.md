
# 🚀 English → Hindi Transformer

![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red)
![Task](https://img.shields.io/badge/Task-Translation-green)
![BLEU](https://img.shields.io/badge/BLEU-90.38-brightgreen)
![Optimization](https://img.shields.io/badge/RayTune+Optuna-Used-orange)

---

## 📌 Overview

This model implements a **Transformer-based Neural Machine Translation (NMT)** system for English → Hindi translation using PyTorch.

Optimized using **Ray Tune + Optuna + ASHA**, achieving high BLEU score with reduced training time.

---

## Results :

# 1) Best Configuration Found: (Ray Tune + Optuna + ASHA)

|     Hyperparameter    |     Best   Value    |     Notes                                                               |
|-----------------------|---------------------|-------------------------------------------------------------------------|
|     lr                |     0.00010092277088632719        |     Low stable LR   with cosine decay to near-zero over 30 epochs       |
|     batch_size        |     64              |     Best balance of   gradient quality and training speed               |
|     num_heads         |     4               |     Fewer heads;   large d_ff compensates for attention capacity        |
|     d_ff              |     4096            |     Largest FFN —   richer feature representations, key to BLEU gain    |
|     dropout           |     0.05440091406982377           |     Very low dropout;   13k pairs sufficient to avoid over-fitting      |
|     weight_decay      |     0.0002615454502348676        |     Mild AdamW L2   regularisation                                      |

__________________________________
# 2) Tuned Model vs Baseline: (Efficiency Challenge)

| Metric               | Baseline | Tuned  | Improvement     |
|----------------------|----------|--------|-----------------|
| Epochs               | 100      | 30     | ⬇️ 70% reduction |
| Sweep Time (8 GPUs)  | –        | 49.98m | –               |
| Retrain Time (GPU 0) | –        | 29.33m | –               |
| Total Training Time  | 129.42m  | 79.31m | 1.6x Speedup    |
| Final Loss           | 0.0972   | 0.0959 | ⬇️ 1.3%          |
| BLEU Score           | 68.02%   | 90.38% | ⬆️ 22.36pp       |

__________________________________

## 3) Model link (M25CSA011_ass_4_best_model.pth): https://huggingface.co/MSG1999/m25csa011-transformer-english-hindi

# __________________________________
## 🏗️ Model Architecture

* Encoder–Decoder Transformer
* 6 Encoder + 6 Decoder layers
* Multi-head attention
* Feed-forward network
* Positional encoding
* Residual connections + LayerNorm

### Configuration

* d_model: 512
* num_heads: 4
* num_layers: 6
* d_ff: 4096
* dropout: 0.054

---

## ⚙️ Training Details

* Dataset: ~13,186 English-Hindi sentence pairs
* Optimizer: AdamW
* Loss: CrossEntropy (ignore padding)
* Scheduler: CosineAnnealingLR
* Device: GPU

---

## 📊 Results

| Metric | Baseline   | Tuned      |
| ------ | ---------- | ---------- |
| Epochs | 100        | 30         |
| Time   | 129.42 min | 79.31 min  |
| Loss   | 0.0972     | **0.0959** |
| BLEU   | 68.02      | **90.38**  |

---

## ⚙️ Best Hyperparameters

* LR: 0.0001009
* Batch size: 64
* Heads: 4
* d_ff: 4096
* Dropout: 0.054
* Weight decay: 0.000261

---

## 🧪 Sample Outputs

* EN: I love you  
  HI: मैं तुमसे प्यार करता हूँ

* EN: What is your name?   
  HI: आपका नाम क्या है?

* EN: How are you?   
  HI: आप कैसे हैं?

---

## 📂 Files

* M25CSA011_ass_4_best_model.pth
* en_vocab.pkl
* hi_vocab.pkl
* best_config.json

---

## 👤 Author

Mahek Shankesh Gadiya
M.Tech AI – IIT Jodhpur

---

## 📚 Assignment

Transformer Optimization using Ray Tune + Optuna

