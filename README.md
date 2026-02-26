# 📚 Goodreads Genre Classification

## Fine-Tuning DistilBERT --- End-to-End MLOps Pipeline

**Mahek Gadiya · M25CSA011 · MLOps Assignment 3**

This project implements a complete MLOps workflow for multi-class genre
classification using DistilBERT, including training, evaluation, Hugging
Face deployment, and Docker-based reproducibility.

------------------------------------------------------------------------

## 🎯 Overview

-   Fine-tuned `distilbert-base-cased` for 8 Goodreads genres
-   Evaluated locally with macro metrics
-   Uploaded model to Hugging Face Hub
-   Re-evaluated from Hugging Face to verify deployment
-   Containerized training (GPU) and evaluation (CPU) using Docker

------------------------------------------------------------------------

## 🗂️ Project Structure
```
Assignment_3/
├── src/
│   ├── data.py                 # Data loading & preprocessing
│   ├── train.py                # Training pipeline with HF Hub upload
│   └── evaluate_model.py       # Evaluation from HF Hub
├── results/                            
│   ├── classification_report.json      # Per-genre metrics
│   ├── eval_results.json               # Training evaluation metrics
│   ├── hf_classification_report.json   # Per-genre metrics
│   ├── hf_eval_results.json            # HuggingFace model evaluation
│   └── plots/                
│       ├── classwise_f1_score.png
│       ├── model_comparison.png
│       ├── overall_performance.png
│       └── precision_vs_recall.png
├── ML_DL_Ops_Ass_3_Fine_Tuning_Classification.ipynb
├── Dockerfile.train            # Training container (GPU)
├── Dockerfile.eval             # Evaluation container (CPU)
├── requirements.txt
└── README.md
```
------------------------------------------------------------------------

## 📊 Results & Model Performance

### 1) Overall Metrics (Local vs Hugging Face Evaluation)

| Metric  | Score | HF Hub  |
|---|---|---|
| Accuracy | **0.61** | **0.61** |
| Precision (Macro) | 0.607 | 0.607 |
| Recall (Macro) | 0.61 | 0.61 |
| F1-Score (Macro) | 0.607 | 0.607 |

### 2) Per-Class Performance (Local vs Hugging Face Evaluation)

| Genre                     | Precision | Recall | F1-Score |
|---------------------------|-----------|--------|----------|
| Children                  | 0.65      | 0.62   | 0.63     |
| Comics & Graphic          | 0.80      | 0.80   | *0.80* |
| Fantasy & Paranormal      | 0.45      | 0.47   | 0.46     |
| History & Biography       | 0.58      | 0.54   | 0.56     |
| Mystery/Thriller/Crime    | 0.57      | 0.62   | 0.59     |
| Poetry                    | 0.77      | 0.81   | *0.79* |
| Romance                   | 0.61      | 0.66   | 0.63     |
| Young Adult               | 0.42      | 0.36   | 0.39     |

### 3) Note : 
      i) All metrics show **0.0000 difference** between local and hub models
      ii) this confirm's that correct model uploaded and downloaded.

------------------------------------------------------------------------

### Best & Weakest Genres

-   🏆 Best: Comics & Graphic (0.80 -> F1)
-   🏆 Strong: Poetry (0.79 -> F1)
-   ⚠ Most challenging: Young Adult (0.39 -> F1)

Note : Balanced macro and weighted averages indicate uniform class distribution.

------------------------------------------------------------------------

## 🐳 Docker Build Instruction
```
### Build Training Image

docker build -f Dockerfile.train -t mahek-train .

### Run Training :

docker run --rm -v \$(pwd)/models:/app/models -v
\$(pwd)/results:/app/results mahek-train

### Build Evaluation Image :

docker build -f Dockerfile.eval -t mahek-eval .

### Run Evaluation :

docker run --rm -v \$(pwd)/results:/app/results mahek-eval

```
------------------------------------------------------------------------

### Training Hyperparameters

| Parameter | Value |
|---|---|
| Epochs | 3 |
| Train Batch Size | 8 |
| Eval Batch Size | 16 |
| Learning Rate | 2e-5 |
| Weight Decay | 0.01 |
| Max Token Length | 512 |
| Optimizer | AdamW |
| Training Time | ~ 40-50 min (GPU) |

## 📈 Evaluation Visualizations

The pipeline auto-generates 4 evaluation plots in `results/plots/`:

| Plot | Description |
|---|---|
| `classwise_f1 score.png` | Classwise_f1_score Comparsion |
| `model_comparsion` | model_comparsion |
| `overall_performance.png` | Overall metrics bar chart |
| `precision_vs_recall.png` |precision vs recall |


------------------------------------------------------------------------

## 🎓 Key Learnings

-   Transformer fine-tuning workflow
-   Multi-class macro evaluation
-   Docker-based ML deployment
-   Hugging Face model versioning
-   Reproducible ML pipelines

------------------------------------------------------------------------

## 📎 Submission Links

🤗 Hugging Face Model:\
https://huggingface.co/MSG1999/bert-goodreads-genres

📦 GitHub Repository:\
(https://github.com/MSG-1999/MLOps-MahekGadiya-M25CSA011/edit/Assignment_3)

------------------------------------------------------------------------

© 2026 Mahek Gadiya
