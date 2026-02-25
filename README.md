# ML-DLOps Minor Exam — SET A

**Topics:** [Docker-A]

---

## Data Structure

```
data/
├── train/   →  0/  1/  2/  3/  4/  5/  6/  7/  8/  9/
└── test/    →  0/  1/  2/  3/  4/  5/  6/  7/  8/  9/
```

---

## Q1 [Docker-A] — Test Set Results

| Metric | Value |
|---|---|
| Overall Accuracy | _fill after running evaluate.py_ |
| F1 Score | _fill after running evaluate.py_ |
| Class 5 Accuracy | _fill after running evaluate.py_ |
| Output for Data/5/0005.jpg | 7 |

---

## Q1(5) — Docker Commands

### Build the image
```bash
docker build -t mlops-exam .
```

### Run the container (evaluate)
```bash
docker run --rm -v $(pwd)/data:/app/data -v $(pwd)/model.pth:/app/model.pth mlops-exam
```

---

## Q1(7) — New Container Without Dockerfile

### Container creation
```bash
docker run -it --name mlops_scratch -v $(pwd)/data:/app/data -w /app python:3.10-slim bash
```

### Dependencies installation (inside container)
```bash
pip install torch torchvision matplotlib seaborn scikit-learn Pillow numpy
```

### Run training inside container
```bash
python train.py
```

---

## Q7 — Hyperparameter Analysis

| Run | Learning Rate | Optimizer | Batch Size | Val Accuracy | Notes |
|---|---|---|---|---|---|
| 1 (Baseline) | 0.001 | adam | 32 | _fill_ | Balanced convergence |
| 2 | 0.01 | sgd | 64 | _fill_ | Faster but unstable |
| 3 | 0.0001 | adam | 16 | _fill_ | Slow but stable |

### Analysis
_Write your observations here after running all 3 experiments._
- Run 1 (adam, lr=0.001): 
- Run 2 (sgd, lr=0.01): 
- Run 3 (adam, lr=0.0001): 

### Best Setting
- Learning Rate: ___
- Optimizer: ___
- Batch Size: ___

### Best Setting Results
| Metric | Value |
|---|---|
| Overall Accuracy | _fill_ |
| Class 5 Accuracy | _fill_ |

---

## Files

| File | Description |
|---|---|
| `train.py` | Training script with custom ImageDataset, plots training curves |
| `evaluate.py` | Evaluation script with classwise accuracy + confusion matrix |
| `Dockerfile` | Docker image definition |
| `requirements.txt` | Python dependencies |
| `model.pth` | Trained model weights |
| `training_curves_*.png` | Training curves for each hyperparameter run |
| `confusion_matrix.png` | Confusion matrix on test set |
