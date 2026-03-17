## Results :

# 1) Best Configuration Found: (Ray Tune + Optuna + ASHA)

|     Hyperparameter    |     Best   Value    |     Notes                                                               |
|-----------------------|---------------------|-------------------------------------------------------------------------|
|     lr                |     0.000101        |     Low stable LR   with cosine decay to near-zero over 30 epochs       |
|     batch_size        |     64              |     Best balance of   gradient quality and training speed               |
|     num_heads         |     4               |     Fewer heads;   large d_ff compensates for attention capacity        |
|     d_ff              |     4096            |     Largest FFN —   richer feature representations, key to BLEU gain    |
|     dropout           |     0.054           |     Very low dropout;   13k pairs sufficient to avoid over-fitting      |
|     weight_decay      |     0.000263        |     Mild AdamW L2   regularisation                                      |


# 2) Tuned Model vs Baseline (Part 3 – Efficiency Challenge)

| Metric               | Baseline | Tuned  | Improvement     |
|----------------------|----------|--------|-----------------|
| Epochs               | 100      | 30     | ⬇️ 70% reduction |
| Sweep Time (8 GPUs)  | –        | 49.98m | –               |
| Retrain Time (GPU 0) | –        | 29.33m | –               |
| Total Training Time  | 129.42m  | 79.31m | 1.6x Speedup    |
| Final Loss           | 0.0972   | 0.0959 | ⬇️ 1.3%          |
| BLEU Score           | 68.02%   | 90.38% | ⬆️ 22.36pp       |


3) Model Link : https://huggingface.co/MSG1999/m25csa011-transformer-english-hindi
