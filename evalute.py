import os
import json
import numpy as np
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)
from sklearn.metrics import confusion_matrix, classification_report
from data import load_dataset
from utils import compute_metrics

# ==============================
# CHANGE THIS
# ==============================
MODEL_NAME = "MSG1999/ml-ops-3"
RESULTS_DIR = "./results"
NUM_LABELS = 4
# ==============================


def tokenize_function(example, tokenizer):
    return tokenizer(
        example["text"],
        padding="max_length",
        truncation=True,
        max_length=64
    )


def main():

    # -----------------------------
    # Load dataset (only val needed)
    # -----------------------------
    train_dataset, val_dataset = load_dataset()

    # -----------------------------
    # Load tokenizer from HF
    # -----------------------------
    print("Downloading tokenizer from Hugging Face...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # -----------------------------
    # Tokenize validation dataset
    # -----------------------------
    val_dataset = val_dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True
    )

    val_dataset.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "label"]
    )

    # -----------------------------
    # Load model from HF
    # -----------------------------
    print("Downloading model from Hugging Face...")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS
    )

    # -----------------------------
    # Evaluation arguments
    # -----------------------------
    training_args = TrainingArguments(
        output_dir="./eval_temp",
        per_device_eval_batch_size=2,
        report_to="none",
        fp16=False
    )

    # -----------------------------
    # Trainer (NO training dataset)
    # -----------------------------
    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics
    )

    # -----------------------------
    # Evaluate
    # -----------------------------
    print("Running evaluation...")
    metrics = trainer.evaluate()
    print("Evaluation Results:", metrics)

    # -----------------------------
    # Save results
    # -----------------------------
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Save metrics
    with open(os.path.join(RESULTS_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    # Predictions
    predictions = trainer.predict(val_dataset)
    preds = np.argmax(predictions.predictions, axis=-1)
    labels = predictions.label_ids

    # Confusion Matrix
    cm = confusion_matrix(labels, preds)
    np.save(os.path.join(RESULTS_DIR, "confusion_matrix.npy"), cm)

    # Classification Report
    report = classification_report(labels, preds)
    with open(os.path.join(RESULTS_DIR, "classification_report.txt"), "w") as f:
        f.write(report)

    print("Metrics and reports saved successfully!")


if __name__ == "__main__":
    main()
