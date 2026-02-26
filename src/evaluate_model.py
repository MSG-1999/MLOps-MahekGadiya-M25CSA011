import os
import json
import numpy as np
import pickle
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    Trainer
)
import evaluate
from sklearn.metrics import classification_report
from data import load_and_prepare_data

# Disable wandb logging
os.environ["WANDB_DISABLED"] = "true"

# -----------------------------
# Configuration
# -----------------------------
HF_MODEL_NAME = "MSG1999/bert-goodreads-genres"  # Load from HuggingFace
RESULTS_DIR = "results"
MAX_LENGTH = 512
TEST_SIZE = 200

os.makedirs(RESULTS_DIR, exist_ok=True)

# -----------------------------
# Load model and tokenizer from Hugging Face
# -----------------------------
print("="*50)
print(f"Loading model from Hugging Face: {HF_MODEL_NAME}")
print("="*50 + "\n")

try:
    tokenizer = DistilBertTokenizerFast.from_pretrained(HF_MODEL_NAME)
    model = DistilBertForSequenceClassification.from_pretrained(HF_MODEL_NAME)
    print(f"✓ Model successfully loaded from https://huggingface.co/{HF_MODEL_NAME}\n")
except Exception as e:
    print(f"✗ Error loading model from HuggingFace: {e}")
    print("Falling back to locally saved model...\n")
    MODEL_DIR = "models"
    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_DIR)
    model = DistilBertForSequenceClassification.from_pretrained(MODEL_DIR)
    print("✓ Model loaded from local directory\n")

# -----------------------------  
# Load dataset
# -----------------------------
print("="*50)
print("Loading Goodreads test dataset...")
print("="*50 + "\n")

_, test_dataset, label_encoder = load_and_prepare_data(
    tokenizer=tokenizer,
    max_length=MAX_LENGTH,
    test_size=TEST_SIZE,
    force_reload=False
)

print(f" Test samples: {len(test_dataset)}")
print(f"Number of genres: {len(label_encoder.classes_)}")
print(f"Genres: {list(label_encoder.classes_)}\n")

# -----------------------------
# Metrics
# -----------------------------
accuracy_metric = evaluate.load("accuracy")
precision_metric = evaluate.load("precision")
recall_metric = evaluate.load("recall")
f1_metric = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)

    # Use macro averaging for multi-class classification
    accuracy = accuracy_metric.compute(predictions=predictions, references=labels)
    precision = precision_metric.compute(predictions=predictions, references=labels, average="macro")
    recall = recall_metric.compute(predictions=predictions, references=labels, average="macro")
    f1 = f1_metric.compute(predictions=predictions, references=labels, average="macro")

    return {
        "accuracy": accuracy["accuracy"],
        "precision": precision["precision"],
        "recall": recall["recall"],
        "f1": f1["f1"],
    }

# -----------------------------
# Trainer for evaluation only
# -----------------------------
trainer = Trainer(
    model=model,
    compute_metrics=compute_metrics,
)

# -----------------------------
# Evaluate
# -----------------------------
print("="*50)
print("Evaluating model on test set...")
print("="*50 + "\n")

eval_results = trainer.evaluate(eval_dataset=test_dataset)

print("\nOverall Evaluation Results:")
for key, value in eval_results.items():
    print(f"  {key}: {value}")

# Save evaluation results
results_path = os.path.join(RESULTS_DIR, "hf_eval_results.json")
with open(results_path, "w") as f:
    json.dump(eval_results, f, indent=4)
print(f"\nResults saved to {results_path}")

# Get predictions for detailed analysis
print("\nGenerating predictions for detailed analysis...")
predicted_results = trainer.predict(test_dataset)
predicted_labels = predicted_results.predictions.argmax(-1)

# Get true labels
true_labels = predicted_results.label_ids

# Convert numeric labels to genre names
id2label = {i: label for i, label in enumerate(label_encoder.classes_)}
true_label_names = [id2label[l] for l in true_labels]
predicted_label_names = [id2label[l] for l in predicted_labels]

# Print detailed classification report
print("\n" + "="*50)
print("Detailed Classification Report:")
print("="*50 + "\n")
print(classification_report(true_label_names, predicted_label_names))

# Save classification report
report_dict = classification_report(true_label_names, predicted_label_names, output_dict=True)
report_path = os.path.join(RESULTS_DIR, "hf_classification_report.json")
with open(report_path, "w") as f:
    json.dump(report_dict, f, indent=4)
print(f"Classification report saved to {report_path}")

print("\n" + "="*50)
print("Evaluation complete!")
print("="*50)

print(f"\nResults saved to {results_path}")
