import os
import numpy as np
import json
import pickle
import torch
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments
)
import evaluate
from sklearn.metrics import classification_report
from data import load_and_prepare_data

# Disable wandb logging
os.environ["WANDB_DISABLED"] = "true"

# -----------------------------
# Configuration
# -----------------------------
MODEL_NAME = "bert-goodreads-genres"
OUTPUT_DIR = "models"
RESULTS_DIR = "results"
HF_REPO = "MSG1999/bert-goodreads-genres"
MAX_LENGTH = 512
TRAIN_SIZE = 800  # Reviews per genre for training
TEST_SIZE = 200   # Reviews per genre for testing

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# -----------------------------
# Check GPU availability
# -----------------------------
print("="*50)
print("Hardware Configuration")
print("="*50)
if torch.cuda.is_available():
    print(f"✓ GPU available: {torch.cuda.get_device_name(0)}")
    print(f"  Number of GPUs: {torch.cuda.device_count()}")
    print(f"  CUDA Version: {torch.version.cuda}")
else:
    print("⚠ No GPU detected - training will use CPU (slower)")
print("="*50 + "\n")

# -----------------------------
# Load tokenizer
# -----------------------------
print("Loading tokenizer...")
tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)

# -----------------------------
# Load dataset
# -----------------------------
print("Loading Goodreads dataset...")
train_dataset, test_dataset, label_encoder = load_and_prepare_data(
    tokenizer=tokenizer,
    max_length=MAX_LENGTH,
    train_size=TRAIN_SIZE,
    test_size=TEST_SIZE,
    force_reload=False
)

num_labels = len(label_encoder.classes_)
print(f"\nNumber of genres: {num_labels}")
print(f"Genres: {list(label_encoder.classes_)}")

# Save label encoder for later use
os.makedirs("data", exist_ok=True)
with open("data/label_encoder.pickle", 'wb') as f:
    pickle.dump(label_encoder, f)

# Create label mappings
id2label = {i: label for i, label in enumerate(label_encoder.classes_)}
label2id = {label: i for i, label in enumerate(label_encoder.classes_)}

# -----------------------------
# Load model
# -----------------------------
print("\nLoading model...")
model = DistilBertForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=num_labels,
    id2label=id2label,
    label2id=label2id
)

# Verify model device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Model will use device: {device}")
if torch.cuda.is_available():
    print(f"✓ Model will train on CUDA GPU")
else:
    print(f"⚠ Model will train on CPU")

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
# Training arguments
# -----------------------------
print("\nSetting up training arguments...")
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_steps=10,
    logging_dir='./logs',
    load_best_model_at_end=True,
    metric_for_best_model='accuracy',
    push_to_hub=False,
    save_total_limit=2
)

# -----------------------------
# Trainer
# -----------------------------
print("\nInitializing Trainer...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
)

# -----------------------------
# Train
# -----------------------------
print("\n" + "="*50)
print("Starting training...")
if torch.cuda.is_available():
    print(f"✓ Training on GPU: {torch.cuda.get_device_name(0)}")
else:
    print("⚠ Training on CPU")
print("="*50 + "\n")
trainer.train()

# -----------------------------
# Evaluate
# -----------------------------
print("\n" + "="*50)
print("Evaluating model...")
print("="*50 + "\n")
eval_results = trainer.evaluate()

print("\nEvaluation Results:")
for key, value in eval_results.items():
    print(f"  {key}: {value}")

# Save evaluation results
results_path = os.path.join(RESULTS_DIR, "eval_results.json")
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
report_path = os.path.join(RESULTS_DIR, "classification_report.json")
with open(report_path, "w") as f:
    json.dump(report_dict, f, indent=4)
print(f"Classification report saved to {report_path}")

# -----------------------------
# Save model
# -----------------------------
print("\n" + "="*50)
print("Saving model...")
print("="*50 + "\n")
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

# Save config mapping
label2id = {label: i for i, label in enumerate(label_encoder.classes_)}
config_path = os.path.join(OUTPUT_DIR, "config_labels.json")
with open(config_path, "w") as f:
    json.dump({"id2label": id2label, "label2id": label2id}, f, indent=4)

print(f"Model saved to {OUTPUT_DIR}")

# -----------------------------
# Push to Hugging Face Hub (Optional)
# -----------------------------
print("\n" + "="*50)
print("Upload to Hugging Face Hub")
print("="*50 + "\n")
push_to_hub = input("Push model to Hugging Face Hub? (yes/no): ").strip().lower()

if push_to_hub == 'yes':
    try:
        print(f"Pushing model to {HF_REPO}...")
        trainer.push_to_hub(
            repo_id=HF_REPO,
            commit_message="Fine-tuned DistilBERT for Goodreads genre classification"
        )
        tokenizer.push_to_hub(HF_REPO)
        print(f"\n✓ Model successfully pushed to https://huggingface.co/{HF_REPO}")
    except Exception as e:
        print(f"\n✗ Error pushing to hub: {e}")
        print("You may need to login first: huggingface-cli login")
else:
    print("Skipping push to Hugging Face Hub.")

print("\n" + "="*50)
print("Training complete!")
print("="*50)
