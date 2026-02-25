import os
import json
import numpy as np
from transformers import (
    autotokenizer,
    automodelforsequenceclassification,
    trainer,
    trainingarguments
)
from sklearn.metrics import confusion_matrix, classification_report
from data import load_dataset
from utils import compute_metrics

model_name = "distilbert-base-uncased"
output_dir = "./model"
results_dir = "./results"
num_labels = 4  # 4 genres


def tokenize_function(example, tokenizer):
    return tokenizer(
        example["text"],
        padding="max_length",
        truncation=true,
        max_length=64
    )


def main():

    # -----------------------------
    # load dataset
    # -----------------------------
    train_dataset, val_dataset = load_dataset()

    # -----------------------------
    # load tokenizer
    # -----------------------------
    tokenizer = autotokenizer.from_pretrained(model_name)

    # -----------------------------
    # tokenize datasets
    # -----------------------------
    train_dataset = train_dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=true
    )

    val_dataset = val_dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=true
    )

    train_dataset.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "label"]
    )

    val_dataset.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "label"]
    )

    # -----------------------------
    # load model
    # -----------------------------
    model = automodelforsequenceclassification.from_pretrained(
        model_name,
        num_labels=num_labels
    )

    # -----------------------------
    # training arguments
    # -----------------------------
    training_args = trainingarguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        num_train_epochs=3,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        logging_dir="./logs",
        load_best_model_at_end=true,
        report_to="none",
        fp16=true
    )

    # -----------------------------
    # trainer
    # -----------------------------
    trainer = trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics
    )

    # -----------------------------
    # train
    # -----------------------------
    trainer.train()

    # -----------------------------
    # evaluate
    # -----------------------------
    metrics = trainer.evaluate()
    print("evaluation results:", metrics)

    # -----------------------------
    # save metrics + reports
    # -----------------------------
    os.makedirs(results_dir, exist_ok=true)

    # save metrics json
    with open(os.path.join(results_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    # predictions
    predictions = trainer.predict(val_dataset)
    preds = np.argmax(predictions.predictions, axis=-1)
    labels = predictions.label_ids

    # confusion matrix
    cm = confusion_matrix(labels, preds)
    np.save(os.path.join(results_dir, "confusion_matrix.npy"), cm)

    # classification report
    report = classification_report(labels, preds)
    with open(os.path.join(results_dir, "classification_report.txt"), "w") as f:
        f.write(report)

    print("metrics and reports saved successfully!")

    # -----------------------------
    # save model + tokenizer
    # -----------------------------
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    print("model saved successfully!")


if __name__ == "__main__":
    main()
