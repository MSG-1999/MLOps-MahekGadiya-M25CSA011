import json
import pandas as pd
from datasets import Dataset
from sklearn.model_selection import train_test_split
import os

DATA_DIR = "data"

GENRE_FILES = {
    "children": "goodreads_reviews_children.json",
    "history": "goodreads_reviews_history_biography.json",
    "mystery": "goodreads_reviews_mystery_thriller_crime.json",
    "young_adult": "goodreads_reviews_young_adult.json"
}

def load_json_file(path, label):
    texts = []
    max_samples = 2000

    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= max_samples:
                break

            obj = json.loads(line)
            if "review_text" in obj and obj["review_text"].strip() != "":
                texts.append({
                    "text": obj["review_text"],
                    "label": label
                })

    return texts

def load_dataset():
    all_data = []

    for idx, (genre, filename) in enumerate(GENRE_FILES.items()):
        file_path = os.path.join(DATA_DIR, filename)
        print(f"Loading {genre}...")
        all_data.extend(load_json_file(file_path, idx))

    df = pd.DataFrame(all_data)

    train_df, val_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df["label"]
    )

    train_dataset = Dataset.from_pandas(train_df.reset_index(drop=True))
    val_dataset = Dataset.from_pandas(val_df.reset_index(drop=True))

    return train_dataset, val_dataset
