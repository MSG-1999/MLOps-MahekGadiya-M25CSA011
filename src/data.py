import json
import gzip
import random
import pickle
import os
import requests
from sklearn.preprocessing import LabelEncoder
import torch

# Goodreads genre URLs
GENRE_URL_DICT = {
    'poetry': 'https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_poetry.json.gz',
    'children': 'https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_children.json.gz',
    'comics_graphic': 'https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_comics_graphic.json.gz',
    'fantasy_paranormal': 'https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_fantasy_paranormal.json.gz',
    'history_biography': 'https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_history_biography.json.gz',
    'mystery_thriller_crime': 'https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_mystery_thriller_crime.json.gz',
    'romance': 'https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_romance.json.gz',
    'young_adult': 'https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_young_adult.json.gz'
}


def load_reviews(url, head=10000, sample_size=2000):
    reviews = []
    count = 0

    response = requests.get(url, stream=True)
    print("Response: " + str(response.status_code))

    with gzip.open(response.raw, 'rt', encoding='utf-8') as file:
        for line in file:
            d = json.loads(line)
            reviews.append(d['review_text'])
            count += 1

            if head is not None and count >= head:
                break

    return random.sample(reviews, min(sample_size, len(reviews)))


def load_goodreads_data(cache_file='data/genre_reviews_dict.pickle',
                        head=10000,
                        sample_size=2000,
                        force_reload=False):
    os.makedirs(os.path.dirname(cache_file) if os.path.dirname(cache_file) else '.', exist_ok=True)

    if os.path.exists(cache_file) and not force_reload:
        print("Loading cached data from " + cache_file)
        return pickle.load(open(cache_file, 'rb'))

    genre_reviews_dict = {}

    for genre, url in GENRE_URL_DICT.items():
        print("Loading reviews for genre: " + genre)
        genre_reviews_dict[genre] = load_reviews(url, head=head, sample_size=sample_size)

    pickle.dump(genre_reviews_dict, open(cache_file, 'wb'))
    print("Cached data to " + cache_file)

    return genre_reviews_dict


def prepare_train_test_split(genre_reviews_dict, train_ratio=0.8):
    train_texts = []
    train_labels = []
    test_texts = []
    test_labels = []

    for genre, reviews in genre_reviews_dict.items():
        random.shuffle(reviews)
        split = int(len(reviews) * train_ratio)

        for review in reviews[:split]:
            train_texts.append(review)
            train_labels.append(genre)

        for review in reviews[split:]:
            test_texts.append(review)
            test_labels.append(genre)

    return train_texts, train_labels, test_texts, test_labels


class GoodreadsDataset(torch.utils.data.Dataset):
    """Custom Dataset for Goodreads reviews."""

    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def load_and_prepare_data(tokenizer,
                          max_length=512,
                          cache_file='data/genre_reviews_dict.pickle',
                          head=10000,
                          sample_size=2000,
                          train_ratio=0.8,
                          force_reload=False):
    # Load reviews
    genre_reviews_dict = load_goodreads_data(
        cache_file=cache_file,
        head=head,
        sample_size=sample_size,
        force_reload=force_reload
    )

    # Split 80/20
    train_texts, train_labels, test_texts, test_labels = prepare_train_test_split(
        genre_reviews_dict,
        train_ratio=train_ratio
    )

    # Print lengths like notebook
    print(
        "len(train_texts)=" + str(len(train_texts)) +
        ", len(train_labels)=" + str(len(train_labels)) +
        ", len(test_texts)=" + str(len(test_texts)) +
        ", len(test_labels)=" + str(len(test_labels))
    )

    # Encode labels
    label_encoder = LabelEncoder()
    train_labels_encoded = label_encoder.fit_transform(train_labels)
    test_labels_encoded = label_encoder.transform(test_labels)

    print("Number of classes: " + str(len(label_encoder.classes_)))
    print("Classes: " + str(label_encoder.classes_))

    # Tokenize
    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=max_length)
    test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=max_length)

    # Create datasets
    train_dataset = GoodreadsDataset(train_encodings, train_labels_encoded)
    test_dataset = GoodreadsDataset(test_encodings, test_labels_encoded)

    return train_dataset, test_dataset, label_encoder
