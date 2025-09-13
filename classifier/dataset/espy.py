import re
import matplotlib.pyplot as plt
import classifier.dataset.utils as utils
import tensorflow as tf

from classifier.dataset.genres import Genres
from classifier.dataset.features import Features
from typing import Set


class EspyDataset:
    def __init__(
        self,
        examples: list,  # specify the type of objects in the list
        texts: list[str],
        features: tf.Tensor,  # shape: (N, F), N = num examples, F = feature vector size
        genres: tf.Tensor = [],  # shape: (N, G), N = num examples, G = num genres (labels)
    ) -> None:
        self.examples = examples

        # List of text descriptions for each example.
        self.texts = texts

        # (N,F) dimensional tensor where N is the number of examples and F is
        # the size of their input feature vector.
        self.features = features

        # (N,G) dimensional tensor where N is the number of examples and G is
        # the number of genre labels.
        self.genres = genres


    def from_csv(filename: str):
        """
        Loads a dataset from a csv file.

        Args:
            filename (str): Path to the csv file describing the dataset.
        """
        examples = utils.load_examples(filename)

        features = Features.load()
        genres = Genres.load()

        keywords = set()
        for tag in features.tags:
            keywords.add(tag.split('_')[-1].lower())

        X, Y, texts = [], [], []
        for example in examples:
            # X input array dimensions are IGDB + Steam + GOG tags where the
            # value for each feature is the genres/tags position in the listing
            # to encode its importance.
            igdb_genres = example.igdb_genres.split(
                '|') if example.igdb_genres else []
            steam_genres = example.steam_genres.split(
                '|') if example.steam_genres else []
            gog_genres = example.gog_genres.split(
                '|') if example.gog_genres else []
            wiki_genres = example.wiki_genres.split(
                '|') if example.wiki_genres else []
            igdb_tags = example.igdb_tags.split(
                '|') if example.igdb_tags else []
            steam_tags = example.steam_tags.split(
                '|') if example.steam_tags else []
            gog_tags = example.gog_tags.split(
                '|') if example.gog_tags else []
            wiki_tags = example.wiki_tags.split(
                '|') if example.wiki_tags else []

            X.append(
                features.build_array(
                    igdb_genres=igdb_genres,
                    steam_genres=steam_genres,
                    gog_genres=gog_genres,
                    wiki_genres=wiki_genres,
                    igdb_tags=igdb_tags,
                    steam_tags=steam_tags,
                    gog_tags=gog_tags,
                    wiki_tags=wiki_tags,
                    description=example.description,
                )
            )

            text = preprocess_text(example.description, keywords)
            texts.append(text)

            # Y label array represents the espy genres, where the value of each
            # cell is either 0 or 1. Each example may be assigned a few genres.
            if example.espy_genres:
                espy_genres = example.espy_genres.split('|')
                Y.append(genres.build_array(espy_genres))

        # plot([len(x) for x in texts])

        return EspyDataset(
            examples,
            features=tf.concat(X, axis=0),
            genres=tf.concat(Y, axis=0) if Y else [],
            texts=texts)


def preprocess_text(text: str, keywords: Set[str]) -> str:
    """
    Returns preprocessed text for training and prediction.
    """

    # Convert to lowercase
    text = str(text).lower()
    
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', ' ', text)
    
    # Remove special characters but keep spaces and basic punctuation
    text = re.sub(r'[^\w\s.,!?-]', ' ', text)

    # Replace hyphens with whitespace.
    text = re.sub(r'-', ' ', text)

    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    # Split text into stentences.
    sentences = re.split('\.|\?|\!', text)

    # Keep only sentences that contain `keywords`. This is meant to reduce fluff
    # for text processing.
    texts = []
    for sentence in sentences:
        words = sentence.split()
        unigrams = set(words)
        bigrams = set([' '.join(words[i:i+2]) for i in range(len(words)-1)])
        trigrams = set([' '.join(words[i:i+3]) for i in range(len(words)-2)])
        ngrams = unigrams | bigrams | trigrams

        intersection = ngrams & keywords
        if intersection and (len(texts) == 0 or texts[-1] != sentence):
            texts.append(sentence)

    return '. '.join(texts)


def plot(data):
    # Create the histogram
    plt.hist(data, bins=100, cumulative=True)

    # Customize the plot (optional)
    plt.xlabel("Values")
    plt.ylabel("Frequency")
    plt.title("Histogram of Sample Data")
    plt.grid(True)

    # Show the plot
    plt.show()
