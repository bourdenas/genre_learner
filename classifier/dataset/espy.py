import classifier.dataset.utils as utils
import numpy as np
import tensorflow as tf
import tensorflow.data as tf_data
import tensorflow.keras.layers as tfl

from classifier.dataset.genres import Genres
from classifier.dataset.tags import Tags
from typing import Dict


class EspyDataset:
    def __init__(self, examples, tags, texts, genres=[], word_index: Dict[str, int] = {}) -> None:
        self.examples = examples

        # (N,F) dimentional tensor where N is the number of examples and F is
        # the size of their input feature vector.
        self.tags = tags

        # (N,L) dimentional tensor where N is the number of examples and L is
        # the size of their output label vector.
        self.genres = genres

        self.texts = texts
        self.word_index = word_index

    def from_csv(filename):
        '''
        Loads a dataset from a csv file.

        Args:
            filename (str): Path to the csv file describing the dataset.
        '''
        examples = utils.load_examples(filename)

        tags = Tags.load()
        genres = Genres.load()

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
            igdb_keywords = example.igdb_keywords.split(
                '|') if example.igdb_keywords else []
            steam_tags = example.steam_tags.split(
                '|') if example.steam_tags else []
            gog_tags = example.gog_tags.split(
                '|') if example.gog_tags else []

            X.append(
                tags.build_array(
                    igdb_genres=['IGDB_' + v for v in igdb_genres],
                    igdb_keywords=['KW_IGDB_' + v for v in igdb_keywords],
                    steam_genres=['STEAM_' + v for v in steam_genres],
                    steam_tags=['KW_STEAM_' + v for v in steam_tags],
                    gog_genres=['GOG_' + v for v in gog_genres],
                    gog_tags=['KW_GOG_' + v for v in gog_tags],
                )
            )

            texts.append(
                example.description.replace('-', ' ').replace('\n', ' ').lower())

            # Y label array represents the espy genres, where the value of each
            # cell is either 0 or 1. Each example may be assigned a few genres.
            if example.espy_genres:
                espy_genres = example.espy_genres.split('|')
                Y.append(genres.build_array(espy_genres))

        texts_ds = tf_data.Dataset.from_tensor_slices(texts).batch(128)
        vectorizer = tfl.TextVectorization(
            max_tokens=20000, output_sequence_length=200)
        vectorizer.adapt(texts_ds)

        vocab = vectorizer.get_vocabulary()
        word_index = dict(zip(vocab, range(len(vocab))))

        return EspyDataset(
            examples,
            tags=tf.concat(X, axis=0),
            genres=tf.concat(Y, axis=0) if Y else [],
            texts=vectorizer(np.array([[s] for s in texts])).numpy(),
            word_index=word_index)
