import itertools
import numpy as np
import tensorflow as tf
import classifier.dataset.utils as utils

from typing import List, Dict, Set


class Features:
    def load():
        with open('classifier/features.txt') as f:
            features = set([line.strip() for line in f])

        feature_index = {genre: i for i, genre in enumerate(sorted(features))}
        reverse_index = {v: k for k, v in feature_index.items()}

        return Features(features, feature_index, reverse_index)

    def __init__(self,
                 features: Set[str],
                 feature_index: Dict[str, int],
                 reverse_index: Dict[int, str]):
        self.features = features

        # Mapping of feature names to index in the input vector and reverse.
        self.feature_index = feature_index
        self.reverse_index = reverse_index

    def N(self) -> int:
        return len(self.features)

    def build_array(
            self,
            igdb_genres: List[str],
            steam_genres: List[str],
            igdb_keywords: List[str],
            steam_tags: List[str]
    ):
        '''
        Returns an feature input vector of shape (1,N) encoding the input
        `igdb_genres` and `steam_tags`.

        Args:
            igdb_genres: List of IGDB genres provided as input. The ordering of
                         the genres is encoded in the output vector.
            steam_tags: List of STEAM tags provided as input. The ordering of
                        the tags is encoded in the output vector.

        Returns:
            Tensor(1, N): List of features in the array with a non-zero value.
        '''
        indices = []
        for feature in itertools.chain(igdb_genres, steam_genres, igdb_keywords, steam_tags):
            if feature in self.features:
                indices.append(self.feature_index[feature])

        values = []
        for (i, feature) in itertools.chain(enumerate(igdb_genres), enumerate(steam_genres), enumerate(igdb_keywords), enumerate(steam_tags)):
            if feature in self.features:
                values.append(i + 1)

        X = np.zeros(self.N(), dtype=int)
        X[indices] = values
        X = tf.expand_dims(X, axis=0)
        return X

    def decode_array(self, X) -> List[str]:
        '''
        Returns human readable description of the input vector `X`.

        Args:
            X (Tensor(N,)): Input tensor X with values for each input feature.

        Returns:
            list(str): List of features in the array with a non-zero value.
        '''
        return [f'{self.reverse_index[i]}: {p}' for i, p in enumerate(X) if p > 0]


class Labels:
    def load():
        with open('classifier/espy_genres.txt') as f:
            espy_genres = set([line.strip() for line in f])

        label_index = {genre: i for i, genre in enumerate(sorted(espy_genres))}
        reverse_index = {v: k for k, v in label_index.items()}
        return Labels(espy_genres, label_index, reverse_index)

    def __init__(self,
                 espy_genres: Set[str],
                 label_index: Dict[str, int],
                 reverse_index: Dict[int, str]):
        self.espy_genres = espy_genres

        # Mapping of label names to index in the output vector and reverse.
        self.label_index = label_index
        self.reverse_index = reverse_index

    def N(self) -> int:
        return len(self.label_index)

    def build_array(self, espy_genres: List[str]):
        indices = [self.label_index[genre] for genre in espy_genres]
        values = [1 for _ in espy_genres]

        Y = np.zeros(self.N())
        Y[indices] = values
        Y = tf.expand_dims(Y, axis=0)
        return Y

    def decode_array(self, Y) -> List[str]:
        '''
        Returns human readable description of the predictions output array Y.

        Args:
            Y (tensor (L,)): Output tensor Y with values for each label predicition.

        Returns:
            list(str): List of labels in the array with a non-zero value.
        '''
        return [f'{self.reverse_index[i]}: {p:.2}' for i, p in enumerate(Y) if p > 0]

    def labels(self, Y, threshold=0.5) -> List[str]:
        return [self.reverse_index[i] for i, p in enumerate(Y) if p >= threshold]


class EspyDataset:
    def __init__(self, examples, X, Y=[]) -> None:
        self.examples = examples

        # (N,F) dimentional tensor where N is the number of examples and F is
        # the size of their input feature vector.
        self.X = X
        # (N,C) dimentional tensor where N is the number of examples and C is
        # the size of their output classification vector.
        self.Y = Y

    def from_csv(filename):
        '''
        Loads a dataset from a csv file.

        Args:
            filename (str): Path to the csv file describing the dataset.
        '''
        examples = utils.load_examples(filename)

        features = Features.load()
        labels = Labels.load()

        X, Y = [], []
        for example in examples:
            # X input array dimensions are IGDB genres + Steam tags where the
            # value for each feature is the genres/tags position in the listing
            # to encode its importance.
            igdb_genres = example.igdb_genres.split(
                '|') if example.igdb_genres else []
            steam_genres = example.steam_genres.split(
                '|') if example.steam_genres else []
            igdb_keywords = example.igdb_keywords.split(
                '|') if example.igdb_keywords else []
            steam_tags = example.steam_tags.split(
                '|') if example.steam_tags else []

            X.append(
                features.build_array(
                    igdb_genres=['IGDB_' + v for v in igdb_genres],
                    igdb_keywords=['KW_IGDB_' + v for v in igdb_keywords],
                    steam_genres=['STEAM_' + v for v in steam_genres],
                    steam_tags=['KW_STEAM_' + v for v in steam_tags]
                )
            )

            # Y labels array represents the espy genres, where the value of each
            # cell is either 0 or 1. Each example may be assigned a few genres.
            if example.espy_genres:
                espy_genres = example.espy_genres.split('|')
                Y.append(labels.build_array(espy_genres))

        return EspyDataset(examples, tf.concat(X, axis=0), tf.concat(Y, axis=0) if Y else [])
