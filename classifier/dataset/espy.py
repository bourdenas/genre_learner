import numpy as np
import tensorflow as tf
import dataset.utils as utils


class EspyDataset:
    def __init__(self) -> None:
        # Mapping of a feature name to its index in the input vector and
        # reverse.
        self.feature_index = {}
        self.index_to_feature = {}

        # Mapping of a predicted genre name to its index in the output vector
        # and reverse.
        self.genre_index = {}
        self.index_to_genre = {}

        # (N,F) dimentional tensor where N is the number of examples and F is
        # the size of their input feature vector.
        self.X = []
        # (N,C) dimentional tensor where N is the number of examples and C is
        # the size of their output classification vector.
        self.Y = []

    def decodeX(self, X):
        '''
        Returns human readable description of the input vector X.

        Args:
            X (tensor (F,)): Input tensor X with values for each input feature.

        Returns:
            str: Human readable description of input features.
        '''
        return ', '.join([f'{self.index_to_feature[i]}: {p}' for i, p in enumerate(X) if p > 0])

    def decodeY(self, Y):
        '''
        Returns human readable description of the predictions output vector Y.

        Args:
            Y (tensor (C,)): Output tensor Y with values for each class predicition.

        Returns:
            str: Human readable description of the prediction output.
        '''
        return ', '.join([f'{self.index_to_genre[i]}: {p:.2}' for i, p in enumerate(Y) if p >= 0.1])

    def feature_names(self):
        '''
        Returns names of input features.

        Returns:
            list: List with str of size F with the names of input features.
        '''
        return self.feature_index.keys()

    def class_names(self):
        '''
        Returns names of output classes.

        Returns:
            list: List with str of size C with the names of output classes.
        '''
        return self.genre_index.keys()

    def load(self, filename):
        '''
        Loads a dataset from a csv file.

        Args:
            filename (str): Path to the csv file describing the dataset.
        '''
        examples = utils.load_examples(filename)

        igdb_genres, steam_tags, espy_genres = set(), set(), set()
        for example in examples:
            igdb_genres.update(example.igdb_genres.split('|'))
            steam_tags.update(example.steam_tags.split('|'))
            espy_genres.update(example.genres.split('|'))

        i = 0
        for genre in sorted(igdb_genres):
            self.feature_index[f'igdb_{genre}'] = i
            i += 1
        for tag in sorted(steam_tags):
            self.feature_index[f'steam_{tag}'] = i
            i += 1
        self.index_to_feature = {v: k for k, v in self.feature_index.items()}

        self.genre_index = {genre: i for i,
                            genre in enumerate(sorted(espy_genres))}
        self.index_to_genre = {v: k for k, v in self.genre_index.items()}

        self.__build_xy(examples)
        self.examples = examples

    def __build_xy(self, examples):
        for example in examples:
            # X input array dimensions are IGDB genres + Steam tags where the
            # value for each feature is the genres/tags position in the listing
            # to encode its importance.
            igdb_genres = example.igdb_genres.split('|')
            steam_tags = example.steam_tags.split('|')
            indices = [self.feature_index[f'igdb_{genre}'] for genre in igdb_genres] + \
                [self.feature_index[f'steam_{tag}'] for tag in steam_tags]
            values = [i + 1 for (i, _) in enumerate(igdb_genres)] + \
                [i + 1 for (i, _) in enumerate(steam_tags)]

            X = np.zeros(len(self.feature_index), dtype=int)
            X[indices] = values
            X = tf.expand_dims(X, axis=0)
            self.X.append(X)

            # Y labels array represents the espy genres, where the value of each
            # cell is either 0 or 1. Each example may be assigned a few genres.
            espy_genres = example.genres.split('|')
            indices = [self.genre_index[genre] for genre in espy_genres]
            values = [1 for _ in espy_genres]

            Y = np.zeros(len(self.genre_index))
            Y[indices] = values
            Y = tf.expand_dims(Y, axis=0)
            self.Y.append(Y)

        self.X = tf.concat(self.X, axis=0)
        self.Y = tf.concat(self.Y, axis=0)
