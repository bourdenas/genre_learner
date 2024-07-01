import itertools
import numpy as np
import tensorflow as tf
import classifier.dataset.utils as utils


from classifier.dataset.genres import Genres
from classifier.dataset.tags import Tags


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

        tags = Tags.load()
        genres = Genres.load()

        X, Y = [], []
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

            # Y label array represents the espy genres, where the value of each
            # cell is either 0 or 1. Each example may be assigned a few genres.
            if example.espy_genres:
                espy_genres = example.espy_genres.split('|')
                Y.append(genres.build_array(espy_genres))

        return EspyDataset(examples, tf.concat(X, axis=0), tf.concat(Y, axis=0) if Y else [])
