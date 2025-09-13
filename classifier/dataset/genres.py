import numpy as np
import tensorflow as tf

from typing import List, Dict, Set

from classifier.predictor.debug_info import PredictionInfo


class Genres:
    def load():
        with open('classifier/espy_genres.txt') as f:
            espy_genres = set([line.strip() for line in f])

        label_index = {genre: i for i, genre in enumerate(sorted(espy_genres))}
        reverse_index = {v: k for k, v in label_index.items()}
        return Genres(espy_genres, label_index, reverse_index)


    def __init__(
            self,
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


    def debug(self, Y) -> PredictionInfo:
        '''
        Returns human readable description of the predictions output array Y.

        Args:
            Y (tensor (L,)): Output tensor Y with values for each label predicition.

        Returns:
            Dict(str, str): Activated labels with their associated value in [0.001, 1].
        '''
        return PredictionInfo(genres={self.reverse_index[i]: f'{p:.3}' for i, p in enumerate(Y) if p >= 0.001})


    def labels(self, Y, threshold=0.5) -> List[str]:
        return [self.reverse_index[i] for i, p in enumerate(Y) if p >= threshold]


    def prediction(self, Y, threshold=0.1) -> str:
        """
        Returns label prediction based on output array Y.

        Args:
            Y (tensor (L,)): Output tensor Y with values for each label predicition.
            threshold (float, optional): The minimum score required to assign a label. Defaults to 0.1.

        Returns:
            str: The predicted label if its score meets or exceeds the threshold; otherwise, an empty string.
        """
        label = self.reverse_index[max(range(len(Y)), key=Y.__getitem__)]
        return label if Y[self.label_index[label]] >= threshold else ''
