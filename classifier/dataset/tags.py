import itertools
import numpy as np
import tensorflow as tf

from typing import List, Dict, Set


class Tags:
    def load():
        with open('classifier/tags.txt') as f:
            tags = set([line.strip() for line in f])

        tags_index = {genre: i for i, genre in enumerate(sorted(tags))}
        reverse_index = {v: k for k, v in tags_index.items()}

        return Tags(tags, tags_index, reverse_index)

    def __init__(
            self,
            tags: Set[str],
            tags_index: Dict[str, int],
            reverse_index: Dict[int, str]):
        self.tags = tags

        # Mapping of tags to their index in the input vector and reverse.
        self.tags_index = tags_index
        self.reverse_index = reverse_index

    def N(self) -> int:
        return len(self.tags)

    def build_array(
            self,
            igdb_genres: List[str],
            steam_genres: List[str],
            gog_genres: List[str],
            igdb_keywords: List[str],
            steam_tags: List[str],
            gog_tags: List[str]):
        '''
        Returns a tensor of shape (1,N) encoding all instance tags.

        Args:
            Instance tags. Each string in the input tags is expected to be
            unique, so each tag type uses an identifiable prefix.

        Returns:
            Tensor(1, N): Each position represents a tag and the value is the
            relative position of the tag for the instance.
        '''
        indices = []
        for tag in itertools.chain(igdb_genres, steam_genres, gog_genres, igdb_keywords, steam_tags, gog_tags):
            if tag in self.tags:
                indices.append(self.tags_index[tag])

        values = []
        for (i, tag) in itertools.chain(enumerate(igdb_genres), enumerate(steam_genres), enumerate(gog_genres), enumerate(igdb_keywords), enumerate(steam_tags), enumerate(gog_tags)):
            if tag in self.tags:
                values.append(i + 1)

        X = np.zeros(self.N(), dtype=int)
        X[indices] = values
        X = tf.expand_dims(X, axis=0)
        return X

    def decode_array(self, X) -> Dict[str, str]:
        '''
        Returns human readable description of a (N,) Tensor representing tags.

        Args:
            X (Tensor(N,)): Input tensor X with values for each input tag.

        Returns:
            Dict[str,str]: Instance tags with their non-zero values.
        '''
        return {self.reverse_index[i]: f'{p}' for i, p in enumerate(X) if p > 0}
