import itertools
import numpy as np
import tensorflow as tf

from typing import List, Dict, Set

from classifier.predictor.debug_info import FeaturesInfo


class Features:
    def load():
        with open('classifier/external_genres.txt') as f:
            external_genres = set([line.strip() for line in f])
        external_genres_index = {genre: i for i,
                                 genre in enumerate(sorted(external_genres))}
        external_genres_reverse_index = {
            v: k for k, v in external_genres_index.items()}

        with open('classifier/tags.txt') as f:
            tags = set([line.strip() for line in f])
        tags_index = {tag: i for i, tag in enumerate(sorted(tags))}
        tags_reverse_index = {v: k for k, v in tags_index.items()}

        return Features(external_genres, external_genres_index, external_genres_reverse_index, tags, tags_index, tags_reverse_index)

    def __init__(
            self,
            external_genres: Set[str],
            external_genres_index: Dict[str, int],
            external_genres_reverse_index: Dict[int, str],
            tags: Set[str],
            tags_index: Dict[str, int],
            tags_reverse_index: Dict[int, str]):
        self.external_genres = external_genres
        self.external_genres_index = external_genres_index
        self.external_genres_reverse_index = external_genres_reverse_index
        self.tags = tags
        self.tags_index = tags_index
        self.tags_reverse_index = tags_reverse_index

    def N(self) -> int:
        # Number of external genres plus tags for IGDB, Steam, GOG, Wikipedia
        # and from description.
        return len(self.external_genres) + len(self.tags) * 5

    def build_array(
            self,
            igdb_genres: List[str],
            steam_genres: List[str],
            gog_genres: List[str],
            wiki_genres: List[str],
            igdb_tags: List[str],
            steam_tags: List[str],
            gog_tags: List[str],
            wiki_tags: List[str],
            description: str):
        '''
        Returns a tensor of shape (1,N) encoding all instance tags.

        Args:
            Instance tags. Each string in the input tags is expected to be
            unique, so each tag type uses an identifiable prefix.

        Returns:
            Tensor(1, N): Each position represents a tag and the value is the
            relative position of the tag for the instance.
        '''
        igdb_genres = ['IGDB_' + v for v in igdb_genres]
        steam_genres = ['STEAM_' + v for v in steam_genres]
        gog_genres = ['GOG_' + v for v in gog_genres]
        wiki_genres = ['WIKI_' + normalize(v).replace(' game', '').replace(' video', '').replace(' simulations', '').replace(
            ' simulation', '').replace(' simulator', '').replace(' sim', '').replace(' and ', ' ').replace(' & ', ' ').replace(' n ', ' ') for v in wiki_genres]

        igdb_tags = [normalize(kw) for kw in igdb_tags]
        steam_tags = [normalize(kw) for kw in steam_tags]
        gog_tags = [normalize(kw) for kw in gog_tags]
        wiki_tags = [normalize(kw) for kw in wiki_tags]

        indices = []
        values = []

        # Fill external genres in the feature vector.
        for i, genre in itertools.chain(enumerate(igdb_genres), enumerate(steam_genres), enumerate(gog_genres), enumerate(wiki_genres)):
            if genre not in self.external_genres:
                # print(f'W: Found {genre} that is not in external_genres set.')
                continue

            indices.append(self.external_genres_index[genre])
            values.append(i + 1)

        # Fill external tags/keywords in the feature vector.
        for i, tags in enumerate([igdb_tags, steam_tags, gog_tags, wiki_tags]):
            start_index = len(self.external_genres) + len(self.tags) * i
            for (j, tag) in enumerate(tags):
                # take only tags/keywords that are in the supported tag list
                if tag in self.tags:
                    indices.append(start_index + self.tags_index[tag])
                    values.append(j + 1)

        # Fill tags/keywords extracted from the description in the feature vector.
        description = normalize(description)
        start_index = len(self.external_genres) + len(self.tags) * 4
        for tag in self.tags:
            if tag in description:
                indices.append(start_index + self.tags_index[tag])
                values.append(1)

        X = np.zeros(self.N(), dtype=int)
        X[indices] = values
        X = tf.expand_dims(X, axis=0)
        return X

    def debug(self, X) -> FeaturesInfo:
        '''
        Returns human readable description of a (N,) Tensor representing tags.

        Args:
            X (Tensor(N,)): Input tensor X with values for each input tag.

        Returns:
            FeaturesInfo: Human readable representation of input features.
        '''
        features = FeaturesInfo()

        # print(f'X.shape = {X.shape}')
        for i, v in enumerate(X[0]):
            # print(f'v.shape = {v.shape}')
            if v < 1:
                continue

            if i < len(self.external_genres):
                features.external_genres.append(
                    f'{self.external_genres_reverse_index[i]} = {v}')
            elif i < (len(self.external_genres) + len(self.tags)):
                i = i - len(self.external_genres)
                features.igdb_tags.append(
                    f'{self.tags_reverse_index[i]} = {v}')
            elif i < (len(self.external_genres) + 2*len(self.tags)):
                i = i - len(self.external_genres)
                i = i % len(self.tags)
                features.steam_tags.append(
                    f'{self.tags_reverse_index[i]} = {v}')
            elif i < (len(self.external_genres) + 3*len(self.tags)):
                i = i - len(self.external_genres)
                i = i % len(self.tags)
                features.gog_tags.append(
                    f'{self.tags_reverse_index[i]} = {v}')
            elif i < (len(self.external_genres) + 4*len(self.tags)):
                i = i - len(self.external_genres)
                i = i % len(self.tags)
                features.wiki_tags.append(
                    f'{self.tags_reverse_index[i]} = {v}')
            elif i < (len(self.external_genres) + 5*len(self.tags)):
                i = i - len(self.external_genres)
                i = i % len(self.tags)
                features.description_tags.append(
                    f'{self.tags_reverse_index[i]} = {v}')
            else:
                print(f'BAD INDEX={i} in input feature vector')

        return features


def normalize(s: str) -> str:
    return s.lower().replace("'", '').replace('-', ' ')
