import itertools
import numpy as np
import tensorflow as tf

from typing import List, Dict, Set


class Features:
    def load():
        with open('classifier/external_genres.txt') as f:
            external_genres = set([line.strip() for line in f])
        external_genres_index = {genre: i for i,
                                 genre in enumerate(sorted(external_genres))}
        external_genres_reverse_index = {
            v: k for k, v in external_genres_index.items()}

        # tag_sources = ['IGDB', 'STEAM', 'GOG', 'TEXT']
        with open('classifier/tags.txt') as f:
            tags = set([line.strip() for line in f])
            # tags = set()
            # for source in tag_sources:
            #     tags.update(set([f'KW_{source}_{line.strip()}' for line in f]))
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
        return len(self.external_genres) + len(self.tags) * 4

    def build_array(
            self,
            igdb_genres: List[str],
            steam_genres: List[str],
            gog_genres: List[str],
            wiki_genres: List[str],
            igdb_keywords: List[str],
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
        wiki_genres = ['WIKI_' + v.lower().replace('-', ' ').replace(' game', '').replace(' video', '').replace(' simulations', '').replace(
            ' simulation', '').replace(' simulator', '').replace(' sim', '').replace(' and ', ' ').replace(' & ', ' ').replace(' n ', ' ') for v in wiki_genres]

        igdb_keywords = [kw.lower().replace("'", '').replace('-', ' ')
                         for kw in igdb_keywords]
        steam_tags = [kw.lower().replace("'", '').replace('-', ' ')
                      for kw in steam_tags]
        gog_tags = [kw.lower().replace("'", '').replace('-', ' ')
                    for kw in gog_tags]
        wiki_tags = [kw.lower().replace("'", '').replace('-', ' ')
                     for kw in wiki_tags]

        indices = []
        values = []

        for i, genre in itertools.chain(enumerate(igdb_genres), enumerate(steam_genres), enumerate(gog_genres), enumerate(wiki_genres)):
            if genre not in self.external_genres:
                # print(f'W: Found {genre} that is not in external_genres set.')
                continue

            indices.append(self.external_genres_index[genre])
            values.append(i + 1)

        for i, tags in enumerate([igdb_keywords, steam_tags, gog_tags, wiki_tags]):
            start_index = len(self.external_genres) + len(self.tags) * i
            discarding = []
            for (j, tag) in enumerate(tags):
                if tag in self.tags:
                    indices.append(start_index + self.tags_index[tag])
                    values.append(j + 1)
                else:
                    discarding.append(tag)
            # if len(discarding) > 0:
            #     print(f'Discarding: {discarding}')

        description = description.lower().replace("'", '').replace('-', ' ')
        start_index = len(self.external_genres) + len(self.tags) * 3
        for tag in self.tags:
            if tag in description:
                indices.append(start_index + self.tags_index[tag])
                values.append(1)
                # tt = {
                #     'IGDB': [tag for tag in igdb_keywords if tag in self.tags],
                #     'STEAM': [tag for tag in steam_tags if tag in self.tags],
                #     'GOG': [tag for tag in gog_tags if tag in self.tags],
                # }

                # print(f'GENRES = {[igdb_genres, steam_genres, gog_genres]}')
                # print(
                #     f'TAGS = {tt}')
                # print(f'INDICES = {indices}')
                # print(f'VALUES = {values}\n')

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
        features = {}

        # print(f'X.shape = {X.shape}')
        for i, p in enumerate(X[0]):
            # print(f'p.shape = {p.shape}')
            if not p > 0:
                continue

            if i < len(self.external_genres):
                features[self.external_genres_reverse_index[i]] = f'{p}'
            elif i < (len(self.external_genres) + len(self.tags)):
                i = i - len(self.external_genres)
                features['IGDB_KW_' + self.tags_reverse_index[i]] = f'{p}'
            elif i < (len(self.external_genres) + 2*len(self.tags)):
                i = i - len(self.external_genres)
                i = i % len(self.tags)
                features['STEAM_KW_' + self.tags_reverse_index[i]] = f'{p}'
            else:
                i = i - len(self.external_genres)
                i = i % len(self.tags)
                features['GOG_KW_' + self.tags_reverse_index[i]] = f'{p}'

        return features
