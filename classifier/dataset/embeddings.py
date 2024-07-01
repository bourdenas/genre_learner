import numpy as np

from typing import Dict


def embedding_matrix(embeddings_path: str, word_dict: Dict[str, int]):
    '''
    Returns an embeddings matrix for the input word dictionary.

    Args:
        embeddings_path: Filepath to the embeddings file.
        word_dict: Dictionary from word to its index in the vocabulary.
    '''
    embeddings_dim = None
    embeddings_index = {}
    with open(embeddings_path) as f:
        for line in f:
            word, coefs = line.split(maxsplit=1)
            coefs = np.fromstring(coefs, 'f', sep=' ')
            if not embeddings_dim:
                embeddings_dim = coefs.shape[0]
            embeddings_index[word] = coefs
    print(f'Found {len(embeddings_index)} word vectors.')

    vocab_size = len(word_dict) + 2
    embedding_matrix = np.zeros((vocab_size, embeddings_dim))

    hits, misses = 0, 0
    for word, i in word_dict.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
            hits += 1
        else:
            misses += 1
    print(f'Converted {hits} words ({misses} misses)')

    return embedding_matrix
