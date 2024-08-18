import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as tfl

from typing import Dict


def build(embeddings_path: str, word_dict: Dict[str, int], genres: int):
    '''
    Builds an LSTM model to predict genres from text description. Expected
    model's input is text vectorized with the input `word_dict`.

    Args:
        embeddings_path: Filepath to the embeddings file.
        word_dict: Dictionary from word to its index in the vocabulary.
        genres: Number of total genres used in prediction.
    '''
    embedding_matrix = build_embedding_matrix(embeddings_path, word_dict)
    vocab_size, embedding_dim = embedding_matrix.shape
    embedding_layer = tfl.Embedding(
        vocab_size,
        embedding_dim,
        trainable=False,
    )
    embedding_layer.build((None,))
    embedding_layer.set_weights([embedding_matrix])

    inputs = tf.keras.Input(shape=(None,), dtype='int32')
    embeddings = embedding_layer(inputs)

    X = tfl.LSTM(units=1024, return_sequences=True)(embeddings)
    X = tfl.Dropout(0.2)(X)
    X = tfl.LSTM(units=512, return_sequences=False)(X)
    X = tfl.Dropout(0.2)(X)
    outputs = tfl.Dense(units=genres, activation='sigmoid')(X)

    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                  #   loss='mean_squared_error',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    return model


def build_embedding_matrix(embeddings_path: str, word_dict: Dict[str, int]):
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
