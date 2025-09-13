import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as tfl

from tensorflow.keras.models import Model


def build(features: int, classes: int, vocab_size: int = 2000, embedding_dim: int = 64,
          embedding_matrix: np.ndarray = None, max_sequence_length: int = 500,
          lstm_units: int = 32, dropout_rate: float = 0.3,
          trainable_embeddings: bool = False):
    '''
    Builds a hybrid model with pretrained GloVe embeddings.

    Args:
        features (int): Number of input features per example (from dense network).
        classes (int): Number of output classes for prediction.
        vocab_size (int): Size of vocabulary for text processing. Ignored if embedding_matrix is provided.
        embedding_dim (int): Dimension of word embeddings. Ignored if embedding_matrix is provided.
        embedding_matrix (np.ndarray): Pretrained embedding matrix (vocab_size, embedding_dim).
        max_sequence_length (int): Maximum sequence length for text input.
        lstm_units (int): Number of LSTM units.
        dropout_rate (float): Dropout rate for regularization.
        trainable_embeddings (bool): Whether to fine-tune embeddings during training.
    '''

    weights = None
    if embedding_matrix is not None:
        print('adjusting vocab and dims')
        vocab_size, embedding_dim = embedding_matrix.shape
        weights = [embedding_matrix]

    # Dense features input
    dense_input = tf.keras.Input(shape=(features,), name='dense_features')

    # Text input
    text_input = tf.keras.Input(shape=(max_sequence_length,), name='text_sequences')

    # Dense network branch
    dense_x = tfl.Dense(2048, activation='relu', name='dense_1')(dense_input)
    dense_x = tfl.Dropout(dropout_rate, name='dense_dropout_1')(dense_x)
    dense_x = tfl.Dense(1024, activation='relu', name='dense_2')(dense_x)
    dense_x = tfl.Dropout(dropout_rate, name='dense_dropout_2')(dense_x)
    dense_x = tfl.Dense(512, activation='relu', name='dense_3')(dense_x)
    dense_x = tfl.Dropout(dropout_rate, name='dense_dropout_3')(dense_x)


    # LSTM network branch with GloVe embeddings
    text_x = tfl.Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        weights=weights,
        input_length=max_sequence_length,
        trainable=trainable_embeddings,
        name='text_embedding'
    )(text_input)

    # Bidirectional LSTM layers
    text_x = tfl.Bidirectional(
        tfl.LSTM(lstm_units, dropout=dropout_rate, recurrent_dropout=dropout_rate, 
                return_sequences=True),
        name='bi_lstm_1'
    )(text_x)

    text_x = tfl.Bidirectional(
        tfl.LSTM(lstm_units // 2, dropout=dropout_rate, recurrent_dropout=dropout_rate),
        name='bi_lstm_2'
    )(text_x)

    # Dense layers for text features
    text_x = tfl.Dense(256, activation='relu', name='text_dense_1')(text_x)
    text_x = tfl.Dropout(dropout_rate, name='text_dropout_1')(text_x)
    text_x = tfl.Dense(128, activation='relu', name='text_dense_2')(text_x)
    text_x = tfl.Dropout(dropout_rate, name='text_dropout_2')(text_x)

    # Concatenate dense and text features
    combined = tfl.Concatenate(name='feature_fusion')([dense_x, text_x])

    # Final classification layers
    combined = tfl.Dense(512, activation='relu', name='fusion_dense_1')(combined)
    combined = tfl.Dropout(dropout_rate, name='fusion_dropout_1')(combined)
    combined = tfl.Dense(256, activation='relu', name='fusion_dense_2')(combined)
    combined = tfl.Dropout(dropout_rate, name='fusion_dropout_2')(combined)

    # Output layer
    outputs = tfl.Dense(classes, activation='softmax', name='predictions')(combined)

    model = Model(inputs=[dense_input, text_input], outputs=outputs, name='hybrid_classifier')

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']  #'top_k_categorical_accuracy'
    )

    return model
