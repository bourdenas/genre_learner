import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'    # nopep8

import argparse
import classifier.models.hybrid as hybrid
import numpy as np

from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

from classifier.dataset.espy import EspyDataset
from classifier.dataset.genres import Genres
from classifier.dataset.features import Features


def prepare_text_data_for_hybrid(dataset, vocab_size=2000, max_sequence_length=1000, glove_path=None):
    """
    Prepare text data for the hybrid model, optionally using GloVe embeddings.
    
    Args:
        dataset: EspyDataset instance
        vocab_size: Maximum vocabulary size
        max_sequence_length: Maximum sequence length
        glove_path: Path to GloVe embeddings (optional)
        
    Returns:
        tuple: (text_sequences, embedding_matrix, tokenizer)
    """

    # Create and fit tokenizer
    tokenizer = Tokenizer(num_words=vocab_size, oov_token='<OOV>')
    tokenizer.fit_on_texts(dataset.texts)

    # Convert texts to sequences
    sequences = tokenizer.texts_to_sequences(dataset.texts)
    text_sequences = pad_sequences(sequences, maxlen=max_sequence_length, padding='post', truncating='post')

    embedding_matrix = None
    if glove_path and os.path.exists(glove_path):
        print(f'Loading GloVe embeddings from {glove_path}...')
        embeddings_dict, embedding_dim = load_glove_embeddings(glove_path)
        embedding_matrix = create_embedding_matrix(tokenizer, embeddings_dict, embedding_dim, vocab_size)
        print(f'Created embedding matrix with shape: {embedding_matrix.shape}')
    
    return text_sequences, embedding_matrix, tokenizer


def load_glove_embeddings(glove_path: str) -> tuple[dict[str, np.ndarray], int]:
    """
    Load GloVe embeddings from a file.
    Args:
        glove_path (str): Path to the GloVe embeddings file.
    Returns:
        tuple: (embeddings_dict, embedding_dim)
    """
    embeddings_dict: dict[str, np.ndarray] = {}
    embedding_dim: int | None = None
    with open(glove_path, 'r', encoding='utf8') as f:
        for line in f:
            values = line.strip().split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            embeddings_dict[word] = vector
            if embedding_dim is None:
                embedding_dim = len(vector)
    return embeddings_dict, embedding_dim  # type: ignore


def create_embedding_matrix(
    tokenizer: Tokenizer,
    embeddings_dict: dict[str, np.ndarray],
    embedding_dim: int,
    vocab_size: int
) -> np.ndarray:
    """
    Create an embedding matrix for the words in the tokenizer based on the GloVe embeddings.
    Args:
        tokenizer (Tokenizer): Fitted Keras Tokenizer.
        embeddings_dict (dict): GloVe word vectors.
        embedding_dim (int): Dimension of the embeddings.
        vocab_size (int): Maximum vocabulary size.
    Returns:
        np.ndarray: Embedding matrix.
    """
    word_index = tokenizer.word_index
    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    for word, i in word_index.items():
        if i >= vocab_size:
            continue
        embedding_vector = embeddings_dict.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return embedding_matrix


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train a hybrid genre prediction model using dense features and LSTM text processing.')
    parser.add_argument(
        '--examples', help='Filepath to CSV file with labeled examples.')
    parser.add_argument(
        '--model', help='Name to save the trained model.')
    parser.add_argument(
        '--epochs', help='Number of epochs to train. (default: 40)', type=int, default=40)
    parser.add_argument(
        '--glove', help='Path to GloVe embeddings file (optional).', default=None)
    parser.add_argument(
        '--vocab-size', help='Vocabulary size for text processing (default: 2000)', type=int, default=2000)
    parser.add_argument(
        '--max-sequence-length', help='Maximum sequence length (default: 100)', type=int, default=100)
    parser.add_argument(
        '--lstm-units', help='Number of LSTM units (default: 32)', type=int, default=32)
    parser.add_argument(
        '--embedding-dim', help='Embedding dimension if not using GloVe (default: 64)', type=int, default=64)
    parser.add_argument(
        '--trainable-embeddings', help='Fine-tune GloVe embeddings during training', action='store_true')

    args = parser.parse_args()

    print('Loading dataset and features...')
    features = Features.load()
    genres = Genres.load()
    dataset = EspyDataset.from_csv(args.examples)

    print(f'Dataset loaded with {len(dataset.examples)} examples')
    print(f'Dense features shape: {dataset.features.shape}')
    
    # Prepare text data
    print('Preparing text data...')
    text_sequences, embedding_matrix, tokenizer = prepare_text_data_for_hybrid(
        dataset,
        vocab_size=args.vocab_size,
        max_sequence_length=args.max_sequence_length,
        glove_path=args.glove
    )
    
    print(f'Text sequences shape: {text_sequences.shape}')

    # Save tokenizer for later use
    import pickle
    with open(f'{args.model}_tokenizer.pkl', 'wb') as f:
        pickle.dump(tokenizer, f)
    print(f'Tokenizer saved for {args.model}')

    # Build model
    if embedding_matrix is not None:
        print('Building hybrid model with GloVe embeddings...')
        model = hybrid.build(
            features=features.N(),
            classes=genres.N(),
            embedding_matrix=embedding_matrix,
            max_sequence_length=args.max_sequence_length,
            lstm_units=args.lstm_units,
            trainable_embeddings=args.trainable_embeddings
        )
    else:
        print('Building hybrid model with random embeddings...')
        model = hybrid.build(
            features=features.N(),
            classes=genres.N(),
            vocab_size=args.vocab_size,
            max_sequence_length=args.max_sequence_length,
            embedding_dim=args.embedding_dim,
            lstm_units=args.lstm_units,
            trainable_embeddings=True
        )

    model.summary()

    # Prepare inputs for training
    train_inputs = {
        'dense_features': dataset.features,
        'text_sequences': text_sequences
    }

    # Calculate class weights for imbalanced dataset
    # class_weights = compute_class_weight(
    #     'balanced',
    #     classes=np.unique(dataset.genres),
    #     y=dataset.genres
    # )
    # class_weight_dict = dict(enumerate(class_weights))

    # Define callbacks
    callbacks = [
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-7
        ),
        keras.callbacks.ModelCheckpoint(
            f'{args.model}.keras',
            monitor='val_accuracy',
            save_best_only=True,
            mode='max'
        )
    ]

    print('Training hybrid model...')
    history = model.fit(
        x=train_inputs,
        y=dataset.genres,
        validation_split=0.2,
        epochs=args.epochs,
        batch_size=32,
        # class_weight=class_weight_dict,
        callbacks=callbacks,
        verbose=1
    )

    print(f'Saving model to {args.model}')
    model.save(f'{args.model}.keras')

    # Print training summary
    final_loss = history.history['loss'][-1]
    final_val_loss = history.history['val_loss'][-1]
    final_acc = history.history['accuracy'][-1]
    final_val_acc = history.history['val_accuracy'][-1]

    print(f'\nTraining completed!')
    print(f'Final training loss: {final_loss:.4f}')
    print(f'Final validation loss: {final_val_loss:.4f}')
    print(f'Final training accuracy: {final_acc:.4f}')
    print(f'Final validation accuracy: {final_val_acc:.4f}')
