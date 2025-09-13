import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'    # nopep8

import argparse
import numpy as np
import pickle
import tensorflow as tf

from classifier.dataset.espy import EspyDataset
from classifier.dataset.features import Features
from classifier.dataset.genres import Genres
from tensorflow.keras.preprocessing.sequence import pad_sequences


def predict_hybrid_genre(
    model: tf.keras.Model,
    tokenizer,
    genres_obj: Genres,
    dense_features: np.ndarray,
    text_descriptions: list[str],
    max_sequence_length: int = 100,
    threshold: float = 0.5
) -> list[list[tuple[str, float]]]:
    """
    Make genre predictions for a batch using the hybrid model.

    Args:
        model: Trained hybrid model
        tokenizer: Fitted tokenizer for text processing
        genres_obj: Genres object
        dense_features_list: List/array of dense feature vectors (shape: [batch_size, feature_dim])
        text_descriptions: List of game description texts (length: batch_size)
        max_sequence_length: Maximum sequence length for text
        threshold: Threshold for binary classification

    Returns:
        list of lists: Each inner list contains predicted genres with probabilities for that example
    """

    # Preprocess text
    text_sequences = tokenizer.texts_to_sequences([desc.lower() if desc else "" for desc in text_descriptions])
    text_sequences = pad_sequences(text_sequences, maxlen=max_sequence_length, padding='post', truncating='post')

    # Prepare input
    inputs = {
        'dense_features': np.array(dense_features),
        'text_sequences': text_sequences
    }

    # Make prediction
    predictions = model.predict(inputs)

    # Convert predictions to genre names for each example
    batch_predicted_genres = []
    for pred in predictions:
        predicted_genres = []
        for i, prob in enumerate(pred):
            if prob >= threshold:
                genre_name = genres_obj.reverse_index[i]
                predicted_genres.append((genre_name, float(prob)))
        predicted_genres.sort(key=lambda x: x[1], reverse=True)
        batch_predicted_genres.append(predicted_genres)

    return batch_predicted_genres



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Eval the hybrid genre prediction model.')
    parser.add_argument(
        '--model', help='Path to trained hybrid model file.')
    parser.add_argument(
        '--tokenizer', help='Path to tokenizer pickle file.')
    parser.add_argument(
        '--examples', help='Path to csv file with a examples to use for eval.')
    parser.add_argument(
        '--threshold', help='Prediction threshold.', type=float, default=.5)

    args = parser.parse_args()
    
    if not args.model:
        print("Please provide --model name")
        print("Usage: python hybrid_eval.py --model hybrid_model")
        exit(1)

    # Single prediction mode
    print("Loading model...")
    model = tf.keras.models.load_model(f'{args.model}.keras')
    
    with open(f'{args.model}_tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
    
    features = Features.load()
    genres = Genres.load()
    dataset = EspyDataset.from_csv(args.examples)

    # Make prediction
    batch_predictions = predict_hybrid_genre(
        model=model,
        tokenizer=tokenizer,
        genres_obj=genres,
        dense_features=dataset.features,
        text_descriptions=dataset.texts,
        threshold=args.threshold  # Lower threshold to see more predictions
    )

    wins, mistakes, preds, missing_prediction, total_ground_truths = 0, 0, 0, 0, 0

    for i, example in enumerate(dataset.examples):
        print(f"\n📝 Example {i}:")
        print(f"{example.name} --> {' '.join(dataset.texts[i].split(' ')[:100])}")
        total_ground_truths += 1

        # Build dense features (simplified version)
        # In practice, you'd use the full feature building logic from the dataset
        dense_features = np.zeros(features.N())  # Placeholder - replace with actual feature building

        print(f"🎯 Predicted genres:")
        label = example.espy_genres
        predictions = batch_predictions[i]
        if predictions:
            for genre, prob in predictions[:5]:  # Show top 5
                result = "✅" if genre == label else "❌"
                print(f"  {genre}: {prob:.3f} {result} ({label})")

            genre, prob == predictions[0]
            if genre == label:
                wins += 1
            else:
                mistakes += 1
            preds += 1

        else:
            missing_prediction += 1
            print("  No genres above threshold")

    print()
    print(f'wins={wins}')
    print(f'mistakes={mistakes}')
    print(f'predictions={preds}')
    print(f'missing prediction={missing_prediction}')
    print(f'total={total_ground_truths}')
    print(f'precision={(wins / preds):.4}')
    print(f'coverage={(preds / total_ground_truths):.4}')
