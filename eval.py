import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'    # nopep8

import argparse
import tensorflow as tf

from classifier.dataset.espy import EspyDataset
from classifier.dataset.features import Features
from classifier.dataset.genres import Genres

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate predictions for genre classifications.")
    parser.add_argument(
        '--examples', help='Path to csv file with a model predictions.')
    parser.add_argument(
        '--model', help='Filepath to the model to use for eval.')
    parser.add_argument(
        '--sigmoid_threshold', help='Threshold of the sigmoid function (0,1) for accepting a label prediction. (default: 0.5)', type=float, default=.5)
    args = parser.parse_args()

    genres = Genres.load()
    features = Features.load()
    dataset = EspyDataset.from_csv(args.examples)

    model = tf.keras.models.load_model(args.model)
    model.summary()

    preds = model.predict(dataset.features, verbose=0)

    wins, mistakes, predictions, missing_prediction, total_ground_truths = 0, 0, 0, 0, 0
    errors = []
    for i, example in enumerate(dataset.examples):
        result = genres.prediction(preds[i], args.sigmoid_threshold)
        ground_truth = example.espy_genres

        if result == ground_truth:
            wins += 1
        else:
            mistakes += 1
            print(f'{example.name} ({example.id}) l:{example.espy_genres}  p:{result}')

        if result:
            predictions += 1
        else:
            missing_prediction += 1
        total_ground_truths += 1

    print()
    print(f'wins={wins}')
    print(f'mistakes={mistakes}')
    print(f'predictions={predictions}')
    print(f'missing prediction={missing_prediction}')
    print(f'total={total_ground_truths}')
    print(f'precision={(wins / predictions):.4}')
    print(f'coverage={(predictions / total_ground_truths):.4}')
