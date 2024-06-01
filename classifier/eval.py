import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'    # nopep8

import argparse
import dataset.utils as utils
import tensorflow as tf

from dataset.espy import EspyDataset, Features, Labels

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate predictions for genre classifications.")
    parser.add_argument(
        '--examples', help='Path to csv file with a model predictions.')
    parser.add_argument(
        '--model', help='Filepath to the model to use for eval.')
    args = parser.parse_args()

    features = Features.load()
    labels = Labels.load()
    dataset = EspyDataset.from_csv(args.examples)

    model = tf.keras.models.load_model(args.model)
    model.summary()

    predictions = model.predict(dataset.X, verbose=0)

    wins, total_predictions, total_ground_truths = 0, 0, 0
    for i, example in enumerate(dataset.examples):
        result = set(labels.labels(predictions[i], threshold=.5))
        ground_truth = set(example.espy_genres.split('|'))

        for label in result:
            if label in ground_truth:
                wins += 1
        total_predictions += len(result)
        total_ground_truths += len(ground_truth)

    print(f'precision={(wins / total_predictions):.2}')
    print(f'recall={(wins / total_ground_truths):.2}')
