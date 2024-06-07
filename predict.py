import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'    # nopep8

import csv
import argparse
import tensorflow as tf

from classifier.dataset.espy import EspyDataset, Features, Labels


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Predict genre classes for an espy dataset.')
    parser.add_argument(
        '--model', help='Filepath to the model to use for prediction.')
    parser.add_argument(
        '--input', help='Filepath to CSV file with unlabeled dataset.')
    parser.add_argument(
        '--output', help='Filepath to save a CSV with the model predictions.')

    args = parser.parse_args()

    features = Features.load()
    labels = Labels.load()
    dataset = EspyDataset.from_csv(args.input)

    model = tf.keras.models.load_model(args.model)
    model.summary()

    predictions = model.predict(dataset.X, verbose=0)
    rows = []
    for i, example in enumerate(dataset.examples):
        rows.append({
            'id': example.id,
            'name': example.name,
            'prediction': ','.join(labels.labels(predictions[i])),
        })
        # print(f'{example.name} -- {labels.labels(predictions[i])}')

    print('writing to file')
    with open(args.output, 'w') as csvfile:
        writer = csv.DictWriter(
            csvfile, fieldnames=['id', 'name', 'prediction', 'genres', 'features'])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
