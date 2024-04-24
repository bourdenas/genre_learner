import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'    # nopep8

import csv
import argparse
import models.dense as dense
import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as tfl

from dataset.espy import EspyDataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Predict genre classes for an espy dataset.")
    parser.add_argument("--examples", help="CSV file with labeled examples.")
    parser.add_argument(
        '--model', help='Filepath to an already trained model to resume training. If omitted trains a model from scratch.')
    parser.add_argument(
        '--output', help='Filepath to save the trained model.')
    parser.add_argument(
        '--epochs', help='Number of epochs to train. (default: 10)', type=int, default=40)
    parser.add_argument(
        '--predictions', help='Filepath to save a csv with the model predictions.')

    args = parser.parse_args()

    dataset = EspyDataset()
    dataset.load(args.examples)

    if args.model == None:
        model = dense.build(
            features=len(dataset.feature_names()), classes=len(dataset.class_names()))
        model.summary()

        history = model.fit(x=dataset.X, y=dataset.Y,
                            validation_split=0.2, epochs=args.epochs)
        model.save(args.output)
    else:
        model = tf.keras.models.load_model(args.model)

    predictions = model.predict(dataset.X, verbose=0)

    with open(args.predictions, "w") as csvfile:
        writer = csv.DictWriter(
            csvfile, fieldnames=['name', 'prediction', 'genres', 'features'])
        writer.writeheader()
        for i, example in enumerate(dataset.examples):
            writer.writerow({
                'name': example.name,
                'prediction': dataset.decodeY(predictions[i]),
                'genres': dataset.decodeY(dataset.Y[i]),
                'features': dataset.decodeX(dataset.X[i]),
            })
