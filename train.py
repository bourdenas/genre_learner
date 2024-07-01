import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'    # nopep8

import argparse
import classifier.models.dense as dense

from classifier.dataset.espy import EspyDataset
from classifier.dataset.genres import Genres
from classifier.dataset.tags import Tags


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train an espy genres prediction model using a labeled dataset.')
    parser.add_argument(
        '--examples', help='Filepath to CSV file with labeled examples.')
    parser.add_argument(
        '--output', help='Filepath to save the trained model.')
    parser.add_argument(
        '--epochs', help='Number of epochs to train. (default: 40)', type=int, default=40)

    args = parser.parse_args()

    tags = Tags.load()
    genres = Genres.load()
    dataset = EspyDataset.from_csv(args.examples)

    model = dense.build(features=tags.N(), classes=genres.N())
    model.summary()

    history = model.fit(x=dataset.X, y=dataset.Y,
                        validation_split=0.2, epochs=args.epochs)
    model.save(args.output)
