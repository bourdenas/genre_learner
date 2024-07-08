import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'    # nopep8

import argparse
import classifier.models.lstm as lstm

from classifier.dataset.espy import EspyDataset
from classifier.dataset.genres import Genres


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train an LSTM espy genres prediction model using a description text dataset.')
    parser.add_argument(
        '--examples', help='Filepath to CSV file with labeled examples.')
    parser.add_argument(
        '--embeddings', help='Filepath to GloVe pre-trained embeddings.')
    parser.add_argument(
        '--output', help='Filepath to save the trained model.')
    parser.add_argument(
        '--epochs', help='Number of epochs to train. (default: 10)', type=int, default=10)

    args = parser.parse_args()

    genres = Genres.load()
    dataset = EspyDataset.from_csv(args.examples)

    model = lstm.build(args.embeddings, dataset.word_index, genres.N())
    model.summary()

    history = model.fit(x=dataset.texts, y=dataset.genres,
                        validation_split=0.2, batch_size=128, epochs=args.epochs)
    model.save(args.output)
