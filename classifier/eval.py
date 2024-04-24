import argparse
import dataset.utils as utils

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate predictions for genre classifications.")
    parser.add_argument(
        '--predictions', help='Path to csv file with a model predictions.')
    args = parser.parse_args()

    examples = utils.load_examples(args.predictions)

    wins, partials, losses = 0, 0, 0
    for example in examples:
        predictions = set([p.split(':')[0]
                          for p in example.prediction.split(',')])
        genres = set([g.split(':')[0] for g in example.genres.split(',')])

        diff = genres.difference(predictions)
        if len(diff) == 0:
            wins += 1
        elif predictions.issubset(genres):
            partials += 1
        else:
            losses += 1

    print(f'wins={wins}, partials={partials}, losses={losses}')
    print(f'accuracy={(wins / len(examples)):.2}, partial_accuracy={((wins + partials) / len(examples)):.2}, loss={(losses / len(examples)):.2}')
