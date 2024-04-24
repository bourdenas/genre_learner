import os
import argparse
import utils

from collections import defaultdict


def create_dir_structure(base_dir, dir_structure, image_size):
    """
    Creates a directory structure for each game to label and download images for
    each one.

    Args:
        base_dir (str): The base directory path.
        dir_structure (dict): A dictionary mapping directory to image urls.
    """
    for dir_name, images in dir_structure.items():
        full_path = os.path.join(base_dir, dir_name)
        os.makedirs(full_path, exist_ok=True)

        for image in images:
            utils.download_and_resize_image(image, full_path, image_size)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generates a dataset for training from a csv file with labeled examples.")
    parser.add_argument("--examples", help="CSV file with labeled examples.")
    parser.add_argument(
        "--dataset_path", help="Directory path to generate the dataset from the examples.")
    parser.add_argument(
        "--image_size", help="Image vertical size. (default: 720)", type=int, default=720)
    parser.add_argument("--image_aspect_ratio",
                        help="Image aspect ratio. (default: 1.777)", type=float, default=1.7777777777)

    args = parser.parse_args()

    examples = utils.load_examples(args.examples)

    dir_structure = defaultdict(list)
    for example in examples:
        dir_structure[utils.normalize_to_dir_name(example.name)].extend(
            example.images.split("|"))

    create_dir_structure(args.dataset_path, dir_structure,
                         (round(args.image_size * args.image_aspect_ratio), args.image_size))
