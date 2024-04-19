import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'    # nopep8

import time
import argparse
import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as tfl

from collections import defaultdict
from tensorflow.keras.preprocessing import image, image_dataset_from_directory
from utils import collect_images                                   # nopep8


def get_top_n(data, N):
    '''
    Get indices and values of the top N elements in a list.

    Args:
        data (list): A list of sortable elements.
        N (int): An integer specifying the number of top elements to return.

    Returns:
        A tuple containing two lists:
            - top_indices: Indices of the top N elements in the original order.
            - top_values: Values of the top N elements.
    '''
    if not N or N < 0:
        return [], []

    sorted_data = sorted(enumerate(data), key=lambda x: x[1], reverse=True)
    return sorted_data[:N]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Annotate games with genres based on an input model using screenshot annotations.')
    parser.add_argument(
        '--dataset', help='Filepath with the dataset of images and classes to load for predictions.')
    parser.add_argument(
        '--model', help='Filepath to a model for annotating screenshots.')
    parser.add_argument(
        '--image_size', help='Image vertical size used as input for training. (default: 720)', type=int, default=720)
    parser.add_argument(
        '--image_aspect_ratio', help='Image aspect ratio. (default: 1.777)', type=float, default=1.7777777777)
    args = parser.parse_args()

    # TODO: That's a hack to load the dataset class names. Find a better way to retreive class labels.
    (train_dataset, validation_dataset) = image_dataset_from_directory('dataset_single_genre/',
                                                                       shuffle=True,
                                                                       batch_size=32,
                                                                       image_size=(
                                                                           1280, 720),
                                                                       validation_split=0.2,
                                                                       subset='both',
                                                                       seed=42)
    class_names = train_dataset.class_names

    image_size = (
        round(args.image_size * args.image_aspect_ratio), args.image_size)

    model = tf.keras.models.load_model(args.model)

    games = collect_images(args.dataset)
    for (game, images) in games.items():
        processed_images = []
        for img in images:
            img = image.load_img(os.path.join(game, img),
                                 target_size=image_size)
            img_array = image.img_to_array(img)
            img_array = tf.expand_dims(img_array, axis=0)
            processed_images.append(
                tf.keras.applications.mobilenet_v2.preprocess_input(img_array))
            # processed_images.append(img_array)

        img_processed = tf.concat(processed_images, axis=0)
        print(img_processed.shape)
        prediction = model.predict(img_processed, verbose=0)
        prediction = prediction.tolist()
        prediction = [get_top_n(pred, 3) for pred in prediction]

        classification = defaultdict(int)
        # `prediction` are the top N predictions for each image in the batch.
        for pred in prediction:
            (index, p) = pred[0]
            classification[class_names[index]] += 1
        genres = sorted(classification.items(),
                        key=lambda x: x[1], reverse=True)
        print(f'{game} -> {genres[0]}')
