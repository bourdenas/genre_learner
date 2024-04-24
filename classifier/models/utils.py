import os
from collections import defaultdict


def collect_images(directory, extensions=('.jpg', '.jpeg', '.png')):
    '''
    Iterates through a directory structure and collects paths of images.

    Args:
        directory (str): The root directory to search within.
        extensions (tuple, optional): A tuple of image file extensions to collect.
                                      Defaults to ('.jpg', '.jpeg', '.png').

    Returns:
        list: A list of image file paths found within the directory structure.
    '''

    image_paths = defaultdict(list)
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(extensions):
                image_paths[root].append(file)
    return image_paths
