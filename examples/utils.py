import csv
import os
import re

from PIL import Image
from urllib.request import urlretrieve


class CsvRowDict:
    def __init__(self, **kwargs):
        # Set attributes based on keyword arguments
        self.__dict__.update(kwargs)


def load_examples(filename):
    """
    Reads a CSV file and returns entries for label predictions.

    Args:
        filename (str): The path to the CSV file.

    Returns:
        list: A list CsvRowDict instances representing each row in the CSV file.
    """
    examples = []
    with open(filename, 'r') as csvfile:
        reader = csv.DictReader(csvfile)

        # Get header row (field names)
        headers = reader.fieldnames

        for row in reader:
            # Create class instance with row data as attributes
            instance = CsvRowDict(**row)
            examples.append(instance)

    return examples


def normalize_to_dir_name(text):
    """
    Normalizes a string to be a valid directory name.

    Args:
        text (str): The string to be normalized.

    Returns:
        str: The normalized string suitable for a directory name.
    """
    # Replace | with _
    text = text.replace("|", "_")

    # Remove all characters except alphanumeric, underscore, hyphen, space and exclamation mark.
    pattern = r"[^\w\-_\s]"
    normalized = re.sub(pattern, "", text.strip())

    # Replace spaces with underscores
    normalized = normalized.replace(" ", "-")

    # Convert to lowercase
    normalized = normalized.lower()

    return normalized


def download_and_resize_image(image_url, output_dir, image_size):
    """
    Downloads an image from a URL, resizes it, and saves it to a directory.

    Args:
        image_url (str): The URL of the image to download.
        output_dir (str): The directory path to save the resized image.
        new_width (int): The desired width of the resized image.
        new_height (int): The desired height of the resized image.
    """
    # Get filename from URL
    filename = os.path.basename(image_url).split("?")[0]
    # Create full output path
    output_path = os.path.join(output_dir, filename)

    try:
        # Download the image
        image_url = image_url.replace("https://", "http://")
        urlretrieve(image_url, output_path)

        # Open the downloaded image
        image = Image.open(output_path)

        # Resize the image
        resized_image = image.resize(image_size)

        # Save the resized image
        resized_image.save(output_path)

        print(f"Image downloaded and resized: {filename}")
    except Exception as e:
        print(f"Error downloading/resizing image: {e}")
