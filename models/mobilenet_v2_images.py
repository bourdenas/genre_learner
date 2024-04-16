import os                                   # nopep8
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'    # nopep8

import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
import tensorflow.keras.layers as tfl


def create_model(image_size, classes, learning_rate):
    """
    Create a model based on MobileNetV2 that can classify `classes`.

    Args:
        image_size (tuple): A tuple with (width, height) of input images.
        classes (int): Number of classes used in the classification task.
        learning_rate (int): Learning rate for training the model.

    Returns:
        model: A Keras model for classification based on MobileNetV2
    """
    input_shape = image_size + (3,)
    base_model = tf.keras.applications.MobileNetV2(input_shape=input_shape,
                                                   include_top=False,
                                                   weights='imagenet')

    base_model.trainable = False
    inputs = tf.keras.Input(shape=input_shape)

    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
    x = base_model(x, training=False)

    # Add the final classification layer.
    x = tfl.GlobalAveragePooling2D()(x)
    x = tfl.Dropout(0.2)(x)
    outputs = tfl.Dense(classes, activation='softmax')(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def optimize_base_model(model, layer_to_start_tunning, learning_rate):
    """
    Create a model based on MobileNetV2 that can classify `classes`.

    Args:
        image_size (tuple): A tuple with (width, height) of input images.
        classes (int): Number of classes used in the classification task.
        learning_rate (int): Learning rate for training the model.

    Returns:
        model: A Keras model for classification based on MobileNetV2
    """
    base_model = model.layers[3]
    base_model.trainable = True
    print("Number of layers in the base model: ", len(base_model.layers))

    for layer in base_model.layers[:layer_to_start_tunning]:
        layer.trainable = False

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Optimize the mobilenet_v2 model to predict game genre from single screenshots.")
    parser.add_argument(
        "--train_dataset", help="Filepath with the dataset of images and classes to load for training.", default='dataset/')
    parser.add_argument(
        "--output_path", help="Filepath with the dataset of images and classes to load for training.", default='trained/')
    parser.add_argument(
        "--image_size", help="Image vertical size used as input for training. (default: 720)", type=int, default=720)
    parser.add_argument("--image_aspect_ratio",
                        help="Image aspect ratio. (default: 1.777)", type=float, default=1.7777777777)
    parser.add_argument("--batch_size",
                        help="Image training batch size. (default: 32)", type=float, default=32)

    args = parser.parse_args()

    image_size = (
        round(args.image_size * args.image_aspect_ratio), args.image_size)

    (train_dataset, validation_dataset) = image_dataset_from_directory(args.train_dataset,
                                                                       shuffle=True,
                                                                       batch_size=args.batch_size,
                                                                       image_size=image_size,
                                                                       validation_split=0.2,
                                                                       subset='both',
                                                                       seed=42)

    class_names = train_dataset.class_names

    train_dataset = train_dataset.prefetch(
        buffer_size=tf.data.experimental.AUTOTUNE)

    base_learning_rate = 0.001
    model = create_model(image_size, classes=len(
        class_names), learning_rate=base_learning_rate)
    model.summary()

    initial_epochs = 10
    history = model.fit(
        train_dataset, validation_data=validation_dataset, epochs=initial_epochs)

    # Optimize last layers of the original model.
    optimize_base_model(model, layer_to_start_tunning=120,
                        learning_rate=base_learning_rate * 0.1)

    fine_tune_epochs = 5
    total_epochs = initial_epochs + fine_tune_epochs
    history_fine = model.fit(train_dataset,
                             epochs=total_epochs,
                             initial_epoch=history.epoch[-1],
                             validation_data=validation_dataset)

    model.save(args.output_path)
