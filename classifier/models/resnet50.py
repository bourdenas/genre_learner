import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'    # nopep8

import argparse
import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as tfl

from tensorflow.keras.preprocessing import image, image_dataset_from_directory
from tensorflow.keras.initializers import random_uniform, glorot_uniform, constant, identity


def identity_block(X, f, filters, initializer=random_uniform):
    """
    Implementation of the identity block as defined in Figure 4

    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    f -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    initializer -- to set up the initial weights of a layer. Equals to random uniform initializer

    Returns:
    X -- output of the identity block, tensor of shape (m, n_H, n_W, n_C)
    """

    F1, F2, F3 = filters

    X_shortcut = X

    # First component
    X = tfl.Conv2D(filters=F1, kernel_size=1, strides=(1, 1),
                   padding='valid', kernel_initializer=initializer(seed=0))(X)
    X = tfl.BatchNormalization(axis=3)(X)  # Default axis
    X = tfl.Activation('relu')(X)

    # Second component
    X = tfl.Conv2D(filters=F2, kernel_size=f, strides=(1, 1),
                   padding='same', kernel_initializer=initializer(seed=0))(X)
    X = tfl.BatchNormalization(axis=3)(X)  # Default axis
    X = tfl.Activation('relu')(X)

    # Third component
    X = tfl.Conv2D(filters=F3, kernel_size=1, strides=(1, 1),
                   padding='valid', kernel_initializer=initializer(seed=0))(X)
    X = tfl.BatchNormalization(axis=3)(X)  # Default axis

    # Add shortcut
    X = tfl.Add()([X, X_shortcut])
    X = tfl.Activation('relu')(X)

    return X


def convolutional_block(X, f, filters, s=2, initializer=glorot_uniform):
    """
    Implementation of the convolutional block as defined in Figure 4

    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    f -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    s -- Integer, specifying the stride to be used
    initializer -- to set up the initial weights of a layer. Equals to Glorot uniform initializer,
                   also called Xavier uniform initializer.

    Returns:
    X -- output of the convolutional block, tensor of shape (m, n_H, n_W, n_C)
    """

    # Retrieve Filters
    F1, F2, F3 = filters

    # Save the input value
    X_shortcut = X

    ##### MAIN PATH #####

    # First component of main path
    X = tfl.Conv2D(filters=F1, kernel_size=1, strides=(s, s),
                   padding='valid', kernel_initializer=initializer(seed=0))(X)
    X = tfl.BatchNormalization(axis=3)(X)
    X = tfl.Activation('relu')(X)

    # Second component
    X = tfl.Conv2D(filters=F2, kernel_size=f, strides=(1, 1),
                   padding='same', kernel_initializer=initializer(seed=0))(X)
    X = tfl.BatchNormalization(axis=3)(X)
    X = tfl.Activation('relu')(X)

    # Third component
    X = tfl.Conv2D(filters=F3, kernel_size=1, strides=(1, 1),
                   padding='valid', kernel_initializer=initializer(seed=0))(X)
    X = tfl.BatchNormalization(axis=3)(X)

    # Shortcut
    X_shortcut = tfl.Conv2D(filters=F3, kernel_size=1, strides=(
        s, s), padding='valid', kernel_initializer=initializer(seed=0))(X_shortcut)
    X_shortcut = tfl.BatchNormalization(axis=3)(X_shortcut)

    # Final step: Add shortcut value to main path (Use this order [X, X_shortcut]), and pass it through a RELU activation
    X = tfl.Add()([X, X_shortcut])
    X = tfl.Activation('relu')(X)

    return X


def ResNet50(input_shape=(64, 64, 3), classes=6, training=False):
    """
    Stage-wise implementation of the architecture of the popular ResNet50:
    CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> CONVBLOCK -> IDBLOCK*2 -> CONVBLOCK -> IDBLOCK*3
    -> CONVBLOCK -> IDBLOCK*5 -> CONVBLOCK -> IDBLOCK*2 -> AVGPOOL -> FLATTEN -> DENSE

    Arguments:
    input_shape -- shape of the images of the dataset
    classes -- integer, number of classes

    Returns:
    model -- a Model() instance in Keras
    """

    # Define the input as a tensor with shape input_shape
    X_input = tfl.Input(input_shape)

    X = tfl.ZeroPadding2D((3, 3))(X_input)

    # Stage 1
    X = tfl.Conv2D(64, (7, 7), strides=(2, 2),
                   kernel_initializer=glorot_uniform(seed=0))(X)
    X = tfl.BatchNormalization(axis=3)(X)
    X = tfl.Activation('relu')(X)
    X = tfl.MaxPooling2D((3, 3), strides=(2, 2))(X)

    # Stage 2
    X = convolutional_block(X, f=3, filters=[64, 64, 256], s=1)
    X = identity_block(X, 3, [64, 64, 256])
    X = identity_block(X, 3, [64, 64, 256])

    # Stage 3
    X = convolutional_block(X, f=3, filters=[128, 128, 512], s=2)

    X = identity_block(X, 3, [128, 128, 512])
    X = identity_block(X, 3, [128, 128, 512])
    X = identity_block(X, 3, [128, 128, 512])

    # Stage 4
    X = convolutional_block(X, f=3, filters=[256, 256, 1024], s=2)

    X = identity_block(X, 3, [256, 256, 1024])
    X = identity_block(X, 3, [256, 256, 1024])
    X = identity_block(X, 3, [256, 256, 1024])
    X = identity_block(X, 3, [256, 256, 1024])
    X = identity_block(X, 3, [256, 256, 1024])

    # Stage 5
    X = convolutional_block(X, f=3, filters=[512, 512, 2048], s=2)

    X = identity_block(X, 3, [512, 512, 2048])
    X = identity_block(X, 3, [512, 512, 2048])

    X = tfl.AveragePooling2D(pool_size=(2, 2))(X)

    # output layer
    X = tfl.Flatten()(X)
    X = tfl.Dense(classes, activation='softmax',
                  kernel_initializer=glorot_uniform(seed=0))(X)

    model = tf.keras.models.Model(inputs=X_input, outputs=X)

    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Optimize the mobilenet_v2 model to predict game genre from single screenshots.')
    parser.add_argument(
        '--dataset', help='Filepath with the dataset of images and classes to load for training.')
    parser.add_argument(
        '--retrain_model', help='Filepath to an already trained model to resume training. If omitted trains a model from scratch.')
    parser.add_argument(
        '--output', help='Filepath to save the trained model.')
    parser.add_argument(
        '--image_size', help='Image vertical size used as input for training. (default: 720)', type=int, default=720)
    parser.add_argument(
        '--image_aspect_ratio', help='Image aspect ratio. (default: 1.777)', type=float, default=1.7777777777)
    parser.add_argument(
        '--batch_size', help='Image training batch size. (default: 32)', type=int, default=32)
    parser.add_argument(
        '--epochs', help='Number of epochs to train. (default: 10)', type=int, default=10)

    args = parser.parse_args()

    image_size = (
        round(args.image_size * args.image_aspect_ratio), args.image_size)

    (train_dataset, validation_dataset) = image_dataset_from_directory(args.dataset,
                                                                       shuffle=True,
                                                                       batch_size=args.batch_size,
                                                                       image_size=image_size,
                                                                       validation_split=0.2,
                                                                       subset='both',
                                                                       seed=42)
    class_names = train_dataset.class_names

    train_dataset = train_dataset.prefetch(
        buffer_size=tf.data.experimental.AUTOTUNE)

    model = None
    if args.retrain_model == None:
        model = ResNet50(input_shape=image_size + (3,),
                         classes=len(class_names))
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.00015),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])
        model.summary()
    else:
        model = tf.keras.models.load_model(args.retrain_model)

    model.fit(
        train_dataset, validation_data=validation_dataset, epochs=args.epochs)
    model.save(args.output, save_format='keras')
