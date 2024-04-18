import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'    # nopep8

import argparse
import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as tfl

from tensorflow.keras.preprocessing import image, image_dataset_from_directory
from tensorflow.keras.initializers import random_uniform, glorot_uniform, constant, identity
from utils import collect_images


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

    # Retrieve Filters
    F1, F2, F3 = filters

    # Save the input value. You'll need this later to add back to the main path.
    X_shortcut = X

    # First component of main path
    X = tfl.Conv2D(filters=F1, kernel_size=1, strides=(1, 1),
                   padding='valid', kernel_initializer=initializer(seed=0))(X)
    X = tfl.BatchNormalization(axis=3)(X)  # Default axis
    X = tfl.Activation('relu')(X)

    # Second component of main path (≈3 lines)
    # Set the padding = 'same'
    X = tfl.Conv2D(filters=F2, kernel_size=f, strides=(1, 1),
                   padding='same', kernel_initializer=initializer(seed=0))(X)
    X = tfl.BatchNormalization(axis=3)(X)  # Default axis
    X = tfl.Activation('relu')(X)

    # Third component of main path (≈2 lines)
    # Set the padding = 'valid'
    X = tfl.Conv2D(filters=F3, kernel_size=1, strides=(1, 1),
                   padding='valid', kernel_initializer=initializer(seed=0))(X)
    X = tfl.BatchNormalization(axis=3)(X)  # Default axis

    # Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)
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

    # First component of main path glorot_uniform(seed=0)
    X = tfl.Conv2D(filters=F1, kernel_size=1, strides=(s, s),
                   padding='valid', kernel_initializer=initializer(seed=0))(X)
    X = tfl.BatchNormalization(axis=3)(X)
    X = tfl.Activation('relu')(X)

    # START CODE HERE

    # Second component of main path (≈3 lines)
    X = tfl.Conv2D(filters=F2, kernel_size=f, strides=(1, 1),
                   padding='same', kernel_initializer=initializer(seed=0))(X)
    X = tfl.BatchNormalization(axis=3)(X)
    X = tfl.Activation('relu')(X)

    # Third component of main path (≈2 lines)
    X = tfl.Conv2D(filters=F3, kernel_size=1, strides=(1, 1),
                   padding='valid', kernel_initializer=initializer(seed=0))(X)
    X = tfl.BatchNormalization(axis=3)(X)

    # SHORTCUT PATH ##### (≈2 lines)
    X_shortcut = tfl.Conv2D(filters=F3, kernel_size=1, strides=(
        s, s), padding='valid', kernel_initializer=initializer(seed=0))(X_shortcut)
    X_shortcut = tfl.BatchNormalization(axis=3)(X_shortcut)

    # END CODE HERE

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

    # Zero-Padding
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

    # START CODE HERE

    # Use the instructions above in order to implement all of the Stages below
    # Make sure you don't miss adding any required parameter

    # Stage 3 (≈4 lines)
    # `convolutional_block` with correct values of `f`, `filters` and `s` for this stage
    X = convolutional_block(X, f=3, filters=[128, 128, 512], s=2)

    # the 3 `identity_block` with correct values of `f` and `filters` for this stage
    X = identity_block(X, 3, [128, 128, 512])
    X = identity_block(X, 3, [128, 128, 512])
    X = identity_block(X, 3, [128, 128, 512])

    # Stage 4 (≈6 lines)
    # add `convolutional_block` with correct values of `f`, `filters` and `s` for this stage
    X = convolutional_block(X, f=3, filters=[256, 256, 1024], s=2)

    # the 5 `identity_block` with correct values of `f` and `filters` for this stage
    X = identity_block(X, 3, [256, 256, 1024])
    X = identity_block(X, 3, [256, 256, 1024])
    X = identity_block(X, 3, [256, 256, 1024])
    X = identity_block(X, 3, [256, 256, 1024])
    X = identity_block(X, 3, [256, 256, 1024])

    # Stage 5 (≈3 lines)
    # add `convolutional_block` with correct values of `f`, `filters` and `s` for this stage
    X = convolutional_block(X, f=3, filters=[512, 512, 2048], s=2)

    # the 2 `identity_block` with correct values of `f` and `filters` for this stage
    X = identity_block(X, 3, [512, 512, 2048])
    X = identity_block(X, 3, [512, 512, 2048])

    # AVGPOOL (≈1 line). Use "X = AveragePooling2D()(X)"
    X = tfl.AveragePooling2D(pool_size=(2, 2))(X)

    # END CODE HERE

    # output layer
    X = tfl.Flatten()(X)
    X = tfl.Dense(classes, activation='softmax',
                  kernel_initializer=glorot_uniform(seed=0))(X)

    # Create model
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
        '--image_aspect_ratio',
        help='Image aspect ratio. (default: 1.777)', type=float, default=1.7777777777)
    parser.add_argument(
        '--batch_size',
        help='Image training batch size. (default: 32)', type=float, default=32)

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

    # genres = collect_images(args.dataset)

    # X_train_orig = []
    # Y_train_orig = []
    # class_names = []
    # classes = len(genres)
    # for (i, (genre, images)) in enumerate(genres.items()):
    #     class_names.append(genre)
    #     Y = tf.zeros((classes,))
    #     indices = [[i]]
    #     updates = [1.0]
    #     Y = tf.scatter_nd(indices, updates, Y.shape)

    #     # processed_images = []
    #     for filename in images:
    #         img = image.load_img(os.path.join(genre, filename),
    #                              target_size=(1280, 720))
    #         X = image.img_to_array(img)
    #         try:
    #             X = tf.expand_dims(X, axis=0)
    #         except Exception as e:
    #             print(f'{genre} -- {filename} -- {Y} -- {X.shape}')
    #         X_train_orig.append(X)

    #         # Y = tf.expand_dims(Y, axis=0)
    #         # Y_train_orig.append(Y)

    # X_train_orig = tf.concat(X_train_orig, axis=0)
    # print(X_train_orig.shape)

    # Y_train_orig = tf.concat(Y_train_orig, axis=0)
    # print(Y_train_orig.shape)

    model.fit(train_dataset, validation_data=validation_dataset, epochs=10)
    model.save(args.output)

    # base_learning_rate = 0.001
    # model = create_model(image_size, classes=len(
    #     class_names), learning_rate=base_learning_rate)
    # model.summary()

    # initial_epochs = 10
    # history = model.fit(
    #     train_dataset, validation_data=validation_dataset, epochs=initial_epochs)

    # # Optimize last layers of the original model.
    # optimize_base_model(model, layer_to_start_tunning=120,
    #                     learning_rate=base_learning_rate * 0.1)

    # fine_tune_epochs = 5
    # total_epochs = initial_epochs + fine_tune_epochs
    # history_fine = model.fit(train_dataset,
    #                          epochs=total_epochs,
    #                          initial_epoch=history.epoch[-1],
    #                          validation_data=validation_dataset)

    # model.save(args.output, save_format='keras')
