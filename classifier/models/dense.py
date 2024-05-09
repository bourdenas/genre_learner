import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as tfl


def build(features, classes):
    '''
    Builds a dense model.

    Args:
        features (int): Number of input features per examples.
        classes (int): Number of output classes for prediction.
    '''
    inputs = tf.keras.Input(shape=(features,))
    X = tfl.Dense(2048, activation='relu')(inputs)
    X = tfl.Dropout(0.2)(X)
    outputs = tfl.Dense(classes, activation='sigmoid')(X)

    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                  #   loss='mean_squared_error',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model
