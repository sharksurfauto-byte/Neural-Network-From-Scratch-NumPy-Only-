import tensorflow as tf
import numpy as np

def load_and_preprocess_mnist():
    '''Loads and preprocesses the MNIST dataset.'''
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # Reshape and normalize data
    X_train = x_train.reshape(x_train.shape[0], -1).T / 255.0
    X_test = x_test.reshape(x_test.shape[0], -1).T / 255.0

    return X_train, y_train, X_test, y_test
