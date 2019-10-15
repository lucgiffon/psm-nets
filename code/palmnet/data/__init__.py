from keras.datasets import mnist
import numpy as np
import keras

class Mnist:
    num_classes = 10

    @staticmethod
    def load_data():
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train = np.expand_dims(x_train, axis=-1)
        x_test = np.expand_dims(x_test, axis=-1)
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train /= 255
        x_test /= 255
        y_train = keras.utils.to_categorical(y_train, Mnist.num_classes)
        y_test = keras.utils.to_categorical(y_test, Mnist.num_classes)
        return (x_train, y_train), (x_test, y_test)