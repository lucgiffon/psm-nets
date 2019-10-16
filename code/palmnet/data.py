import keras
import numpy as np
from keras.datasets import mnist


def mnist_lenet():
    from palmnet.utils import root_dir

    model_path = root_dir / "models/external" / "mnist_lenet_1570207294.h5"
    model = keras.models.load_model(str(model_path), compile=False)
    return model

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

    @staticmethod
    def load_model(name="lenet"):
        if name == "lenet":
            return mnist_lenet()
        else:
            raise ValueError("Unknown model name {}".format(name))