import keras
import numpy as np
from keras import Sequential
from keras.datasets import mnist, cifar10, cifar100
from keras.initializers import he_normal
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Activation

MAP_EXTERNAL_MODEL_FILENAME = {
    "mnist_lenet": "mnist_lenet_1570207294.h5",
    "cifar10_vgg19_4096x4096": "cifar10_vgg19_4096x4096_1570693209.h5",
    "cifar100_vgg19_4096x4096": "cifar100_vgg19_4096x4096_1570789868.h5",
    "svhn_vgg19_4096x4096": "svhn_vgg19_4096x4096_1570786657.h5",
    "cifar10_vgg19_2048x2048": "cifar10_vgg19_2048x2048_1572303047.h5",
    "cifar100_vgg19_2048x2048": "cifar100_vgg19_2048x2048_1572278802.h5",
    "svhn_vgg19_2048x2048": "svhn_vgg19_2048x2048_1572278831.h5",
}

def get_external_model(name):
    from palmnet.utils import root_dir

    model_path = root_dir / "models/external" / MAP_EXTERNAL_MODEL_FILENAME[name]
    model = keras.models.load_model(str(model_path), compile=False)
    return model

def get_svhn():
    from palmnet.utils import root_dir

    data_dir = root_dir / "data/external" / "svhn.npz"
    loaded_npz = np.load(data_dir)

    return (loaded_npz["x_train"], loaded_npz["y_train"]), (loaded_npz["x_test"], loaded_npz["y_test"])


def random_small_model(input_shape, num_classes):
    model = Sequential()
    model.add(Conv2D(1, tuple([v-7 for v in input_shape[:2]]), padding='valid', activation='relu', kernel_initializer=he_normal(), input_shape=input_shape))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(Flatten(name='flatten'))
    model.add(Dense(1, use_bias=True, kernel_initializer=he_normal(), name='fc1'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dense(num_classes, kernel_initializer=he_normal(), name='predictions'))
    model.add(BatchNormalization())
    model.add(Activation('softmax'))
    return model

class Test:
    num_classes = 10
    input_shape = (28, 28, 1)

    @staticmethod
    def load_data():
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train = np.expand_dims(x_train, axis=-1)
        x_test = np.expand_dims(x_test, axis=-1)
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train /= 255
        x_test /= 255
        y_train = keras.utils.to_categorical(y_train, Test.num_classes)
        y_test = keras.utils.to_categorical(y_test, Test.num_classes)
        return (x_train, y_train), (x_test, y_test)

    @staticmethod
    def load_model(name="random"):
        if name == "random":
            return random_small_model(Test.input_shape, Test.num_classes)
        else:
            raise ValueError("Unknown model name {}".format(name))

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
    def load_model(name="mnist_lenet"):
        return get_external_model(name)


class Cifar10:
    num_classes = 10

    @staticmethod
    def load_data():
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        mean_x_train = np.mean(x_train, axis=(0, 1, 2))
        std_x_train = np.std(x_train, axis=(0, 1, 2))
        x_train -= mean_x_train
        x_train /= std_x_train
        mean_x_test = np.mean(x_test, axis=(0, 1, 2))
        std_x_test = np.std(x_test, axis=(0, 1, 2))
        x_test -= mean_x_test
        x_test /= std_x_test

        y_train = keras.utils.to_categorical(y_train, Cifar10.num_classes)
        y_test = keras.utils.to_categorical(y_test, Cifar10.num_classes)
        return (x_train, y_train), (x_test, y_test)

    @staticmethod
    def load_model(name="cifar10_vgg19_2048x2048"):
        return get_external_model(name)

class Cifar100:
    num_classes = 100

    @staticmethod
    def load_data():
        (x_train, y_train), (x_test, y_test) = cifar100.load_data()
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        mean_x_train = np.mean(x_train, axis=(0, 1, 2))
        std_x_train = np.std(x_train, axis=(0, 1, 2))
        x_train -= mean_x_train
        x_train /= std_x_train
        mean_x_test = np.mean(x_test, axis=(0, 1, 2))
        std_x_test = np.std(x_test, axis=(0, 1, 2))
        x_test -= mean_x_test
        x_test /= std_x_test

        y_train = keras.utils.to_categorical(y_train, Cifar100.num_classes)
        y_test = keras.utils.to_categorical(y_test, Cifar100.num_classes)
        return (x_train, y_train), (x_test, y_test)

    @staticmethod
    def load_model(name="cifar100_vgg19_2048x2048"):
        return get_external_model(name)

class Svhn:
    num_classes = 10

    @staticmethod
    def load_data():
        (x_train, y_train), (x_test, y_test) = get_svhn()
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        mean_x_train = np.mean(x_train, axis=(0, 1, 2))
        std_x_train = np.std(x_train, axis=(0, 1, 2))
        x_train -= mean_x_train
        x_train /= std_x_train
        mean_x_test = np.mean(x_test, axis=(0, 1, 2))
        std_x_test = np.std(x_test, axis=(0, 1, 2))
        x_test -= mean_x_test
        x_test /= std_x_test

        y_train = keras.utils.to_categorical(y_train - 1, Svhn.num_classes)
        y_test = keras.utils.to_categorical(y_test - 1, Svhn.num_classes)
        return (x_train, y_train), (x_test, y_test)

    @staticmethod
    def load_model(name="svhn_vgg19_2048x2048"):
        return get_external_model(name)