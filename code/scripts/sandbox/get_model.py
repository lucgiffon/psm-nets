from copy import deepcopy

import keras
from keras.layers import Conv2D, Dense
from skluc.utils import logger

from palmnet.utils import root_dir
from skluc.utils.osutils import download_file, check_file_md5
# import tensorflow as tf
import keras.backend as K
import numpy as np
from palmnet.data import Mnist


def imagette_flatten(X, window_h, window_w, window_c, out_h, out_w, stride=1, padding=0):
    X_padded = K.pad(X, [[0, 0], [padding, padding], [padding, padding], [0, 0]])

    windows = []
    for y in range(out_h):
        for x in range(out_w):
            window = K.slice(X_padded, [0, y * stride, x * stride, 0], [-1, window_h, window_w, -1])
            windows.append(window)
    stacked = K.stack(windows)  # shape : [out_h, out_w, n, filter_h, filter_w, c]

    return K.reshape(stacked, [-1, window_c * window_w * window_h])

def convolution(X, W, b, padding, stride):
    """

    :param X: The input 3D tensor.
    :param W: The 4D tensor of filters.
    :param b: The bias for each filter.
    :param padding: The padding size in both dimension.
    :param stride: The stride size in both dimension.
    :return:
    """
    # todo padding "half" on the basis of W
    sample_size, input_height, input_width, nb_in_channels = map(lambda d: d.value, X.get_shape())
    filter_height, filter_width, filter_in_channels, filter_nbr = [d.value for d in W.get_shape()]

    output_height = (input_height + 2*padding - filter_height) // stride + 1
    output_width = (input_width + 2*padding - filter_width) // stride + 1

    X_flat = imagette_flatten(X, filter_height, filter_width, nb_in_channels, output_height, output_width, stride, padding)
    W_flat = K.reshape(W, [filter_height*filter_width*nb_in_channels, filter_nbr])

    z = K.dot(X_flat, W_flat) + b     # b: 1 X filter_n

    return K.permute_dimensions(K.reshape(z, [output_height, output_width, sample_size, filter_nbr]), [2, 0, 1, 3])

def apply_palm(matrix):
    logger.warning("'Apply palm function' doesn't do anything!")
    return matrix
    # return np.ones_like(matrix)

def palminize_layer(layer_obj):
    """
    Takes a keras layer object as entry and modify its weights as reconstructed by the palm approximation.

    The layer is modifed in place but the inner weight tensor is returned modifed.

    :param layer_obj: The layer object to which modify weights
    :return: The new weights
    """
    if isinstance(layer_obj, Conv2D):
        logger.debug("Find {}".format(layer_obj.__class__.__name__))
        layer_weights, layer_bias = layer_obj.get_weights()
        filter_height, filter_width, in_chan, out_chan = layer_weights.shape
        filter_matrix = layer_weights.reshape(filter_height*filter_width*in_chan, out_chan)
        reconstructed_filter_matrix = apply_palm(filter_matrix)
        new_layer_weights = reconstructed_filter_matrix.reshape(filter_height, filter_width, in_chan, out_chan)
        layer_obj.set_weights((new_layer_weights, layer_bias))
        return new_layer_weights
    elif isinstance(layer_obj, Dense):
        logger.debug("Find {}".format(layer_obj.__class__.__name__))
        layer_weights, layer_bias = layer_obj.get_weights()
        reconstructed_dense_matrix = apply_palm(layer_weights)
        new_layer_weights = reconstructed_dense_matrix
        layer_obj.set_weights((new_layer_weights, layer_bias))
        return new_layer_weights
    else:
        logger.debug("Find {}. Can't Palminize this. Pass.".format(layer_obj.__class__.__name__))
        return

def palminize_model(model):
    """
    Takes a keras model object as entry and returns a version of it with all weights matrix palminized.

    Modifications are in-place but the model is still returned.

    :param model: Keras model
    :return: The same, model object with new weights.
    """
    for layer in model.layers:
        palminize_layer(layer)
    return model

if __name__ == "__main__":
    url_mnist = "https://pageperso.lis-lab.fr/~luc.giffon/saved_models/mnist_lenet_1570207294.h5"
    md5sum = "26d44827c84d44a9fc8f4e021b7fe4d2"
    download_path = download_file(url_mnist, root_dir / "models")
    check_file_md5(download_path, md5sum)
    (x_train, y_train), (x_test, y_test) = Mnist.load_data()

    model = keras.models.load_model(download_path, compile=False)
    model_init = deepcopy(model)

    model = palminize_model(model)
    model.compile(loss='binary_crossentropy',
                  optimizer="adam",
                  metrics=['accuracy'])

    score, acc = model.evaluate(x_test, y_test)



    print(score, acc)



    print(download_path)