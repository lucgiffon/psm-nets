from copy import deepcopy

import keras
from keras import Sequential
from keras.datasets import mnist
from keras.layers import Dense, Flatten
from scipy.sparse import coo_matrix

from palmnet.core.palminizable import Palminizable
from palmnet.layers.sparse_tensor import RandomSparseFactorisationDense, RandomSparseFactorisationConv2D
from palmnet.utils import create_sparse_factorization_pattern, get_sparsity_pattern
import numpy as np
import matplotlib.pyplot as plt

from skluc.utils import logger


def count_model_param_and_flops(model, dct_layer_sparse_facto_op=None):
    """
    Return the number of params and the number of flops of 2DConvolutional Layers and Dense Layers for both the base model and the compressed model.

    :return:
    """
    from keras.layers import Conv2D, Dense

    from palmnet.layers import Conv2DCustom
    from palmnet.layers.sparse_tensor import SparseFactorisationDense

    nb_param_base, nb_param_compressed, nb_flop_base, nb_flop_compressed = 0, 0, 0, 0

    param_by_layer = {}
    flop_by_layer = {}

    for layer in model.layers:
        logger.warning("Process layer {}".format(layer.name))
        if isinstance(layer, Conv2D) or isinstance(layer, Conv2DCustom):
            nb_param_layer, nb_param_compressed_layer  = Palminizable.count_nb_param_layer(layer, dct_layer_sparse_facto_op)
            nb_flop_layer, nb_flop_compressed_layer = Palminizable.count_nb_flop_conv_layer(layer, nb_param_layer, dct_layer_sparse_facto_op)

        elif isinstance(layer, Dense) or isinstance(layer, SparseFactorisationDense):
            nb_param_layer, nb_param_compressed_layer  = Palminizable.count_nb_param_layer(layer, dct_layer_sparse_facto_op)
            nb_flop_layer, nb_flop_compressed_layer = Palminizable.count_nb_flop_dense_layer(layer, nb_param_layer, dct_layer_sparse_facto_op)

        else:
            logger.warning("Layer {}, class {}, hasn't been compressed".format(layer.name, layer.__class__.__name__))
            nb_param_compressed_layer, nb_param_layer, nb_flop_layer, nb_flop_compressed_layer = 0, 0, 0, 0

        param_by_layer[layer.name] = nb_param_layer
        flop_by_layer[layer.name] = nb_flop_layer

        nb_param_base += nb_param_layer
        nb_param_compressed += nb_param_compressed_layer
        nb_flop_base += nb_flop_layer
        nb_flop_compressed += nb_flop_compressed_layer

    return nb_param_base, nb_param_compressed, nb_flop_base, nb_flop_compressed, param_by_layer, flop_by_layer

def main_create_sparse_facto():
    dim1 = 20
    dim2 = 20
    sparsity = 2
    nb_fac = int(np.log(max(dim1, dim2)))

    sparse_factors = create_sparse_factorization_pattern((dim1, dim2), sparsity, nb_fac)
    print(sparse_factors)

def show_weights(nb_sparse_factors, sparsity_patterns, weights_before, weights_after, name):
    an_weights_before = []
    an_weights_after = []
    for i, sparse_pattern in enumerate(sparsity_patterns):
        coo_sparse = coo_matrix(sparse_pattern)
        coo_sparse.data = weights_before[i]
        coo_sparse = coo_sparse.toarray()[:100, :]
        an_weights_before.append(coo_sparse)

        coo_sparse = coo_matrix(sparse_pattern)
        coo_sparse.data = weights_after[i]
        coo_sparse = coo_sparse.toarray()[:100, :]
        an_weights_after.append(coo_sparse)

    an_sparsity_pattern = [w[:100, :] for w in sparsity_patterns]
    sparsity_pattern_before = [get_sparsity_pattern(w) for w in an_weights_before]
    sparsity_pattern_after = [get_sparsity_pattern(w) for w in an_weights_after]

    f, ax = plt.subplots(4, nb_sparse_factors)
    plt.title(name)

    for i in range(nb_sparse_factors):
        ax[0, i].imshow(an_sparsity_pattern[i])
    for i in range(nb_sparse_factors):
        ax[1, i].imshow(an_weights_before[i])
    for i in range(nb_sparse_factors):
        ax[2, i].imshow(an_weights_after[i])
    for i in range(nb_sparse_factors):
        ax[3, i].imshow(np.abs(sparsity_pattern_before[i] - sparsity_pattern_after[i]))
    plt.show()

def main():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    img_rows, img_cols = 28, 28
    num_classes = 10
    batch_size = 64
    epochs = 10
    hidden_layer_dim = 100
    sparse_factors = 3
    sparsity_factor = 2
    nb_filter = 8
    kernel_size = (5, 5)
    padding = "same"

    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    # x_train = x_train.reshape(x_train.shape[0], img_rows*  img_cols* 1)
    # x_test = x_test.reshape(x_test.shape[0], img_rows* img_cols* 1)
    input_shape = x_train[0].shape
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)


    model = Sequential()
    model.add(RandomSparseFactorisationConv2D(input_shape=input_shape, filters=nb_filter, kernel_size=kernel_size, padding=padding, sparsity_factor=sparsity_factor, nb_sparse_factors=sparse_factors))
    model.add(Flatten())
    model.add(RandomSparseFactorisationDense(units=hidden_layer_dim, sparsity_factor=sparsity_factor, nb_sparse_factors=sparse_factors))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])

    count_model_param_and_flops(model)

    weights_before_dense = deepcopy(model.layers[2].get_weights())[1:-1]
    sparsity_patterns_dense = model.layers[2].sparsity_patterns
    weights_before_conv = deepcopy(model.layers[0].get_weights())[1:-1]
    sparsity_patterns_conv = model.layers[0].sparsity_patterns

    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(x_test, y_test))

    weights_after_dense = model.layers[2].get_weights()[1:-1]
    weights_after_conv = model.layers[0].get_weights()[1:-1]

    show_weights(sparse_factors, sparsity_patterns_dense, weights_before_dense, weights_after_dense, "factorisation dense")
    show_weights(sparse_factors, sparsity_patterns_conv, weights_before_conv, weights_after_conv, "factorisation conv")

    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])


if __name__ == "__main__":
    main()