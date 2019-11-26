from copy import deepcopy

from keras import Sequential
from keras.layers import Dense, Conv2D, Flatten
from keras.datasets import mnist
import numpy as np
import keras
import matplotlib.pyplot as plt
from operator import mul
import pathlib

from palmnet.core.palminize import Palminizable
from palmnet.layers import SparseFixed, SparseFactorisationConv2D, SparseFactorisationDense
from palmnet.utils import insert_layer_nonseq, get_sparsity_pattern
import pickle


def show_weights(nb_sparse_factors, sparsity_patterns, weights_before, weights_after, name):
    an_weights_before = [w[:100, :] for w in weights_before]
    an_weights_after = [w[:100, :] for w in weights_after]
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


def fac_prod(lst_fac):
    prod = lst_fac[0]
    for i in range(1, len(lst_fac)):
        prod = np.dot(prod, lst_fac[i])
    return prod


def mainSparseLayer():

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    img_rows, img_cols = 28,  28
    num_classes = 10
    batch_size = 64
    epochs = 3
    hidden_layer_dim = 100

    x_train = x_train.reshape(x_train.shape[0], img_rows * img_cols)
    x_test = x_test.reshape(x_test.shape[0], img_rows * img_cols)
    input_shape = (img_rows * img_cols,)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    sparsity_pattern = np.random.choice([0, 1], (input_shape[-1], hidden_layer_dim), p=[0.9, 0.1]).astype('float32')

    model = Sequential()
    model.add(SparseFixed(input_shape=input_shape, output_dim=hidden_layer_dim, sparsity_pattern=sparsity_pattern))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])

    model.layers[0].set_weights([model.layers[0].get_weights()[0] * sparsity_pattern])
    weights_before = deepcopy(model.layers[0].get_weights()[0])

    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(x_test, y_test))

    weights_after = model.layers[0].get_weights()[0]

    def get_sparsity_pattern(arr):
        non_zero = arr != 0
        sparsity_pattern = np.zeros_like(arr)
        sparsity_pattern[non_zero] = 1
        return sparsity_pattern

    an_weights_before = weights_before[:100, :]
    an_weights_after = weights_after[:100, :]
    an_sparsity_pattern = sparsity_pattern[:100, :]
    sparsity_pattern_before = get_sparsity_pattern(an_weights_before)
    sparsity_pattern_after = get_sparsity_pattern(an_weights_after)

    f, ax = plt.subplots(2, 4)
    ax[0, 0].imshow(an_sparsity_pattern)
    ax[0, 1].imshow(an_weights_before)
    ax[0, 2].imshow(an_weights_after)
    ax[0, 3].imshow(np.abs(an_weights_before - an_weights_after))

    # ax[1, 0].imshow(model.layers[0].get_weights()[1][:100, :])
    ax[1, 1].imshow(sparsity_pattern_before)
    ax[1, 2].imshow(sparsity_pattern_after)
    ax[1, 3].imshow(np.abs(sparsity_pattern_before - sparsity_pattern_after))

    plt.show()

    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])


def mainSparseFactorisation():

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    img_rows, img_cols = 28,  28
    num_classes = 10
    batch_size = 64
    epochs = 10
    hidden_layer_dim = 100
    sparse_factors = 3
    nb_filter = 5
    kernel_size = (5, 5)
    padding = "same"

    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)


    sparsity_patterns_dense = [np.random.choice([0, 1], (3920, hidden_layer_dim), p=[0.8, 0.2]).astype('float32')]
    for i in range(1, sparse_factors):
        sparsity_patterns_dense.append(np.random.choice([0, 1], (hidden_layer_dim, hidden_layer_dim), p=[0.8, 0.2]).astype('float32'))

    sparsity_patterns_conv = [np.random.choice([0, 1], (mul(*kernel_size) * 1, nb_filter), p=[0.8, 0.2]).astype('float32')]
    for i in range(1, sparse_factors):
        sparsity_patterns_conv.append(np.random.choice([0, 1], (nb_filter, nb_filter), p=[0.8, 0.2]).astype('float32'))

    model = Sequential()
    model.add(SparseFactorisationConv2D(sparsity_patterns=sparsity_patterns_conv, input_shape=input_shape, filters=nb_filter, kernel_size=kernel_size, padding=padding))
    model.add(Flatten())
    model.add(SparseFactorisationDense(units=hidden_layer_dim, sparsity_patterns=sparsity_patterns_dense))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])

    new_weights_dense = [model.layers[2].get_weights()[0]] + [w*sparsity_patterns_dense[i] for i, w in enumerate(model.layers[2].get_weights()[1:-1])] + [model.layers[2].get_weights()[-1]]
    model.layers[2].set_weights(new_weights_dense)

    new_weights_conv = [model.layers[0].get_weights()[0]] + [w*sparsity_patterns_conv[i] for i, w in enumerate(model.layers[0].get_weights()[1:-1])] + [model.layers[0].get_weights()[-1]]
    model.layers[0].set_weights(new_weights_conv)

    weights_before_dense = deepcopy(model.layers[2].get_weights())[1:-1]
    weights_before_conv = deepcopy(model.layers[0].get_weights())[1:-1]


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


def mainRefinePalminizedModel():
    root_source_dir = pathlib.Path("/home/luc/PycharmProjects/palmnet/results/")
    expe_path = "2019/10/0_0_hierarchical_palminize"
    src_results_dir = root_source_dir / expe_path

    mypalminizedmodel = pickle.load(open(src_results_dir / "1939301_model_layers.pckle", "rb"))  # type: Palminizable

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    img_rows, img_cols = 28, 28
    num_classes = 10
    batch_size = 64
    epochs = 10

    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    base_model = mypalminizedmodel.base_model
    dct_name_facto = mypalminizedmodel.sparsely_factorized_layers
    base_score = base_model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', base_score[0])
    print('Test accuracy:', base_score[1])
    #
    palminized_model = mypalminizedmodel.compressed_model
    palminized_score = palminized_model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', palminized_score[0])
    print('Test accuracy:', palminized_score[1])
    modified_layer_names = []
    fine_tuned_model = deepcopy(palminized_model)
    for i, layer in enumerate(fine_tuned_model.layers):
        layer_name = layer.name
        sparse_factorization = dct_name_facto[layer_name]

        if sparse_factorization != (None, None):
            scaling = sparse_factorization[0]
            factors = [fac.toarray() for fac in sparse_factorization[1].get_list_of_factors()]
            sparsity_patterns = [get_sparsity_pattern(w) for w in factors]

            # create new layer
            if isinstance(layer, Dense):
                hidden_layer_dim = layer.units
                activation = layer.activation
                replacing_layer = SparseFactorisationDense(units=hidden_layer_dim, sparsity_patterns=sparsity_patterns, use_bias=layer.use_bias, activation=activation)
                replacing_weights = [np.array(scaling)[None]] + factors + [layer.get_weights()[-1]] if layer.use_bias else []
                fine_tuned_model = insert_layer_nonseq(fine_tuned_model, layer_name, lambda: replacing_layer, position="replace")
                replacing_layer.set_weights(replacing_weights)

            elif isinstance(layer, Conv2D):
                nb_filters = layer.filters
                kernel_size = layer.kernel_size
                activation = layer.activation
                replacing_layer = SparseFactorisationConv2D(filters=nb_filters, kernel_size=kernel_size, sparsity_patterns=sparsity_patterns, use_bias=layer.use_bias, activation=activation)
                replacing_weights = [np.array(scaling)[None]] + factors + [layer.get_weights()[-1]] if layer.use_bias else []
                fine_tuned_model = insert_layer_nonseq(fine_tuned_model, layer_name, lambda: replacing_layer, position="replace")
                replacing_layer.set_weights(replacing_weights)

            else:
                raise ValueError("unknown layer class")
            modified_layer_names.append((layer_name, replacing_layer.name))



    fine_tuned_model.compile(loss='categorical_crossentropy',
                  optimizer="adam",
                  metrics=['categorical_accuracy'])

    finetuned_score = fine_tuned_model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', finetuned_score[0])
    print('Test accuracy:', finetuned_score[1])

    fine_tuned_model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(x_test, y_test))

    score = fine_tuned_model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    for layer in fine_tuned_model.layers:
        if isinstance(layer, SparseFactorisationDense) or isinstance(layer, SparseFactorisationConv2D):
            sparse_factors = len(layer.sparsity_patterns)
            sparsity_patterns_layer = layer.sparsity_patterns
            weights = layer.get_weights()[1:-1]
            show_weights(sparse_factors, sparsity_patterns_layer, weights, weights, "factorisation {}".format(layer.__class__.__name__))


if __name__ == "__main__":
    mainRefinePalminizedModel()