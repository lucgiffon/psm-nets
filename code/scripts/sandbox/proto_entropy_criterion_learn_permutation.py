from copy import deepcopy
from munkres import Munkres
from keras import Sequential, initializers, regularizers, activations, constraints
from keras.layers import Dense, Conv2D, Flatten
from keras.datasets import mnist
import numpy as np
import keras
import matplotlib.pyplot as plt
from operator import mul
import pathlib
from keras.layers.convolutional import _Conv
from keras.utils import conv_utils
from scipy.sparse import coo_matrix
from scipy.special import softmax
from tensorflow.python.keras.engine.base_layer import InputSpec

from palmnet.core.palminizable import Palminizable
# from palmnet.layers.sparse_masked import SparseFixed, SparseFactorisationConv2D#, SparseFactorisationDense
from palmnet.layers.pbp_layer import PBPDense, PBPDenseDensify
from palmnet.utils import insert_layer_nonseq, get_sparsity_pattern, create_random_block_diag, create_permutation_matrix
import pickle
from keras.layers import Layer
import keras.backend as K
import tensorflow as tf



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
    sparsity_factor = 3

    x_train = x_train.reshape(x_train.shape[0], img_rows *img_cols* 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows *img_cols* 1)
    input_shape = (img_rows* img_cols* 1,)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)


    model = Sequential()
    # model.add(SparseFactorisationConv2D(sparsity_patterns=sparsity_patterns_conv, input_shape=input_shape, filters=nb_filter, kernel_size=kernel_size, padding=padding))
    # model.add(Flatten())
    model.add(PBPDenseDensify(input_shape=input_shape, units=hidden_layer_dim, nb_factor=sparse_factors, sparsity_factor=sparsity_factor, entropy_regularization_parameter=1))
    model.add(PBPDenseDensify(units=hidden_layer_dim, nb_factor=sparse_factors, sparsity_factor=sparsity_factor, entropy_regularization_parameter=1))
    model.add(Dense(num_classes, activation='softmax'))

    tb = keras.callbacks.tensorboard_v1.TensorBoard(log_dir='./logs', histogram_freq=1, batch_size=32, write_graph=True, write_grads=1, write_images=True, embeddings_freq=0,
                                               embeddings_layer_names=None, embeddings_metadata=None, embeddings_data=None, update_freq='epoch')

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adam(),
                  metrics=['accuracy'])

    names = [weight.name for layer in model.layers for weight in layer.weights]
    weights = model.get_weights()

    for name, w in zip(names, weights):
        try:
            plt.imshow(w)
            plt.title("before " + name)
            plt.show()
        except:
            print("except" + name)

    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(x_test, y_test),
              # callbacks=[tb]
              )

    names = [weight.name for layer in model.layers for weight in layer.weights]
    weights = model.get_weights()

    for name, w in zip(names, weights):
        if "permutation" in name:
            m = Munkres()
            indexes = m.compute(softmax(w, axis=1))
            coo_matrix(w[indexes])
        try:
            plt.imshow(softmax(w, axis=1))
            plt.title(name)
            plt.show()
        except:
            print("except" + name)

    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

if __name__ == "__main__":
    mainSparseFactorisation()