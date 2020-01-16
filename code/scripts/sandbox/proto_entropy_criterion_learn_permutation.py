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

from palmnet.core.palminize import Palminizable
# from palmnet.layers.sparse_masked import SparseFixed, SparseFactorisationConv2D#, SparseFactorisationDense
from palmnet.utils import insert_layer_nonseq, get_sparsity_pattern, create_random_block_diag, create_permutation_matrix
import pickle
from keras.layers import Layer
import keras.backend as K
import tensorflow as tf


class SparseFactorisationDense(Layer):
    """
    Layer which implements a sparse factorization with fixed sparsity pattern for all factors. The gradient will only be computed for non-zero entries.

    `SparseFactorisationDense` implements the operation:
    `output = activation(dot(input, prod([kernel[i] * sparsity_pattern[i] for i in range(nb_factor)]) + bias)`
    where `activation` is the element-wise activation function
    passed as the `activation` argument, `kernel` is a weights matrix
    created by the layer, `sparsity_pattern` is a mask matrix for the `kernel` and `bias` is a bias vector created by the layer
    (only applicable if `use_bias` is `True`).

    """
    def __init__(self, units, nb_factor, sparsity_factor,
                 entropy_regularization_parameter,
                 activation=None,
                 use_bias=True,
                 scaler_initializer='glorot_uniform',
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 scaler_regularizer=None,
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 scaler_constraint=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):

        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)

        super(SparseFactorisationDense, self).__init__(**kwargs)

        assert nb_factor >=2, "Layer must have at least two sparse factors"
        self.nb_factor = nb_factor
        self.sparsity_factor = sparsity_factor

        self.entropy_regularization_parameter = entropy_regularization_parameter

        self.units = units
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        # todo faire un initialiseur particulier (type glorot) qui prend en compte la sparsité
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        self.scaler_initializer = initializers.get(scaler_initializer)
        self.scaler_regularizer = regularizers.get(scaler_regularizer)
        self.scaler_constraint = constraints.get(scaler_constraint)

    @staticmethod
    def entropy(matrix):
        p_logp =  tf.multiply(matrix, K.log(matrix))
        return -K.sum(p_logp)

    @staticmethod
    def sum_to_one_constraint(matrix):
        columns = K.sum(matrix, axis=0) - K.ones((matrix.shape[1],))
        lines = K.sum(matrix, axis=1) - K.ones((matrix.shape[1],))
        return K.sum(columns) + K.sum(lines)

    def regularization_softmax_entropy(self, weight_matrix):
        # sum to one
        weight_matrix_proba = K.softmax(weight_matrix)
        # high entropy
        entropy = self.entropy(weight_matrix_proba)
        regularization = self.entropy_regularization_parameter * entropy
        return regularization

    def add_block_diag(self, shape, name="block_diag_B"):
        block_diag = create_random_block_diag(*shape, self.sparsity_factor)
        sparse_block_diag = coo_matrix(block_diag)
        kernel_block_diag = self.add_weight(shape=sparse_block_diag.data.shape,
                                     initializer=self.kernel_initializer,
                                     name=name,
                                     regularizer=self.kernel_regularizer,
                                     constraint=self.kernel_constraint)
        sparse_tensor_block_diag = tf.sparse.SparseTensor(list(zip(sparse_block_diag.row, sparse_block_diag.col)), kernel_block_diag, sparse_block_diag.shape)
        return kernel_block_diag, sparse_tensor_block_diag

    def add_permutation(self, d, name="permutation_P"):
        permutation = self.add_weight(
            shape=(d, d),
            initializer=lambda shape, dtype: create_permutation_matrix(shape[0], dtype),
            name=name,
            regularizer=self.regularization_softmax_entropy
        )
        return permutation

    def build(self, input_shape):
        assert len(input_shape) >= 2

        input_dim = input_shape[-1]
        inner_dim = min(input_dim, self.units)

        # create first P: dense with regularization
        self.permutations = [self.add_permutation(input_dim, name="permutation_P_input")]

        # create first B; sparse block diag
        kernel_block_diag, sparse_tensor_block_diag = self.add_block_diag((input_dim, inner_dim), name="block_diag_B_input")
        self.kernels = [kernel_block_diag]
        self.sparse_block_diag_ops = [sparse_tensor_block_diag]

        for i in range(self.nb_factor-1):

            # create P: dense with regularization
            self.permutations.append(self.add_permutation(inner_dim, name="permutation_P_{}".format(i+1)))

            if i < (self.nb_factor-1)-1:
                # create B: sparse block diagonal
                kernel_block_diag, sparse_tensor_block_diag = self.add_block_diag((inner_dim, inner_dim), name="block_diag_B_{}".format(i))
                self.kernels.append(kernel_block_diag)
                self.sparse_block_diag_ops.append(sparse_tensor_block_diag)

        # create last B: block diagonal sparse
        kernel_block_diag, sparse_tensor_block_diag = self.add_block_diag((inner_dim, self.units), name="block_diag_B_output")
        self.kernels.append(kernel_block_diag)
        self.sparse_block_diag_ops.append(sparse_tensor_block_diag)

        # create last P: dense with regularization
        self.permutations.append(self.add_permutation(self.units, name="permutation_P_output"))

        if self.use_bias:
            self.bias = self.add_weight(shape=(self.units,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None

        super(SparseFactorisationDense, self).build(input_shape)  # Be sure to call this at the end

    def call(self, inputs):

        output = tf.transpose(inputs)
        output = K.dot(K.softmax(self.permutations[0], axis=1), output)
        for i in range(self.nb_factor):
            output = tf.sparse.sparse_dense_matmul(tf.sparse.transpose(tf.sparse.reorder(self.sparse_block_diag_ops[i])), output)
            output = K.dot(K.softmax(self.permutations[i+1], axis=1), output)

        output = tf.transpose(output)

        if self.use_bias:
            output = K.bias_add(output, self.bias, data_format='channels_last')
        if self.activation is not None:
            output = self.activation(output)
        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.units)

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
    epochs = 1
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
    model.add(SparseFactorisationDense(input_shape=input_shape, units=hidden_layer_dim, nb_factor=sparse_factors, sparsity_factor=sparsity_factor, entropy_regularization_parameter=1))
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