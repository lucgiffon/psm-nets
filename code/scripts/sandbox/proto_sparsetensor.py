from copy import deepcopy

from keras import Sequential, activations, initializers, regularizers, constraints
from keras.engine import Layer
from keras.layers import Dense, Conv2D, Flatten
from keras.datasets import mnist
import numpy as np
import tensorflow as tf
import keras
from scipy.linalg import block_diag
from scipy.sparse import coo_matrix, csr_matrix
import matplotlib.pyplot as plt
from operator import mul
import pathlib
import keras.backend as K

from palmnet.core.palminize import Palminizable
from palmnet.layers.sparse_tensor import SparseFixed, SparseFactorisationDense, SparseFactorisationConv2D
from palmnet.utils import insert_layer_nonseq, get_sparsity_pattern
import pickle

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

from palmnet.utils import insert_layer_nonseq, get_sparsity_pattern
import pickle

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
    model.add(SparseFixed(input_shape=input_shape, sparsity_pattern=sparsity_pattern, units=hidden_layer_dim))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])

    weights_before = deepcopy(model.layers[0].get_weights()[0])

    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=1,
              verbose=1,
              validation_data=(x_test, y_test))

    weights_after = model.layers[0].get_weights()[0]

    def get_sparsity_pattern(arr):
        non_zero = arr != 0
        sparsity_pattern = np.zeros_like(arr)
        sparsity_pattern[non_zero] = 1
        return sparsity_pattern

    an_weights_before = coo_matrix(sparsity_pattern)
    an_weights_before.data = weights_before
    an_weights_before = an_weights_before.toarray()[:100, :]

    an_weights_after = coo_matrix(sparsity_pattern)
    an_weights_after.data = weights_after
    an_weights_after = an_weights_after.toarray()[:100, :]

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

def tfmainSparseLayer():
    sparsity_patterns_dense = np.random.choice([0, 1], (784, 100), p=[0.8, 0.2]).astype('float32')

    def init_weights(shape, sparse=False):
        """ Weight initialization """
        weights = tf.random_normal(shape, stddev=0.1)
        if not sparse:
            return tf.Variable(weights)
        else:
            weights = np.random.rand(*shape).astype("float32")

            sparse_weights = coo_matrix(weights * sparsity_patterns_dense)
            data = tf.Variable(sparse_weights.data)

            return tf.sparse.SparseTensor(list(zip(sparse_weights.row, sparse_weights.col)), data, sparse_weights.shape)

    def forwardprop(X, w_1, w_2):
        """
        Forward-propagation.
        IMPORTANT: yhat is not softmax since TensorFlow's softmax_cross_entropy_with_logits() does that internally.
        """
        if not isinstance(w_1, tf.sparse.SparseTensor):
            prod = tf.matmul(X, w_1)
        else:
            prod = tf.transpose(tf.sparse.sparse_dense_matmul(tf.sparse.transpose(tf.sparse.reorder(w_1)), tf.transpose(X)))

        h = tf.nn.relu(prod)  # The \sigma function
        yhat = tf.matmul(h, w_2)  # The \varphi function
        return yhat

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    img_rows, img_cols = 28,  28
    num_classes = 10
    batch_size = 64
    epochs = 3
    hidden_layer_dim = 100

    x_train = x_train.reshape(x_train.shape[0], img_rows * img_cols)
    x_test = x_test.reshape(x_test.shape[0], img_rows * img_cols)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    x_size = x_train.shape[-1]
    y_size = y_train.shape[-1]
    input_shape = (x_size,)

    # sparsity_pattern = np.random.choice([0, 1], (input_shape[-1], hidden_layer_dim), p=[0.9, 0.1]).astype('float32')

    X = tf.placeholder("float", shape=[None, x_train.shape[-1]])
    y = tf.placeholder("float", shape=[None, y_train.shape[-1]])

    w_1 = init_weights((x_size, hidden_layer_dim), sparse=True)
    w_2 = init_weights((hidden_layer_dim, y_size))

    yhat = forwardprop(X, w_1, w_2)
    predict = tf.argmax(yhat, axis=1)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=yhat))
    updates = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)

    before = coo_matrix((sess.run(w_1)[1], (sess.run(w_1)[0][:, 0], sess.run(w_1)[0][:, 1]))).toarray()
    before_2 = sess.run(w_2)
    print(before)

    for epoch in range(1):
        # Train with each example
        for i in range(len(x_train[:100])):
            sess.run(updates, feed_dict={X: x_train[i: i + 1], y: y_train[i: i + 1]})

        train_accuracy = np.mean(np.argmax(y_train, axis=1) ==
                                 sess.run(predict, feed_dict={X: x_train, y: y_train}))
        test_accuracy = np.mean(np.argmax(y_test, axis=1) ==
                                sess.run(predict, feed_dict={X: x_test, y: y_test}))

        print("Epoch = %d, train accuracy = %.2f%%, test accuracy = %.2f%%"
              % (epoch + 1, 100. * train_accuracy, 100. * test_accuracy))

    after = coo_matrix((sess.run(w_1)[1], (sess.run(w_1)[0][:, 0], sess.run(w_1)[0][:, 1]))).toarray()
    after2 = sess.run(w_2)
    print(after)
    assert (before_2 != after2).any()
    assert (before != after).any()

    sess.close()
    def get_sparsity_pattern(arr):
        non_zero = arr != 0
        sparsity_pattern = np.zeros_like(arr)
        sparsity_pattern[non_zero] = 1
        return sparsity_pattern

    an_weights_before = before[:100, :]
    an_weights_after = after[:100, :]
    an_sparsity_pattern = sparsity_patterns_dense[:100, :]
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

def mainSparseFactorisation():

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    img_rows, img_cols = 28,  28
    num_classes = 10
    batch_size = 64
    epochs = 1
    hidden_layer_dim = 100
    sparse_factors = 3
    nb_filter = 8
    kernel_size = (5, 5)
    padding = "same"

    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1,)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)


    sparsity_patterns_dense = [np.random.choice([0, 1], (6272, hidden_layer_dim), p=[0.8, 0.2]).astype('float32')]
    for i in range(1, sparse_factors):
        sparsity_patterns_dense.append(np.random.choice([0, 1], (hidden_layer_dim, hidden_layer_dim), p=[0.8, 0.2]).astype('float32'))

    block_diag_mat = block_diag(*[np.ones((2, 2)) for _ in range(4)])
    sparsity_patterns_conv = [np.random.choice([0, 1], (mul(*kernel_size) * 1, nb_filter), p=[0.8, 0.2]).astype('float32')]
    for i in range(1, sparse_factors):

        sparsity_patterns_conv.append(np.eye(nb_filter)[np.random.permutation(nb_filter)] @ block_diag_mat @ np.eye(nb_filter)[np.random.permutation(nb_filter)])

    model = Sequential()
    model.add(SparseFactorisationConv2D(sparsity_patterns=sparsity_patterns_conv, input_shape=input_shape, filters=nb_filter, kernel_size=kernel_size, padding=padding))
    model.add(Flatten())
    model.add(SparseFactorisationDense(input_shape=input_shape, units=hidden_layer_dim, sparsity_patterns=sparsity_patterns_dense))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])


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
    epochs = 1

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
            factors = [coo_matrix(fac.toarray()) for fac in sparse_factorization[1].get_list_of_factors()]
            sparsity_patterns = [get_sparsity_pattern(w.toarray()) for w in factors]
            factor_data = [f.data for f in factors]

            # create new layer
            if isinstance(layer, Dense):
                hidden_layer_dim = layer.units
                activation = layer.activation
                replacing_layer = SparseFactorisationDense(units=hidden_layer_dim, sparsity_patterns=sparsity_patterns, use_bias=layer.use_bias, activation=activation)
                replacing_weights = [np.array(scaling)[None]] + factor_data + [layer.get_weights()[-1]] if layer.use_bias else []
                fine_tuned_model = insert_layer_nonseq(fine_tuned_model, layer_name, lambda: replacing_layer, position="replace")
                replacing_layer.set_weights(replacing_weights)

            elif isinstance(layer, Conv2D):
                nb_filters = layer.filters
                kernel_size = layer.kernel_size
                activation = layer.activation
                replacing_layer = SparseFactorisationConv2D(filters=nb_filters, kernel_size=kernel_size, sparsity_patterns=sparsity_patterns, use_bias=layer.use_bias, activation=activation)
                replacing_weights = [np.array(scaling)[None]] + factor_data + [layer.get_weights()[-1]] if layer.use_bias else []
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
    mainSparseFactorisation()