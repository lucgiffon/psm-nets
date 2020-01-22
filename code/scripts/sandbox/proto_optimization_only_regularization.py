from copy import deepcopy

import keras
import tensorflow as tf
from keras.datasets import mnist
from munkres import Munkres
from scipy.linalg import hadamard
from scipy.sparse import coo_matrix
import numpy as np
from sklearn.datasets import make_blobs
from tensorflow.python.ops.nn_ops import softmax
import matplotlib.pyplot as plt
from scipy.special import softmax as sci_softmax
from palmnet.utils import create_random_block_diag, create_permutation_matrix


class tfPermutation:
    """
    Layer which implements a sparse factorization with fixed sparsity pattern for all factors. The gradient will only be computed for non-zero entries.

    `SparseFactorisationDense` implements the operation:
    `output = activation(dot(input, prod([kernel[i] * sparsity_pattern[i] for i in range(nb_factor)]) + bias)`
    where `activation` is the element-wise activation function
    passed as the `activation` argument, `kernel` is a weights matrix
    created by the layer, `sparsity_pattern` is a mask matrix for the `kernel` and `bias` is a bias vector created by the layer
    (only applicable if `use_bias` is `True`).

    """
    def __init__(self, shape, entropy_placeholder):

        self.entropy_regularization_parameter = entropy_placeholder
        self.shape = shape

    @staticmethod
    def entropy(matrix):
        p_logp =  tf.multiply(matrix, tf.log(matrix))
        return -tf.reduce_sum(p_logp)

    @staticmethod
    def sum_to_one_constraint(matrix):
        columns = tf.reduce_sum.sum(matrix, axis=0) - tf.constant(tf.ones((matrix.shape[1],), dtype=tf.float32))
        lines = tf.reduce_sum.sum(matrix, axis=1) - tf.constant(tf.ones((matrix.shape[1],), dtype=tf.float32))
        return tf.reduce_sum(columns) + tf.reduce_sum(lines)

    def regularization_softmax_entropy(self, weight_matrix):
        # sum to one
        weight_matrix_proba_1 = softmax(weight_matrix, axis=1)
        weight_matrix_proba_0 = softmax(weight_matrix, axis=0)
        # high entropy
        entropy = tf.add(self.entropy(weight_matrix_proba_1), self.entropy(weight_matrix_proba_0))
        regularization = self.entropy_regularization_parameter * entropy
        return regularization

    def add_permutation(self, d, name="permutation_P"):
        # return tf.Variable(create_permutation_matrix(d), dtype=tf.float32)
        return tf.Variable(np.random.rand(d, d), dtype=tf.float32)
        # return tf.Variable(np.eye(d), dtype=tf.float32)

    def build(self):

        input_dim = self.shape[-1]

        # create first P: dense with regularization
        self.permutations = self.add_permutation(input_dim, name="permutation_P")
        return self

    def get_softmax_entropy_regularization(self):
        reg = self.regularization_softmax_entropy(self.permutations)
        return self.entropy_regularization_parameter * reg



def regularization_train(shape, entropy_param, lr, epochs, session):
    entropy_placeholder = tf.placeholder("float", None)

    permutation_obj = tfPermutation(shape, entropy_placeholder).build()

    objective = permutation_obj.get_softmax_entropy_regularization()
    optimizer = tf.train.AdamOptimizer(lr)
    train = optimizer.minimize(objective)
    init = tf.initialize_all_variables()

    session.run(init)
    for step in range(epochs):
        # entropy_param = np.exp(entropy_param_reg * step) - 1
        session.run(train, feed_dict={entropy_placeholder: entropy_param})
        print("entropy_param", entropy_param, "step", step, "objective:", session.run(objective, feed_dict={entropy_placeholder: entropy_param}))

    return permutation_obj

def main():
    d = 512
    epochs = 10000
    with tf.Session() as session:
        entropy_param_reg = regularization_train((d,d), 100, 0.1, epochs, session)

        mat = np.array(session.run(entropy_param_reg.permutations))
        plt.imshow(mat)
        plt.show()
        s_mat = sci_softmax(mat, axis=0) * sci_softmax(mat, axis=1)
        plt.imshow(s_mat)
        plt.show()
        thresholded = (s_mat > 1e-1).astype(int)
        print(np.sum(thresholded @thresholded.T))
        print(thresholded@thresholded.T - np.eye(d))

def main_2():

    d = 8
    epochs = 1000
    P = np.abs(np.random.rand(d, d))
    j = 0
    while j < 100:
        P = np.exp(-P)
        i = 0
        while i < epochs:
            P /= np.linalg.norm(P, axis=1, ord=1)[:, np.newaxis]
            P /= np.linalg.norm(P, axis=0, ord=1)[np.newaxis, :]
            i+= 1
        j +=1

    plt.imshow(P)
    plt.title("P")
    plt.show()



if __name__ == "__main__":
    main_2()
