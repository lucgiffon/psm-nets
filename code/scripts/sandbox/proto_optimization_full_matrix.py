import keras
import tensorflow as tf
from keras.datasets import mnist
from scipy.linalg import hadamard
from scipy.sparse import coo_matrix
import numpy as np
from tensorflow.python.ops.nn_ops import softmax
import matplotlib.pyplot as plt

from palmnet.utils import create_random_block_diag, create_permutation_matrix


class tfSparseFactorization:
    """
    Layer which implements a sparse factorization with fixed sparsity pattern for all factors. The gradient will only be computed for non-zero entries.

    `SparseFactorisationDense` implements the operation:
    `output = activation(dot(input, prod([kernel[i] * sparsity_pattern[i] for i in range(nb_factor)]) + bias)`
    where `activation` is the element-wise activation function
    passed as the `activation` argument, `kernel` is a weights matrix
    created by the layer, `sparsity_pattern` is a mask matrix for the `kernel` and `bias` is a bias vector created by the layer
    (only applicable if `use_bias` is `True`).

    """
    def __init__(self, shape, nb_factor, sparsity_factor,
                 entropy_regularization_parameter):

        assert nb_factor >=2, "Layer must have at least two sparse factors"

        self.nb_factor = nb_factor
        self.sparsity_factor = sparsity_factor

        self.entropy_regularization_parameter = entropy_regularization_parameter
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
        weight_matrix_proba = softmax(weight_matrix)
        # high entropy
        entropy = self.entropy(weight_matrix_proba)
        regularization = self.entropy_regularization_parameter * entropy
        return regularization

    def add_block_diag(self, shape, name="block_diag_B"):
        block_diag = create_random_block_diag(*shape, self.sparsity_factor)
        sparse_block_diag = coo_matrix(block_diag)
        kernel_block_diag = tf.Variable(tf.ones(shape=sparse_block_diag.data.shape), dtype=tf.float32)
        sparse_tensor_block_diag = tf.sparse.SparseTensor(list(zip(sparse_block_diag.row, sparse_block_diag.col)), kernel_block_diag, sparse_block_diag.shape)
        return kernel_block_diag, sparse_tensor_block_diag

    def add_permutation(self, d, name="permutation_P"):
        return tf.Variable(create_permutation_matrix(d), dtype=tf.float32)

    def build(self):

        input_dim = self.shape[-1]
        inner_dim = min(self.shape)
        output_dim = self.shape[0]

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
        kernel_block_diag, sparse_tensor_block_diag = self.add_block_diag((inner_dim, output_dim), name="block_diag_B_output")
        self.kernels.append(kernel_block_diag)
        self.sparse_block_diag_ops.append(sparse_tensor_block_diag)

        # create last P: dense with regularization
        self.permutations.append(self.add_permutation(output_dim, name="permutation_P_output"))
        return self

    def call(self):

        output = softmax(self.permutations[0], axis=1)
        for i in range(self.nb_factor):
            output = tf.sparse.sparse_dense_matmul(tf.sparse.transpose(tf.sparse.reorder(self.sparse_block_diag_ops[i])), output)
            output = tf.matmul(softmax(self.permutations[i+1], axis=1), output)

        output = tf.transpose(output)
        return output


if __name__ == "__main__":


    d = 16
    nb_factor = int(np.log(d))
    sparsity_factor=2

    X = np.random.rand(16, 1000)
    X /= np.linalg.norm(X)
    X = tf.constant(X, dtype=tf.float32)


    sparse_facto_obj = tfSparseFactorization((d, d), nb_factor, sparsity_factor, 1.).build()
    had = np.array(hadamard(d))
    had = had / np.linalg.norm(had)
    had = tf.constant(had, dtype=tf.float32)
    # objective = tf.divide(tf.square(tf.norm(tf.matmul(had, X) - tf.matmul(sparse_facto_obj.call(), X))), tf.norm(tf.matmul(had, X)))
    objective = tf.divide(tf.square(tf.norm(had - sparse_facto_obj.call())), tf.norm(had))
    optimizer = tf.train.AdamOptimizer(1e-4)
    train = optimizer.minimize(objective)

    init = tf.initialize_all_variables()
    with tf.Session() as session:
        session.run(init)
        print("starting at", "objective:", session.run(objective))

        for step in range(100000):
            session.run(train)
            print("step", step, "objective:", session.run(objective))

        final_mat = session.run(sparse_facto_obj.call())
        plt.imshow(final_mat)
        plt.show()