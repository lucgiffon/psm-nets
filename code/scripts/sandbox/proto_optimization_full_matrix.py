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
    def __init__(self, shape, nb_factor, sparsity_factor, entropy_placeholder, l2_placeholder):

        assert nb_factor >=2, "Layer must have at least two sparse factors"

        self.nb_factor = nb_factor
        if type(sparsity_factor) is int:
            self.sparsity_factor_lst = [sparsity_factor] * nb_factor
        else:
            self.sparsity_factor_lst = sparsity_factor

        self.entropy_regularization_parameter = entropy_placeholder
        self.l2_regularization_parameter = l2_placeholder
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

    def add_block_diag(self, shape, sparsity_factor, name="block_diag_B"):
        block_diag = create_random_block_diag(*shape, sparsity_factor)
        sparse_block_diag = coo_matrix(block_diag)
        kernel_block_diag = tf.Variable(tf.ones(shape=sparse_block_diag.data.shape), dtype=tf.float32)
        sparse_tensor_block_diag = tf.sparse.SparseTensor(list(zip(sparse_block_diag.row, sparse_block_diag.col)), kernel_block_diag, sparse_block_diag.shape)
        return kernel_block_diag, sparse_tensor_block_diag

    def add_permutation(self, d, name="permutation_P"):
        # return tf.Variable(create_permutation_matrix(d), dtype=tf.float32)
        return tf.Variable(np.random.rand(d, d), dtype=tf.float32)
        # return tf.Variable(np.eye(d), dtype=tf.float32)

    def build(self):

        input_dim = self.shape[-1]
        inner_dim = min(self.shape)
        output_dim = self.shape[0]

        # create first P: dense with regularization
        self.permutations = [self.add_permutation(input_dim, name="permutation_P_input")]

        # create first B; sparse block diag
        kernel_block_diag, sparse_tensor_block_diag = self.add_block_diag((input_dim, inner_dim), name="block_diag_B_input", sparsity_factor=self.sparsity_factor_lst[0])
        self.kernels = [kernel_block_diag]
        self.sparse_block_diag_ops = [sparse_tensor_block_diag]

        for i in range(self.nb_factor-1):

            # create P: dense with regularization
            self.permutations.append(self.add_permutation(inner_dim, name="permutation_P_{}".format(i+1)))

            if i < (self.nb_factor-1)-1:
                # create B: sparse block diagonal
                kernel_block_diag, sparse_tensor_block_diag = self.add_block_diag((inner_dim, inner_dim), name="block_diag_B_{}".format(i), sparsity_factor=self.sparsity_factor_lst[i])
                self.kernels.append(kernel_block_diag)
                self.sparse_block_diag_ops.append(sparse_tensor_block_diag)

        # create last B: block diagonal sparse
        kernel_block_diag, sparse_tensor_block_diag = self.add_block_diag((inner_dim, output_dim), name="block_diag_B_output", sparsity_factor=self.sparsity_factor_lst[-1])
        self.kernels.append(kernel_block_diag)
        self.sparse_block_diag_ops.append(sparse_tensor_block_diag)

        # create last P: dense with regularization
        self.permutations.append(self.add_permutation(output_dim, name="permutation_P_output"))
        return self

    def call(self):

        output = tf.multiply(softmax(self.permutations[0], axis=1), softmax(self.permutations[0], axis=0))
        for i in range(self.nb_factor):
            output = tf.sparse.sparse_dense_matmul(tf.sparse.transpose(tf.sparse.reorder(self.sparse_block_diag_ops[i])), output)
            output = tf.matmul(tf.multiply(softmax(self.permutations[i+1], axis=1), softmax(self.permutations[i+1], axis=0)), output)

        output = tf.transpose(output)
        return output

    def get_softmax_entropy_regularization(self):
        reg = self.regularization_softmax_entropy(self.permutations[0])
        for perm_matrix in self.permutations[1:]:
            reg = tf.add(reg, self.regularization_softmax_entropy(perm_matrix))
        return self.entropy_regularization_parameter * reg

    def get_l2_regularization(self):
        reg = tf.nn.l2_loss(self.kernels[0])
        for kern in self.kernels[1:]:
            reg = tf.add(reg, tf.nn.l2_loss(kern))
        return self.l2_regularization_parameter * reg


def sparse_facto_train(obj_mat, entropy_param, l2_param, lr, epochs, sparsity_factor, nb_factor, session):
    entropy_placeholder = tf.placeholder("float", None)
    l2_placeholder = tf.placeholder("float", None)

    sparse_facto_obj = tfSparseFactorization(obj_mat.shape, nb_factor, sparsity_factor, entropy_placeholder, l2_placeholder).build()

    objective = tf.divide(tf.square(tf.norm(obj_mat - sparse_facto_obj.call())), tf.norm(obj_mat)) + sparse_facto_obj.get_softmax_entropy_regularization()
    optimizer = tf.train.AdamOptimizer(lr)
    train = optimizer.minimize(objective)
    init = tf.initialize_all_variables()

    session.run(init)
    for step in range(epochs):
        # entropy_param = np.exp(entropy_param_reg * step) - 1
        session.run(train, feed_dict={entropy_placeholder: entropy_param, l2_placeholder: l2_param})
        print("entropy_param", entropy_param, "step", step, "objective:", session.run(objective, feed_dict={entropy_placeholder: 0, l2_placeholder: 0}))

    return sparse_facto_obj

def main():

    d = 16
    nb_factor = int(np.log(d))
    sparsity_factor = 2
    epochs = 10000000

    X, y = make_blobs(n_samples=1000, n_features=d)
    # X = np.random.rand(16, 1000)
    X = X.T
    X /= np.linalg.norm(X)
    X = tf.constant(X, dtype=tf.float32)

    had = np.array(hadamard(d))
    had = had / np.linalg.norm(had)
    had = tf.constant(had, dtype=tf.float32)
    # objective = tf.divide(tf.square(tf.norm(tf.matmul(had, X) - tf.matmul(sparse_facto_obj.call(), X))), tf.norm(tf.matmul(had, X))) + sparse_facto_obj.get_softmax_entropy_regularization()

    y_max_entropy_reg = 1
    x_max = epochs
    # entropy_param_reg = np.log(y_max_entropy_reg + 1) / x_max
    entropy_param_reg = 0.1
    with tf.Session() as session:
        obj = sparse_facto_train(obj_mat=had, entropy_param=entropy_param_reg, l2_param=0, lr=1e-4, epochs=epochs, sparsity_factor=sparsity_factor, nb_factor=nb_factor, session=session)

        final_mat = session.run(obj.call())
        plt.imshow(final_mat)
        plt.show()
        # for perm in sparse_facto_obj.permutations:
        #     perm_ev = np.array(session.run(softmax(perm, axis=1)))
        #     m = Munkres()
        #     indexes = np.array(m.compute(-perm_ev))
        #     rows = indexes[:, 0]
        #     cols = indexes[:, 1]
        #     vals = perm_ev[tuple(indexes.T)]
        #     proj = coo_matrix((np.ones(rows.size), (rows, cols)))
        #     plt.imshow(proj.toarray())
        #     plt.show()

if __name__ == "__main__":
    main()