from keras import initializers, regularizers, activations, constraints
from scipy.sparse import coo_matrix
from palmnet.utils import create_random_block_diag, create_permutation_matrix
from keras.layers import Layer
import keras.backend as K
import tensorflow as tf
import numpy as np

# class PBPDense(Layer):
#     """
#     Layer which implements a sparse factorization with learnable sparsity pattern for all factors.
#
#     `SparseFactorisationDense` implements the operation: activation(PBPBPBP + bias) with all B and P differents from each other.
#
#     where:
#     B are block diagonal matrices with `sparsity_factor_lst` value in each block. Blocks are learnable.
#     P are permutation matrices learnable with the soft cross entropy criterion.
#     `activation` is the element-wise activation function passed as the `activation` argument,
#     (only applicable if `use_bias` is `True`).
#
#     """
#     def __init__(self, units, nb_factor, sparsity_factor,
#                  entropy_regularization_parameter,
#                  activation=None,
#                  use_bias=True,
#                  scaler_initializer='glorot_uniform',
#                  kernel_initializer='glorot_uniform',
#                  permutation_initializer='permutation',
#                  bias_initializer='zeros',
#                  scaler_regularizer=None,
#                  kernel_regularizer=None,
#                  bias_regularizer=None,
#                  scaler_constraint=None,
#                  kernel_constraint=None,
#                  bias_constraint=None,
#                  **kwargs):
#
#         if 'input_shape' not in kwargs and 'input_dim' in kwargs:
#             kwargs['input_shape'] = (kwargs.pop('input_dim'),)
#
#         super(PBPDense, self).__init__(**kwargs)
#
#         assert nb_factor is None or nb_factor >=2, "Layer must have at least two sparse factors"
#         self.nb_factor = nb_factor
#         self.sparsity_factor = sparsity_factor
#
#         self.entropy_regularization_parameter = entropy_regularization_parameter
#
#         self.units = units
#         self.activation = activations.get(activation)
#         self.use_bias = use_bias
#         # todo faire un initialiseur particulier (type glorot) qui prend en compte la sparsité
#         if permutation_initializer == "permutation":
#             self.permutation_initializer = lambda shape, dtype: create_permutation_matrix(shape[0], dtype)
#         else:
#             self.permutation_initializer = initializers.get(permutation_initializer)
#         self.permutation_initializer_name = permutation_initializer
#
#
#         self.kernel_initializer = initializers.get(kernel_initializer)
#         self.bias_initializer = initializers.get(bias_initializer)
#         self.kernel_regularizer = regularizers.get(kernel_regularizer)
#         self.bias_regularizer = regularizers.get(bias_regularizer)
#         self.kernel_constraint = constraints.get(kernel_constraint)
#         self.bias_constraint = constraints.get(bias_constraint)
#
#         self.scaler_initializer = initializers.get(scaler_initializer)
#         self.scaler_regularizer = regularizers.get(scaler_regularizer)
#         self.scaler_constraint = constraints.get(scaler_constraint)
#
#     def get_config(self):
#         base_config = super().get_config()
#         config = {
#             'units': self.units,
#             'activation': activations.serialize(self.activation),
#             'use_bias': self.use_bias,
#             'kernel_initializer': initializers.serialize(self.kernel_initializer),
#             'bias_initializer': initializers.serialize(self.bias_initializer),
#             'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
#             'bias_regularizer': regularizers.serialize(self.bias_regularizer),
#             'kernel_constraint': constraints.serialize(self.kernel_constraint),
#             'bias_constraint': constraints.serialize(self.bias_constraint),
#             'nb_factor': self.nb_factor,
#             'sparsity_factor_lst':self.sparsity_factor,
#             'entropy_regularization_parameter':self.entropy_regularization_parameter,
#             'permutation_initializer': 'permutation' if self.permutation_initializer_name == 'permutation' else initializers.serialize(self.permutation_initializer)
#         }
#         config.update(base_config)
#         return config
#
#     @staticmethod
#     def entropy(matrix):
#         """
#         Return the entropy of the matrix. With M the matrix:
#
#         entropy = - sum_i,j M_i,j * log(M_i,j)
#         """
#         p_logp =  tf.multiply(matrix, K.log(matrix))
#         return -K.sum(p_logp)
#
#     @staticmethod
#     def sum_to_one(matrix):
#         """
#         return the sum of difference between 1 and the sum of columns and rows values
#
#         = sum_col (sum(col) - 1) + sum_row (sum(row)-1)
#         """
#         columns = K.sum(matrix, axis=0) - K.ones((matrix.shape[1],))
#         lines = K.sum(matrix, axis=1) - K.ones((matrix.shape[1],))
#         return K.sum(columns) + K.sum(lines)
#
#     def regularization_softmax_entropy(self, weight_matrix):
#         """
#         Compute softmax entropy regularisation: softmax is applied to the columns of the weight matrix and then the entropy is computed
#         on the resulting matrix.
#         """
#         # sum to one
#         weight_matrix_proba = K.softmax(weight_matrix)
#         # high entropy
#         entropy = self.entropy(weight_matrix_proba)
#         regularization = self.entropy_regularization_parameter * entropy
#         return regularization
#
#     def add_block_diag(self, shape, name="block_diag_B"):
#         block_diag = create_random_block_diag(*shape, self.sparsity_factor)
#         sparse_block_diag = coo_matrix(block_diag)
#         kernel_block_diag = self.add_weight(shape=sparse_block_diag.data.shape,
#                                      initializer=self.kernel_initializer,
#                                      name=name,
#                                      regularizer=self.kernel_regularizer,
#                                      constraint=self.kernel_constraint)
#         sparse_tensor_block_diag = tf.sparse.SparseTensor(list(zip(sparse_block_diag.row, sparse_block_diag.col)), kernel_block_diag, sparse_block_diag.shape)
#         return kernel_block_diag, sparse_tensor_block_diag
#
#     def add_permutation(self, d, name="permutation_P"):
#         permutation = self.add_weight(
#             shape=(d, d),
#             initializer=self.permutation_initializer,
#             name=name,
#             regularizer=self.regularization_softmax_entropy
#         )
#         return permutation
#
#     def build(self, input_shape):
#         assert len(input_shape) >= 2
#
#         input_dim = input_shape[-1]
#         inner_dim = min(input_dim, self.units)
#
#         if self.nb_factor is None or self.nb_factor == "None":
#             self.nb_factor = int(np.log(max(input_dim, self.units)))
#
#         # create first P: dense with regularization
#         self.permutations = [self.add_permutation(input_dim, name="permutation_P_input")]
#
#         # create first B; sparse block diag
#         kernel_block_diag, sparse_tensor_block_diag = self.add_block_diag((input_dim, inner_dim), name="block_diag_B_input")
#         self.kernels = [kernel_block_diag]
#         self.sparse_block_diag_ops = [sparse_tensor_block_diag]
#
#         for i in range(self.nb_factor-1):
#
#             # create P: dense with regularization
#             self.permutations.append(self.add_permutation(inner_dim, name="permutation_P_{}".format(i+1)))
#
#             if i < (self.nb_factor-1)-1:
#                 # create B: sparse block diagonal
#                 kernel_block_diag, sparse_tensor_block_diag = self.add_block_diag((inner_dim, inner_dim), name="block_diag_B_{}".format(i))
#                 self.kernels.append(kernel_block_diag)
#                 self.sparse_block_diag_ops.append(sparse_tensor_block_diag)
#
#         # create last B: block diagonal sparse
#         kernel_block_diag, sparse_tensor_block_diag = self.add_block_diag((inner_dim, self.units), name="block_diag_B_output")
#         self.kernels.append(kernel_block_diag)
#         self.sparse_block_diag_ops.append(sparse_tensor_block_diag)
#
#         # create last P: dense with regularization
#         self.permutations.append(self.add_permutation(self.units, name="permutation_P_output"))
#
#         if self.use_bias:
#             self.bias = self.add_weight(shape=(self.units,),
#                                         initializer=self.bias_initializer,
#                                         name='bias',
#                                         regularizer=self.bias_regularizer,
#                                         constraint=self.bias_constraint)
#         else:
#             self.bias = None
#
#         super(PBPDense, self).build(input_shape)  # Be sure to call this at the end
#
#     def call(self, inputs):
#
#         output = tf.transpose(inputs)
#         output = K.dot(K.softmax(self.permutations[0], axis=1), output)
#         for i in range(self.nb_factor):
#             output = tf.sparse.sparse_dense_matmul(tf.sparse.transpose(tf.sparse.reorder(self.sparse_block_diag_ops[i])), output)
#             output = K.dot(K.softmax(self.permutations[i+1], axis=1), output)
#
#         output = tf.transpose(output)
#
#         if self.use_bias:
#             output = K.bias_add(output, self.bias, data_format='channels_last')
#         if self.activation is not None:
#             output = self.activation(output)
#         return output
#
#     def compute_output_shape(self, input_shape):
#         return (input_shape[0], self.units)


class PBPDenseDensify(Layer):
    """
    Layer which implements a sparse factorization with learnable sparsity pattern for all factors.

    `SparseFactorisationDense` implements the operation: activation(PBPBPBP + bias) with all B and P differents from each other.

    where:
    B are block diagonal matrices with `sparsity_factor_lst` value in each block. Blocks are learnable.
    P are permutation matrices learnable with the soft cross entropy criterion.
    `activation` is the element-wise activation function passed as the `activation` argument,
    (only applicable if `use_bias` is `True`).

    """
    def __init__(self, units, nb_factor, sparsity_factor,
                 entropy_regularization_parameter,
                 add_entropies=False,
                 activation=None,
                 use_bias=True,
                 scaler_initializer='glorot_uniform',
                 kernel_initializer='glorot_uniform',
                 permutation_initializer='identity',
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

        super(PBPDenseDensify, self).__init__(**kwargs)

        assert nb_factor is None or nb_factor >=2, "Layer must have at least two sparse factors"
        self.nb_factor = nb_factor
        self.sparsity_factor = sparsity_factor
        self.add_entropies = add_entropies

        self.entropy_regularization_parameter = entropy_regularization_parameter

        self.units = units
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        # todo faire un initialiseur particulier (type glorot) qui prend en compte la sparsité
        if permutation_initializer == "permutation":
            self.permutation_initializer = lambda shape, dtype: create_permutation_matrix(shape[0], dtype)
        elif permutation_initializer == "identity":
            self.permutation_initializer = lambda shape, dtype: np.eye(shape[0], dtype=dtype)
        else:
            self.permutation_initializer = initializers.get(permutation_initializer)
        self.permutation_initializer_name = permutation_initializer


        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        self.scaler_initializer = initializers.get(scaler_initializer)
        self.scaler_regularizer = regularizers.get(scaler_regularizer)
        self.scaler_constraint = constraints.get(scaler_constraint)

    def get_config(self):
        base_config = super().get_config()
        config = {
            'units': self.units,
            'activation': activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint),
            'nb_factor': self.nb_factor,
            'sparsity_factor':self.sparsity_factor,
            'add_entropies': self.add_entropies,
            'entropy_regularization_parameter':self.entropy_regularization_parameter,
            'permutation_initializer': 'permutation' if self.permutation_initializer_name == 'permutation' else initializers.serialize(self.permutation_initializer)
        }
        config.update(base_config)
        return config

    @staticmethod
    def entropy(matrix):
        """
        Return the entropy of the matrix. With M the matrix:

        entropy = - sum_i,j M_i,j * log(M_i,j)
        """
        p_logp =  tf.multiply(matrix, K.log(matrix))
        return -K.sum(p_logp)

    @staticmethod
    def sum_to_one(matrix):
        """
        return the sum of difference between 1 and the sum of columns and rows values

        = sum_col (sum(col) - 1) + sum_row (sum(row)-1)
        """
        columns = K.sum(matrix, axis=0) - K.ones((matrix.shape[1],))
        lines = K.sum(matrix, axis=1) - K.ones((matrix.shape[1],))
        return K.sum(columns) + K.sum(lines)

    def regularization_softmax_entropy(self, weight_matrix):
        """
        Compute softmax entropy regularisation: softmax is applied to the columns of the weight matrix and then the entropy is computed
        on the resulting matrix.
        """
        # sum to one
        weight_matrix_proba_1 = K.softmax(weight_matrix, axis=1)
        weight_matrix_proba_0 = K.softmax(weight_matrix, axis=0)
        # high entropy
        if self.add_entropies:
            entropy = tf.add(self.entropy(weight_matrix_proba_1), self.entropy(weight_matrix_proba_0))
        else:
            entropy = tf.multiply(self.entropy(weight_matrix_proba_1), self.entropy(weight_matrix_proba_0))
        regularization = self.entropy_regularization_parameter * entropy
        return regularization

    def add_block_diag(self, shape, name="block_diag_B"):
        block_diag_mask = K.constant(create_random_block_diag(*shape, self.sparsity_factor, mask=True), dtype="float32", name="{}_mask".format(name))

        kernel_block_diag = self.add_weight(shape=shape,
                                     initializer=self.kernel_initializer,
                                     name=name,
                                     regularizer=self.kernel_regularizer,
                                     constraint=self.kernel_constraint)
        return kernel_block_diag, block_diag_mask

    def add_permutation(self, d, name="permutation_P"):
        permutation = self.add_weight(
            shape=(d, d),
            initializer=self.permutation_initializer,
            name=name,
            regularizer=self.regularization_softmax_entropy
        )
        return permutation

    def build(self, input_shape):
        assert len(input_shape) >= 2

        input_dim = input_shape[-1]
        inner_dim = min(input_dim, self.units)

        if self.nb_factor is None or self.nb_factor == "None":
            self.nb_factor = int(np.log(max(input_dim, self.units)))

        # create first P: dense with regularization
        self.permutations = [self.add_permutation(input_dim, name="permutation_P_input")]

        # create first B; sparse block diag
        kernel_block_diag, block_diag_mask = self.add_block_diag((input_dim, inner_dim), name="block_diag_B_input")
        self.kernels = [kernel_block_diag]
        self.block_diag_masks = [block_diag_mask]

        for i in range(self.nb_factor-1):

            # create P: dense with regularization
            self.permutations.append(self.add_permutation(inner_dim, name="permutation_P_{}".format(i+1)))

            if i < (self.nb_factor-1)-1:
                # create B: sparse block diagonal
                kernel_block_diag, block_diag_mask = self.add_block_diag((inner_dim, inner_dim), name="block_diag_B_{}".format(i))
                self.kernels.append(kernel_block_diag)
                self.block_diag_masks.append(block_diag_mask)

        # create last B: block diagonal sparse
        kernel_block_diag, block_diag_mask = self.add_block_diag((inner_dim, self.units), name="block_diag_B_output")
        self.kernels.append(kernel_block_diag)
        self.block_diag_masks.append(block_diag_mask)

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

        super(PBPDenseDensify, self).build(input_shape)  # Be sure to call this at the end

    def call(self, inputs):
        output = inputs
        output = K.dot(output, tf.multiply(K.softmax(self.permutations[0], axis=1), K.softmax(self.permutations[0], axis=0)))
        for i in range(self.nb_factor):
            output = K.dot(output, self.kernels[i] * self.block_diag_masks[i])
            output = K.dot(output, tf.multiply(K.softmax(self.permutations[i+1], axis=1), K.softmax(self.permutations[i+1], axis=0)))

        if self.use_bias:
            output = K.bias_add(output, self.bias, data_format='channels_last')
        if self.activation is not None:
            output = self.activation(output)

        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.units)
