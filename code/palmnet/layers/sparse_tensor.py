from abc import abstractmethod

from keras import backend as K, activations, initializers, regularizers, constraints
from keras.layers import Layer, Dense, Conv2D
from keras.utils import conv_utils
import tensorflow as tf
from scipy.sparse import coo_matrix
import numpy as np
from palmnet.layers import Conv2DCustom
from palmnet.utils import create_sparse_factorization_pattern


def cast_sparsity_pattern(sparsity_pattern):
    try:
        return np.array(sparsity_pattern)
    except:
        raise ValueError("Sparsity pattern isn't well formed")


class SparseFixed(Layer):
    """
    Sparse layer with fixed sparsity pattern. The gradient will only be computed for non-zero entries.

    `SparseFixed` implements the operation:
    `output = activation(dot(input, kernel * sparsity_pattern) + bias)`
    where `activation` is the element-wise activation function
    passed as the `activation` argument, `kernel` is a weights matrix
    created by the layer, `sparsity_pattern` is a mask matrix for the `kernel` and `bias` is a bias vector created by the layer
    (only applicable if `use_bias` is `True`).

    """

    def _sparsity_initializer_factory(self):
        return lambda shape, dtype: self.sparsity_pattern

    def __init__(self, units, sparsity_pattern,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        """

        :param units: the number of output units of the layer
        :param sparsity_pattern: the sparsity pattern (mask) to apply to the kernel matrix (must be of the same shape as the kernel matrix)

        :param activation:
        :param use_bias:
        :param kernel_initializer:
        :param bias_initializer:
        :param kernel_regularizer:
        :param bias_regularizer:
        :param kernel_constraint:
        :param bias_constraint:
        :param kwargs:
        """

        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)

        assert sparsity_pattern.shape[1] == units, "sparsity pattern 2nd dim should be equal to output dim in {}".format(__class__.__name__)

        super(SparseFixed, self).__init__(**kwargs)

        self.sparsity_pattern = sparsity_pattern

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



    def build(self, input_shape):
        assert len(input_shape) >= 2
        assert input_shape[1] == self.sparsity_pattern.shape[0], "input shape should be equal to 1st dim of sparsity pattern in {}".format(__class__.__name__)

        sparse_weights = coo_matrix(self.sparsity_pattern)

        self.kernel = self.add_weight(shape=sparse_weights.data.shape,
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)


        self.sparse_op = tf.sparse.SparseTensor(list(zip(sparse_weights.row, sparse_weights.col)), self.kernel, sparse_weights.shape)

        if self.use_bias:
            self.bias = self.add_weight(shape=(self.units,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None

        super(SparseFixed, self).build(input_shape)  # Be sure to call this at the end

    def call(self, inputs):
        # multiply by the constant mask tensor so that gradient is 0 for zero entries.
        output = tf.transpose(tf.sparse.sparse_dense_matmul(tf.sparse.transpose(tf.sparse.reorder(self.sparse_op)), tf.transpose(inputs)))
        if self.use_bias:
            output = K.bias_add(output, self.bias, data_format='channels_last')
        if self.activation is not None:
            output = self.activation(output)
        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.units)


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

    def _initializer_factory(self, idx_fac):
        return lambda shape, dtype: self.sparsity_patterns[idx_fac]

    def __init__(self, units, sparsity_patterns,
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

        if sparsity_patterns is not None:
            self.sparsity_patterns = [cast_sparsity_pattern(s) for s in sparsity_patterns]
            self.nb_factor = len(sparsity_patterns)

            assert [self.sparsity_patterns[i].shape[1] == self.sparsity_patterns[i+1].shape[0] for i in range(len(self.sparsity_patterns)-1)]
            assert self.sparsity_patterns[-1].shape[1] == units, "sparsity pattern last dim should be equal to output dim in {}".format(__class__.__name__)
        else:
            self.sparsity_patterns = None

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
            'sparsity_patterns': self.sparsity_patterns
        }
        config.update(base_config)
        return config

    def build(self, input_shape):
        assert len(input_shape) >= 2
        if self.sparsity_patterns is not None:
            assert input_shape[-1] == self.sparsity_patterns[0].shape[0], "input shape should be equal to 1st dim of sparsity pattern in {}".format(__class__.__name__)
        else:
            raise ValueError("No sparsity pattern found.")

        self.scaling = self.add_weight(shape=(1,),
                                      initializer=self.scaler_initializer,
                                      name='scaler',
                                      regularizer=self.scaler_regularizer,
                                      constraint=self.scaler_constraint)

        self.kernels = []
        self.sparse_ops = []
        for i in range(self.nb_factor):
            sparse_weights = coo_matrix(self.sparsity_patterns[i])

            kernel = self.add_weight(shape=sparse_weights.data.shape,
                                          initializer=self.kernel_initializer,
                                          name='kernel_{}'.format(i),
                                          regularizer=self.kernel_regularizer,
                                          constraint=self.kernel_constraint)
            self.kernels.append(kernel)

            self.sparse_ops.append(tf.sparse.SparseTensor(list(zip(sparse_weights.row, sparse_weights.col)), kernel, sparse_weights.shape))


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

        for i in range(self.nb_factor):
            output = tf.sparse.sparse_dense_matmul(tf.sparse.transpose(tf.sparse.reorder(self.sparse_ops[i])), output)

        output = tf.transpose(self.scaling * output)

        if self.use_bias:
            output = K.bias_add(output, self.bias, data_format='channels_last')
        if self.activation is not None:
            output = self.activation(output)
        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.units)


class SparseFactorisationConv2D(Conv2DCustom):
    """
    Implementation of Conv2DCustom that uses a sparse factorization of the convolutional filters.

    The sparsity patterns are fixed on init.
    """

    def __init__(self, sparsity_patterns,
                 scaler_initializer='glorot_uniform',
                 scaler_regularizer=None,
                 scaler_constraint=None,
                 *args, **kwargs):
        """

        :param sparsity_patterns: Sparsity patterns for each sparse factor for the filter operation.
        :param scaler_initializer: Sparse factors are scaled by a scalar value.
        :param scaler_regularizer: Regularization for scaler value.
        :param scaler_constraint: Constraint for scaler value.
        :param args:
        :param kwargs:
        """

        super(SparseFactorisationConv2D, self).__init__(*args, **kwargs)


        if sparsity_patterns is not None:
            self.sparsity_patterns = [cast_sparsity_pattern(s) for s in sparsity_patterns]
            self.nb_factor = len(sparsity_patterns)

            assert [self.sparsity_patterns[i].shape[1] == self.sparsity_patterns[i + 1].shape[0] for i in range(len(self.sparsity_patterns) - 1)]
            assert self.sparsity_patterns[-1].shape[1] == self.filters, "sparsity pattern last dim should be equal to the number of filters in {}".format(__class__.__name__)
        else:
            self.sparsity_patterns = None

        self.scaler_initializer = initializers.get(scaler_initializer)
        self.scaler_regularizer = regularizers.get(scaler_regularizer)
        self.scaler_constraint = constraints.get(scaler_constraint)


    def get_config(self):
        config = super().get_config()
        config['sparsity_patterns'] = self.sparsity_patterns
        return config

    def build(self, input_shape):
        if input_shape[-1] is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')

        if self.sparsity_patterns is None:
            raise ValueError("No sparsity pattern found.")


        self.scaling = self.add_weight(shape=(1,),
                                       initializer=self.scaler_initializer,
                                       name='scaler',
                                       regularizer=self.scaler_regularizer,
                                       constraint=self.scaler_constraint)

        input_dim = input_shape[-1]
        self.kernel_shape = self.kernel_size + (input_dim, self.filters) # h x w x channels_in x channels_out

        self.kernels = []
        self.sparse_ops = []

        for i in range(self.nb_factor):
            sparse_weights = coo_matrix(self.sparsity_patterns[i])
            kernel = self.add_weight(shape=sparse_weights.data.shape,
                                     initializer=self.kernel_initializer,
                                     name='kernel_{}'.format(i),
                                     regularizer=self.kernel_regularizer,
                                     constraint=self.kernel_constraint)
            self.kernels.append(kernel)
            self.sparse_ops.append(tf.sparse.SparseTensor(list(zip(sparse_weights.row, sparse_weights.col)), kernel, sparse_weights.shape))

        if self.use_bias:
            self.bias = self.add_weight(shape=(self.filters,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None

        super(Conv2DCustom, self).build(input_shape)  # Be sure to call this at the end

    def convolution(self, X):
        X_shape = [d.value for d in X.get_shape()]
        W_shape = self.kernel_shape
        sample_size, input_height, input_width, nb_in_channels = X_shape
        filter_height, filter_width, filter_in_channels, filter_nbr = W_shape

        _, output_height, output_width, _ = self._compute_output_shape(X_shape, W_shape, self.padding_height, self.padding_width, self.strides_height, self.strides_width)

        # X_flat_bis = self.imagette_flatten(X, filter_height, filter_width, nb_in_channels, output_height, output_width, (self.strides_height, self.strides_width), (self.padding_height, self.padding_width))

        imagettes = tf.image.extract_image_patches(X, (1, filter_height, filter_width, 1), (1, self.strides_height, self.strides_width, 1), rates=[1, 1, 1, 1], padding=self.padding.upper())
        X_flat = tf.reshape(imagettes, shape=(-1, filter_height * filter_width * filter_in_channels))

        output = tf.transpose(X_flat)
        for i in range(self.nb_factor):
            output = tf.sparse.sparse_dense_matmul(tf.sparse.transpose(tf.sparse.reorder(self.sparse_ops[i])), output)

        output = tf.transpose(self.scaling * output)

        if self.use_bias:
            output += self.bias

        return K.reshape(output, (-1 if sample_size is None else sample_size, output_height, output_width, filter_nbr))

    def compute_output_shape(self, input_shape):
        return self._compute_output_shape(input_shape, self.kernel_shape, self.padding_height, self.padding_width, self.strides_height, self.strides_width)


class RandomSparseFactorisationDense(SparseFactorisationDense):
    def __init__(self, units, sparsity_factor, nb_sparse_factors=None, permutation=True, **kwargs):

        self.nb_factor = nb_sparse_factors
        self.sparsity_factor = sparsity_factor
        self.permutation = permutation

        if 'sparsity_patterns' not in kwargs:
            super(RandomSparseFactorisationDense, self).__init__(units, None, **kwargs)
        else:
            super(RandomSparseFactorisationDense, self).__init__(units, **kwargs)

    def build(self, input_shape):

        if self.nb_factor is None:
            self.nb_factor = int(np.log(max(input_shape[-1], self.units)))
        self.sparsity_patterns = create_sparse_factorization_pattern((input_shape[-1], self.units), self.sparsity_factor, self.nb_factor, self.permutation)

        super(RandomSparseFactorisationDense, self).build(input_shape)

    def get_config(self):
        base_config = super().get_config()
        config = {
            'nb_sparse_factors': self.nb_factor,
            'sparsity_factor_lst': self.sparsity_factor,
        }
        config.update(base_config)
        return config

class RandomSparseFactorisationConv2D(SparseFactorisationConv2D):
    def __init__(self, sparsity_factor, nb_sparse_factors=None, permutation=True, **kwargs):
        self.nb_factor = nb_sparse_factors
        self.sparsity_factor = sparsity_factor
        self.permutation = permutation

        if 'sparsity_patterns' not in kwargs:
            super(RandomSparseFactorisationConv2D, self).__init__(None, **kwargs)
        else:
            super(RandomSparseFactorisationConv2D, self).__init__(**kwargs)

    def build(self, input_shape):
        dim1, dim2 = np.prod(self.kernel_size) * input_shape[-1], self.filters
        if self.nb_factor is None:
            self.nb_factor = int(np.log(max(dim1, dim2)))
        self.sparsity_patterns = create_sparse_factorization_pattern((dim1, dim2), self.sparsity_factor, self.nb_factor, self.permutation)

        super(RandomSparseFactorisationConv2D, self).build(input_shape)

    def get_config(self):
        config = super().get_config()
        config['sparsity_factor_lst'] = self.sparsity_factor
        config['nb_sparse_factors'] = self.nb_factor
        return config


class SparseFactorisationConv2DDensify(Conv2DCustom):
    """
    Implementation of Conv2DCustom that uses a sparse factorization of the convolutional filters.

    The sparsity patterns are fixed on init.
    """

    def __init__(self, sparsity_patterns,
                 scaler_initializer='glorot_uniform',
                 scaler_regularizer=None,
                 scaler_constraint=None,
                 *args, **kwargs):
        """
        The filter matrix is in R^{CxF} with C the input dimension (chanelle x width x height) and F the number of filters

        :param sparsity_patterns: Sparsity patterns for each sparse factor for the filter operation.
        :param scaler_initializer: Sparse factors are scaled by a scalar value.
        :param scaler_regularizer: Regularization for scaler value.
        :param scaler_constraint: Constraint for scaler value.
        :param args:
        :param kwargs:
        """

        super(SparseFactorisationConv2DDensify, self).__init__(*args, **kwargs)


        if sparsity_patterns is not None:
            self.sparsity_patterns = [cast_sparsity_pattern(s) for s in sparsity_patterns]
            self.nb_factor = len(sparsity_patterns)

            assert [self.sparsity_patterns[i].shape[1] == self.sparsity_patterns[i + 1].shape[0] for i in range(len(self.sparsity_patterns) - 1)]
            assert self.sparsity_patterns[-1].shape[1] == self.filters, "sparsity pattern last dim should be equal to the number of filters in {}".format(__class__.__name__)
        else:
            self.sparsity_patterns = None

        self.scaler_initializer = initializers.get(scaler_initializer)
        self.scaler_regularizer = regularizers.get(scaler_regularizer)
        self.scaler_constraint = constraints.get(scaler_constraint)


    def get_config(self):
        config = super().get_config()
        config['sparsity_patterns'] = self.sparsity_patterns
        return config

    def build(self, input_shape):
        """
        The filter matrix is in R^{CxF} with C the input dimension (chanelle x width x height) and F the number of filters

        :param input_shape:
        :return:
        """
        if input_shape[-1] is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')

        if self.sparsity_patterns is None:
            raise ValueError("No sparsity pattern found.")


        self.scaling = self.add_weight(shape=(1,),
                                       initializer=self.scaler_initializer,
                                       name='scaler',
                                       regularizer=self.scaler_regularizer,
                                       constraint=self.scaler_constraint)

        input_dim = input_shape[-1]
        self.kernel_shape = self.kernel_size + (input_dim, self.filters) # h x w x channels_in x channels_out

        self.kernels = []
        self.sparse_ops = []

        for i in range(self.nb_factor):
            sparse_weights = coo_matrix(self.sparsity_patterns[i])
            kernel = self.add_weight(shape=sparse_weights.data.shape,
                                     initializer=self.kernel_initializer,
                                     name='kernel_{}'.format(i),
                                     regularizer=self.kernel_regularizer,
                                     constraint=self.kernel_constraint)
            self.kernels.append(kernel)
            self.sparse_ops.append(tf.sparse.SparseTensor(list(zip(sparse_weights.row, sparse_weights.col)), kernel, sparse_weights.shape))

        if self.use_bias:
            self.bias = self.add_weight(shape=(self.filters,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None

        super(Conv2DCustom, self).build(input_shape)  # Be sure to call this at the end

    def convolution(self, X):
        to_dense = lambda sp_tensor: tf.sparse_add(tf.zeros(sp_tensor.dense_shape), sp_tensor)
        reconstructed_kernel = to_dense(self.sparse_ops[0])
        for i in range(1, self.nb_factor):
            reconstructed_kernel = tf.matmul(reconstructed_kernel, to_dense(self.sparse_ops[i]), b_is_sparse=True)
            # output = tf.sparse.sparse_dense_matmul(tf.sparse.transpose(tf.sparse.reorder(self.sparse_ops[i])), output)

        reconstructed_kernel = self.scaling * tf.reshape(reconstructed_kernel, (*self.kernel_size, X.get_shape()[-1].value, self.filters))

        output = K.conv2d(
            X,
            reconstructed_kernel,
            strides=self.strides,
            padding=self.padding)

        if self.use_bias:
            output += self.bias

        return output
        # return K.reshape(output, (-1 if sample_size is None else sample_size, output_height, output_width, filter_nbr))

    def compute_output_shape(self, input_shape):
        return self._compute_output_shape(input_shape, self.kernel_shape, self.padding_height, self.padding_width, self.strides_height, self.strides_width)


class RandomSparseFactorisationConv2DDensify(SparseFactorisationConv2DDensify):
    def __init__(self, sparsity_factor, nb_sparse_factors=None, permutation=True, **kwargs):
        self.nb_factor = nb_sparse_factors
        self.sparsity_factor = sparsity_factor
        self.permutation = permutation

        if 'sparsity_patterns' not in kwargs:
            super(RandomSparseFactorisationConv2DDensify, self).__init__(None, **kwargs)
        else:
            super(RandomSparseFactorisationConv2DDensify, self).__init__(**kwargs)

    def build(self, input_shape):
        dim1, dim2 = np.prod(self.kernel_size) * input_shape[-1], self.filters
        if self.nb_factor is None:
            self.nb_factor = int(np.log(max(dim1, dim2)))
        self.sparsity_patterns = create_sparse_factorization_pattern((dim1, dim2), self.sparsity_factor, self.nb_factor, self.permutation)

        super(RandomSparseFactorisationConv2DDensify, self).build(input_shape)

    def get_config(self):
        config = super().get_config()
        config['sparsity_factor_lst'] = self.sparsity_factor
        config['nb_sparse_factors'] = self.nb_factor
        return config