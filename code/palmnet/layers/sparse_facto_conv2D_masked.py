import tensorflow as tf
from keras import initializers, regularizers, constraints, backend as K

from palmnet.layers import Conv2DCustom
from palmnet.utils import cast_sparsity_pattern


class SparseFactorisationConv2D(Conv2DCustom):
    """
    Implementation of Conv2DCustom that uses a sparse factorization of the convolutional filters.

    The sparsity patterns are fixed on init.
    """

    def __init__(self, sparsity_patterns,
                 use_scaling=True,
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

        super(SparseFactorisationConv2D, self).__init__(*args, **kwargs)

        if sparsity_patterns is not None:
            self.sparsity_patterns = [cast_sparsity_pattern(s) for s in sparsity_patterns]
            self.nb_factor = len(sparsity_patterns)

            assert [self.sparsity_patterns[i].shape[1] == self.sparsity_patterns[i + 1].shape[0] for i in range(len(self.sparsity_patterns) - 1)]
            assert self.sparsity_patterns[-1].shape[1] == self.filters, "sparsity pattern last dim should be equal to the number of filters in {}".format(__class__.__name__)
        else:
            self.sparsity_patterns = None

        self.use_scaling = use_scaling
        self.scaler_initializer = initializers.get(scaler_initializer)
        self.scaler_regularizer = regularizers.get(scaler_regularizer)
        self.scaler_constraint = constraints.get(scaler_constraint)


    def get_config(self):
        config = super().get_config()
        config['sparsity_patterns'] = self.sparsity_patterns
        config["scaler_initializer"] = self.scaler_initializer
        config["scaler_regularizer"] = self.scaler_regularizer
        config["scaler_constraint"] = self.scaler_constraint
        config['use_scaling'] = self.use_scaling
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

        if self.use_scaling:
            self.scaling = self.add_weight(shape=(1,),
                                           initializer=self.scaler_initializer,
                                           name='scaler',
                                           regularizer=self.scaler_regularizer,
                                           constraint=self.scaler_constraint)
        else:
            self.scaling = None

        input_dim = input_shape[-1]
        self.kernel_shape = self.kernel_size + (input_dim, self.filters) # h x w x channels_in x channels_out

        self.kernels = []
        self.sparsity_masks = []

        for i in range(self.nb_factor):
            input_dim, output_dim = self.sparsity_patterns[i].shape

            kernel = self.add_weight(shape=(input_dim, output_dim),
                                     initializer=self.kernel_initializer,
                                     name='kernel_{}'.format(i),
                                     regularizer=self.kernel_regularizer,
                                     constraint=self.kernel_constraint)
            self.kernels.append(kernel)
            sparsity_mask = K.constant(self.sparsity_patterns[i], dtype="float32", name="sparsity_mask_{}".format(i))
            self.sparsity_masks.append(sparsity_mask)

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
        reconstructed_kernel = self.kernels[0] * self.sparsity_masks[0]
        for i in range(1, self.nb_factor):
            reconstructed_kernel = K.dot(reconstructed_kernel, self.kernels[i] * self.sparsity_masks[i])
            # output = tf.sparse.sparse_dense_matmul(tf.sparse.transpose(tf.sparse.reorder(self.sparse_ops[i])), output)

        if self.use_scaling:
            reconstructed_kernel = self.scaling * tf.reshape(reconstructed_kernel, (*self.kernel_size, X.get_shape()[-1].value, self.filters))
        else:
            reconstructed_kernel = tf.reshape(reconstructed_kernel, (*self.kernel_size, X.get_shape()[-1].value, self.filters))

        output = K.conv2d(
            X,
            reconstructed_kernel,
            strides=self.strides,
            padding=self.padding)

        if self.use_bias:
            output = K.bias_add(output, self.bias)

        return output
        # return K.reshape(output, (-1 if sample_size is None else sample_size, output_height, output_width, filter_nbr))

    def compute_output_shape(self, input_shape):
        return self._compute_output_shape(input_shape, self.kernel_shape, self.padding_height, self.padding_width, self.strides_height, self.strides_width)