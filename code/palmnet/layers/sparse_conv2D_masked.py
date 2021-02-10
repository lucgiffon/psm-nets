import tensorflow as tf
from keras import initializers, regularizers, constraints, backend as K

from palmnet.layers import Conv2DCustom
from palmnet.utils import cast_sparsity_pattern, sparse_facto_init, NAME_INIT_SPARSE_FACTO
import numpy as np


class SparseConv2D(Conv2DCustom):
    """
    Implementation of Conv2DCustom that uses a sparse factorization of the convolutional filters.

    The sparsity patterns are fixed on init.
    """

    def __init__(self, sparsity_pattern,
                 kernel_initializer="glorot_uniform",
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
        if kernel_initializer != NAME_INIT_SPARSE_FACTO:
            super(SparseConv2D, self).__init__(kernel_initializer=kernel_initializer, *args, **kwargs)
            self.__kernel_initializer = kernel_initializer
        else:
            super(SparseConv2D, self).__init__(kernel_initializer=lambda *args, **kwargs: None, *args, **kwargs)
            self.__kernel_initializer = kernel_initializer

        if sparsity_pattern is not None:
            self.sparsity_pattern = cast_sparsity_pattern(sparsity_pattern)
        else:
            self.sparsity_pattern = None
        self.image_max_size = -1


    def get_config(self):
        config = super().get_config()
        config["sparsity_pattern"] = self.sparsity_pattern
        if self.__kernel_initializer == NAME_INIT_SPARSE_FACTO:
            config['kernel_initializer'] = NAME_INIT_SPARSE_FACTO
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

        if self.sparsity_pattern is None:
            raise ValueError("No sparsity pattern found.")


        input_dim = input_shape[-1]
        self.kernel_shape = self.kernel_size + (input_dim, self.filters) # h x w x channels_in x channels_out

        input_dim, output_dim = self.sparsity_pattern.shape

        if self.__kernel_initializer == NAME_INIT_SPARSE_FACTO:
            # matrix will be applied on right
            kernel_init = sparse_facto_init((input_dim, output_dim), 0, self.sparsity_pattern, multiply_left=False)
        else:
            kernel_init = self.kernel_initializer

        self.kernel = self.add_weight(shape=(input_dim, output_dim),
                                 initializer=kernel_init,
                                 name='kernel',
                                 regularizer=self.kernel_regularizer,
                                 constraint=self.kernel_constraint)
        self.sparsity_mask = K.constant(self.sparsity_pattern, dtype="float32", name="sparsity_mask")

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
        reconstructed_kernel = self.kernel * self.sparsity_mask
        reconstructed_kernel = tf.reshape(reconstructed_kernel, (*self.kernel_size, X.get_shape()[-1].value, self.filters))

        self.image_max_size = max(self.image_max_size, np.prod([val.value for val in X.get_shape()[1:]]))

        output = K.conv2d(
            X,
            reconstructed_kernel,
            strides=self.strides,
            padding=self.padding)

        # self.image_max_size = max(self.image_max_size, np.prod([val.value for val in output.get_shape()[1:]]))

        if self.use_bias:
            output = K.bias_add(output, self.bias)

        return output
        # return K.reshape(output, (-1 if sample_size is None else sample_size, output_height, output_width, filter_nbr))

    def compute_output_shape(self, input_shape):
        return self._compute_output_shape(input_shape, self.kernel_shape, self.padding_height, self.padding_width, self.strides_height, self.strides_width)