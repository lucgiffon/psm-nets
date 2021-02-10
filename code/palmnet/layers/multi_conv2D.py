import tensorflow as tf
from keras import initializers, regularizers, constraints, backend as K

from palmnet.layers.tf_conv2d_custom import TFConv2DCustom
from palmnet.utils import cast_sparsity_pattern, sparse_facto_init, NAME_INIT_SPARSE_FACTO
import numpy as np
import tensorflow_model_optimization as tfmot



class MultiConv2D(TFConv2DCustom, tfmot.sparsity.keras.PrunableLayer):
    """
    Implementation of Conv2DCustom that uses a sparse factorization of the convolutional filters.

    The sparsity patterns are fixed on init.
    """

    def __init__(self, nb_factors,
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
        super(MultiConv2D, self).__init__(*args, **kwargs)
        self.nb_factor = nb_factors
        self.image_max_size = -1

    def get_config(self):
        config = super().get_config()
        config["nb_factors"] = self.nb_factor
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


        input_dim = input_shape[-1]
        self.kernel_shape = self.kernel_size + (input_dim, self.filters)  # h x w x channels_in x channels_out

        self.kernels = []
        input_shape = (self.kernel_shape[0] * self.kernel_shape[1] * self.kernel_shape[2]).value
        inner_dim = min(input_shape, self.filters)

        for i in range(self.nb_factor):
            if i == 0:
                input_dim = input_shape
                output_dim = inner_dim
            elif i < self.nb_factor-1:
                input_dim = inner_dim
                output_dim = inner_dim
            else:
                input_dim = inner_dim
                output_dim = self.filters

            kernel = self.add_weight(shape=(input_dim, output_dim),
                                     initializer=self.kernel_initializer,
                                     name='kernel_{}'.format(i),
                                     regularizer=self.kernel_regularizer,
                                     constraint=self.kernel_constraint)

            self.kernels.append(kernel)

        if self.use_bias:
            self.bias = self.add_weight(shape=(self.filters,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None

        super(TFConv2DCustom, self).build(input_shape)  # Be sure to call this at the end

    def convolution(self, X):
        reconstructed_kernel = self.kernels[0]
        for i in range(1, self.nb_factor):
            reconstructed_kernel = K.dot(reconstructed_kernel, self.kernels[i])

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

    def compute_output_shape(self, input_shape):
        return self._compute_output_shape(input_shape, self.kernel_shape, self.padding_height, self.padding_width, self.strides_height, self.strides_width)

    def get_prunable_weights(self):
        # DOn't prune bias because that usually harms model accuracy too much.
        return self.kernels