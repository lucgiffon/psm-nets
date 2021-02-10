import tensorflow
from tensorflow.keras import activations, initializers, regularizers, constraints, backend as K
# from tensorflow.keras.engine import Layer
from tensorflow.keras.layers import Layer, BatchNormalization
import tensorflow_model_optimization as tfmot

from palmnet.utils import cast_sparsity_pattern, NAME_INIT_SPARSE_FACTO, sparse_facto_init
import numpy as np


class MultiDense(Layer, tfmot.sparsity.keras.PrunableLayer):
    """
    Layer which implements a sparse factorization with fixed sparsity pattern for all factors. The gradient will only be computed for non-zero entries.

    `SparseFactorisationDense` implements the operation:
    `output = activation(dot(input, prod([kernel[i] * sparsity_pattern[i] for i in range(nb_factor)]) + bias)`
    where `activation` is the element-wise activation function
    passed as the `activation` argument, `kernel` is a weights matrix
    created by the layer, `sparsity_pattern` is a mask matrix for the `kernel` and `bias` is a bias vector created by the layer
    (only applicable if `use_bias` is `True`).

    """


    def __init__(self, units,
                 nb_factors,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 intertwine_batchnorm=False,
                 **kwargs):

        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)

        super(MultiDense, self).__init__(**kwargs)

        self.nb_factor = nb_factors

        self.units = units
        self.activation = activations.get(activation)
        self.use_bias = use_bias

        if kernel_initializer != NAME_INIT_SPARSE_FACTO:
            self.kernel_initializer = initializers.get(kernel_initializer)
            self.__kernel_initializer = self.kernel_initializer
        else:
            self.kernel_initializer = lambda *args, **kwargs: None
            self.__kernel_initializer = kernel_initializer

        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)


        self.intertwine_batchnorm = intertwine_batchnorm

        self.image_max_size = -1

    def get_config(self):
        base_config = super().get_config()
        config = {
            'units': self.units,
            'activation': activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer) if self.__kernel_initializer != NAME_INIT_SPARSE_FACTO else NAME_INIT_SPARSE_FACTO,
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint),
            'nb_factors': self.nb_factor,
            'intertwine_batchnorm': self.intertwine_batchnorm
        }
        config.update(base_config)
        return config

    def build(self, input_shape):
        assert len(input_shape) >= 2

        self.kernels = []
        # this list contains batchnorm function if asked or identity function
        if self.intertwine_batchnorm:
            self.batchnorms_or_id = []
        else:
            self.batchnorms_or_id = [lambda x: x] * self.nb_factor

        inner_dim = min(self.units, input_shape[-1].value)
        for i in range(self.nb_factor):
            if i == 0:
                input_dim = input_shape[-1].value
                output_dim = inner_dim
            elif i < self.nb_factor-1:
                input_dim = inner_dim
                output_dim = inner_dim
            else:
                input_dim = inner_dim
                output_dim = self.units

            self.kernels.append(self.add_weight(shape=(input_dim, output_dim),
                                                initializer=self.kernel_initializer,
                                                name='kernel_{}'.format(i),
                                                regularizer=self.kernel_regularizer,
                                                constraint=self.kernel_constraint,
                                                trainable=True))

            if self.intertwine_batchnorm:
                batchnorm = BatchNormalization()
                batchnorm.build((-1, output_dim))
                self.batchnorms_or_id.append(batchnorm)


        if self.use_bias:
            self.bias = self.add_weight(shape=(self.units,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None

        super(MultiDense, self).build(input_shape)  # Be sure to call this at the end

    def call(self, inputs):

        output = inputs
        self.image_max_size = max(self.image_max_size, self.kernels[0].shape[0].value)
        for i in range(self.nb_factor):
            # multiply by the constant mask tensor so that gradient is 0 for zero entries.
            output = K.dot(output, self.kernels[i])
            # batchnorm_or_id contains either a batchnormlayer or the identity function which has no effect
            output = self.batchnorms_or_id[i](output)

        # self.image_max_size = max(self.image_max_size, np.prod([val.value for val in output.get_shape()[1:]]))

        if self.use_bias:
            output = K.bias_add(output, self.bias, data_format='channels_last')
        if self.activation is not None:
            output = self.activation(output)
        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.units)

    def get_prunable_weights(self):
        # DOn't prune bias because that usually harms model accuracy too much.
        return self.kernels