'''
Implementation of the low rank dense layer proposed in LOW-RANK MATRIX FACTORIZATION FOR DEEP NEURAL NETWORK
TRAINING WITH HIGH-DIMENSIONAL OUTPUT TARGETS .
'''
from keras import backend as K, activations, initializers, regularizers, constraints
from keras.engine.topology import Layer
from keras.layers import Dense


class LowRankDense(Layer):

    def __init__(self,
                 units, rank,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):

        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)

        super(LowRankDense, self).__init__(**kwargs)

        self.activation = activations.get(activation)
        self.use_bias = use_bias

        self.units = units
        self.rank = rank

        self.activation = activations.get(activation)
        self.use_bias = use_bias

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

    def build(self, input_shape):
        self.dense_in = Dense(self.rank, use_bias=False, kernel_initializer=self.kernel_initializer,
                              kernel_constraint=self.kernel_constraint, activation=None)
        self.dense_in.build(input_shape)

        self.dense_out = Dense(self.units, use_bias=self.use_bias, kernel_initializer=self.kernel_initializer,
                               kernel_constraint=self.kernel_constraint, activation=self.activation)
        self.dense_out.build(self.dense_in.compute_output_shape(input_shape))

        super().build(input_shape)

    def call(self, input):
        return self.dense_out(self.dense_in(input))

    def compute_output_shape(self, input_shape):
        return self.dense_out.compute_output_shape(self.dense_in.compute_output_shape((input_shape)))

    def get_config(self):
        super_config = super().get_config()
        super_config.update({
            'units': self.units,
            'rank': self.rank,
            'activation': activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint)
        })
        return super_config
