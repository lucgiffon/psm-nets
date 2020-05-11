from keras import activations, initializers, regularizers, constraints, backend as K
from keras.engine import Layer
from keras.layers import BatchNormalization

from palmnet.utils import cast_sparsity_pattern, NAME_INIT_SPARSE_FACTO, sparse_facto_init


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
                 factors_trainable=None,
                 activation=None,
                 use_bias=True,
                 use_scaling=True,
                 scaler_initializer='glorot_uniform',
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 scaler_regularizer=None,
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 scaler_constraint=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 intertwine_batchnorm=False,
                 **kwargs):

        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)

        super(SparseFactorisationDense, self).__init__(**kwargs)

        if sparsity_patterns is not None:
            self.sparsity_patterns = [cast_sparsity_pattern(s) for s in sparsity_patterns]
            self.nb_factor = len(sparsity_patterns)

            assert [self.sparsity_patterns[i].shape[1] == self.sparsity_patterns[i + 1].shape[0] for i in range(len(self.sparsity_patterns) - 1)]
            assert self.sparsity_patterns[-1].shape[1] == units, "sparsity pattern last dim should be equal to output dim in {}".format(__class__.__name__)

        else:
            self.sparsity_patterns = None

        assert factors_trainable is None or all(type(elm) == bool for elm in factors_trainable)
        self.factors_trainable = factors_trainable

        self.units = units
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.use_scaling = use_scaling

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

        self.scaler_initializer = initializers.get(scaler_initializer)
        self.scaler_regularizer = regularizers.get(scaler_regularizer)
        self.scaler_constraint = constraints.get(scaler_constraint)

        self.intertwine_batchnorm = intertwine_batchnorm

    def get_config(self):
        base_config = super().get_config()
        config = {
            'units': self.units,
            'activation': activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'use_scaling': self.use_scaling,
            'kernel_initializer': initializers.serialize(self.kernel_initializer) if self.__kernel_initializer != NAME_INIT_SPARSE_FACTO else NAME_INIT_SPARSE_FACTO,
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint),
            'sparsity_patterns': self.sparsity_patterns,
            'factors_trainable': self.factors_trainable,
            'intertwine_batchnorm': self.intertwine_batchnorm
        }
        config.update(base_config)
        return config

    def build(self, input_shape):
        assert len(input_shape) >= 2
        if self.sparsity_patterns is not None:
            assert input_shape[-1] == self.sparsity_patterns[0].shape[0], "input shape should be equal to 1st dim of sparsity pattern in {}".format(__class__.__name__)
        else:
            raise ValueError("No sparsity pattern found.")

        if self.use_scaling:
            self.scaling = self.add_weight(shape=(1,),
                                          initializer=self.scaler_initializer,
                                          name='scaler',
                                          regularizer=self.scaler_regularizer,
                                          constraint=self.scaler_constraint)
        else:
            self.scaling = None

        self.kernels = []
        self.sparsity_masks = []
        # this list contains batchnorm function if asked or identity function
        if self.intertwine_batchnorm:
            self.batchnorms_or_id = []
        else:
            self.batchnorms_or_id = [lambda x: x] * self.nb_factor

        for i in range(self.nb_factor):
            input_dim, output_dim = self.sparsity_patterns[i].shape
            trainable = self.factors_trainable[i] if self.factors_trainable is not None else True

            if self.__kernel_initializer == NAME_INIT_SPARSE_FACTO:
                # matrix will be applied on right
                kernel_init = sparse_facto_init((input_dim, output_dim), i, self.sparsity_patterns[i], multiply_left=False)
            else:
                kernel_init = self.kernel_initializer

            self.kernels.append(self.add_weight(shape=(input_dim, output_dim),
                                                initializer=kernel_init,
                                                name='kernel_{}'.format(i),
                                                regularizer=self.kernel_regularizer,
                                                constraint=self.kernel_constraint,
                                                trainable=trainable))

            self.sparsity_masks.append(K.constant(self.sparsity_patterns[i], dtype="float32", name="sparsity_mask_{}".format(i)))
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

        super(SparseFactorisationDense, self).build(input_shape)  # Be sure to call this at the end

    def call(self, inputs):

        output = inputs
        for i in range(self.nb_factor):
            # multiply by the constant mask tensor so that gradient is 0 for zero entries.
            output = K.dot(output, self.kernels[i] * self.sparsity_masks[i])
            # batchnorm_or_id contains either a batchnormlayer or the identity function which has no effect
            output = self.batchnorms_or_id[i](output)

        if self.use_scaling:
            output = self.scaling* output

        if self.use_bias:
            output = K.bias_add(output, self.bias, data_format='channels_last')
        if self.activation is not None:
            output = self.activation(output)
        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.units)