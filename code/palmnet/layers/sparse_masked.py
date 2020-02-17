from keras import backend as K, activations, initializers, regularizers, constraints
from keras.layers import Layer

from palmnet.layers import Conv2DCustom
from palmnet.layers.sparse_tensor import cast_sparsity_pattern
import tensorflow as tf

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

        input_dim = input_shape[-1]

        self.kernel = self.add_weight(shape=(input_dim, self.units),
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.units,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None

        self.sparsity_mask_tensor = K.constant(self.sparsity_pattern, dtype="float32", name="sparsity_mask")

        # below is an other way of implementing
        # self.sparsity_mask_tensor = self.add_weight(
        #     name="sparsity_mask",
        #     shape=(input_shape[1], self.output_dim),
        #     initializer=self._sparsity_initializer_factory(),
        #     trainable=False
        # )

        super(SparseFixed, self).build(input_shape)  # Be sure to call this at the end

    def call(self, inputs):
        # multiply by the constant mask tensor so that gradient is 0 for zero entries.
        output = K.dot(inputs, self.kernel * self.sparsity_mask_tensor)

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
            'use_scaling': self.use_scaling,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint),
            'sparsity_patterns': self.sparsity_patterns,
            'factors_trainable': self.factors_trainable
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
        for i in range(self.nb_factor):
            input_dim, output_dim = self.sparsity_patterns[i].shape
            trainable = self.factors_trainable[i] if self.factors_trainable is not None else True

            self.kernels.append(self.add_weight(shape=(input_dim, output_dim),
                                                initializer=self.kernel_initializer,
                                                name='kernel_{}'.format(i),
                                                regularizer=self.kernel_regularizer,
                                                constraint=self.kernel_constraint,
                                                trainable=trainable))

            self.sparsity_masks.append(K.constant(self.sparsity_patterns[i], dtype="float32", name="sparsity_mask_{}".format(i)))



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

        output = self.scaling * output if self.use_scaling else output

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


        self.sparsity_patterns = sparsity_patterns
        self.nb_factor = len(sparsity_patterns)

        self.scaler_initializer = initializers.get(scaler_initializer)
        self.scaler_regularizer = regularizers.get(scaler_regularizer)
        self.scaler_constraint = constraints.get(scaler_constraint)

        assert [sparsity_patterns[i].shape[1] == sparsity_patterns[i+1].shape[0] for i in range(len(sparsity_patterns)-1)]
        assert sparsity_patterns[-1].shape[1] == self.filters, "sparsity pattern last dim should be equal to the number of filters in {}".format(__class__.__name__)

    def build(self, input_shape):
        if input_shape[-1] is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')

        self.scaling = self.add_weight(shape=(1,),
                                       initializer=self.scaler_initializer,
                                       name='scaler',
                                       regularizer=self.scaler_regularizer,
                                       constraint=self.scaler_constraint)

        input_dim = input_shape[-1]
        self.kernel_shape = self.kernel_size + (input_dim, self.filters) # h x w x channels_in x channels_out

        self.kernels = []
        self.sparsity_masks = []

        for i in range(self.nb_factor):
            input_dim, output_dim = self.sparsity_patterns[i].shape

            self.kernels.append(self.add_weight(shape=(input_dim, output_dim),
                                                initializer=self.kernel_initializer,
                                                name='kernel_{}'.format(i),
                                                regularizer=self.kernel_regularizer,
                                                constraint=self.kernel_constraint))

            self.sparsity_masks.append(K.constant(self.sparsity_patterns[i], dtype="float32", name="sparsity_mask_{}".format(i)))

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

        X_flat = self.imagette_flatten(X, filter_height, filter_width, nb_in_channels, output_height, output_width, (self.strides_height, self.strides_width), (self.padding_height, self.padding_width))

        output = X_flat
        for i in range(self.nb_factor):
            # multiply by the constant mask tensor so that gradient is 0 for zero entries.
            output = K.dot(output, self.kernels[i] * self.sparsity_masks[i])

        output = self.scaling * output

        if self.use_bias:
            output += self.bias

        return K.permute_dimensions(K.reshape(output, (output_height, output_width, -1 if sample_size is None else sample_size, filter_nbr)), [2, 0, 1, 3])

    def compute_output_shape(self, input_shape):
        return self._compute_output_shape(input_shape, self.kernel_shape, self.padding_height, self.padding_width, self.strides_height, self.strides_width)


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

        reconstructed_kernel = self.scaling * tf.reshape(reconstructed_kernel, (*self.kernel_size, X.get_shape()[-1].value, self.filters))

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
