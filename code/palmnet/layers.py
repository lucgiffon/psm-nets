from abc import abstractmethod

from keras import backend as K, activations, initializers, regularizers, constraints
from keras.layers import Layer
from keras.utils import conv_utils

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

        assert [sparsity_patterns[i].shape[1] == sparsity_patterns[i+1].shape[0] for i in range(len(sparsity_patterns)-1)]
        assert sparsity_patterns[-1].shape[1] == units, "sparsity pattern last dim should be equal to output dim in {}".format(__class__.__name__)

        super(SparseFactorisationDense, self).__init__(**kwargs)

        self.sparsity_patterns = sparsity_patterns
        self.nb_factor = len(sparsity_patterns)


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


    def build(self, input_shape):
        assert len(input_shape) >= 2
        assert input_shape[-1] == self.sparsity_patterns[0].shape[0], "input shape should be equal to 1st dim of sparsity pattern in {}".format(__class__.__name__)

        self.scaling = self.add_weight(shape=(1,),
                                      initializer=self.scaler_initializer,
                                      name='scaler',
                                      regularizer=self.scaler_regularizer,
                                      constraint=self.scaler_constraint)

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

        output = self.scaling * output

        if self.use_bias:
            output = K.bias_add(output, self.bias, data_format='channels_last')
        if self.activation is not None:
            output = self.activation(output)
        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.units)


class Conv2DCustom(Layer):
    """
    Custom convolution base, abstract, class.

    Create your own conv2D class by rewritting the `convolution` method.
    """
    def __init__(self,
                 filters,
                 kernel_size,
                 strides=1,
                 padding='valid',
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
        Same interface as keras.Conv2D.

        :param filters:
        :param kernel_size:
        :param strides:
        :param padding:
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
        super(Conv2DCustom, self).__init__(**kwargs)

        self.filters = filters
        self.kernel_size = conv_utils.normalize_tuple(kernel_size, 2, 'kernel_size')
        self.strides = conv_utils.normalize_tuple(strides, 2, 'strides')
        self.padding = conv_utils.normalize_padding(padding)

        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        self.init_padding_and_stride() # define padding height/width and stride height/width

    def init_padding_and_stride(self):
        if self.padding == "same":
            # synonym for "half": the output height and width are the same than the input
            assert self.kernel_size[0] % 2 == 1 and self.kernel_size[1] % 2 == 1, "When padding is same, kernel size must have odd dimensions: {}".format(self.kernel_size)
            self.padding_height =self.kernel_size[0] // 2
            self.padding_width = self.kernel_size[1] // 2
        elif self.padding == "valid":
            # no padding: convolution filters are not applied on edges so the output will have lower height and width than input
            self.padding_height = 0
            self.padding_width = 0
        else:
            raise NotImplementedError("Unknown padding: {}".format(self.padding))

        self.strides_height = self.strides[0]
        self.strides_width = self.strides[1]

    @staticmethod
    def imagette_flatten(X, window_h, window_w, window_c, out_h, out_w, stride=(1, 1), padding=(0, 0)):
        """
        From batch input images. For each image in batch: slice it in imagettes of height `imagette_h` and width `imagette_width`, and then stack them.

        The size of the output matrix depends on the padding and stride. But if padding is `same` and the stride is 1, the matrix is in `R^{HW \times hwc}`.

        :param X:
        :param window_h:
        :param window_w:
        :param window_c:
        :param out_h:
        :param out_w:
        :param stride:
        :param padding:
        :return:
        """
        padding_height = padding[0]
        padding_width = padding[1]
        stride_height = stride[0]
        stride_width = stride[1]

        X_padded = K.spatial_2d_padding(X, ((padding_height, padding_height), (padding_width, padding_width)))

        windows = []
        for y in range(out_h):
            for x in range(out_w):
                window = K.slice(X_padded, [0, y * stride_height, x * stride_width, 0], [-1, window_h, window_w, -1])
                windows.append(window)
        stacked = K.stack(windows)  # shape : [out_h, out_w, n, filter_h, filter_w, c]

        return K.reshape(stacked, [-1, window_c * window_w * window_h])

    def call(self, inputs):

        output = self.convolution(inputs)

        if self.activation is not None:
            output = self.activation(output)

        return output

    @abstractmethod
    def convolution(self, X):
        pass

    @abstractmethod
    def build(self, input_shape):
        pass

    @staticmethod
    def _compute_output_shape(input_shape, kernel_shape, padding_height, padding_width, stride_height, stride_width):
        sample_size, input_height, input_width, _ = input_shape
        filter_height, filter_width, _, filter_nbr = kernel_shape

        output_height = (input_height + 2 * padding_height - filter_height) // stride_height + 1
        output_width = (input_width + 2 * padding_width - filter_width) // stride_width + 1

        return sample_size, output_height, output_width, filter_nbr

    @abstractmethod
    def compute_output_shape(self, input_shape):
        pass


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


class DenseConv2D(Conv2DCustom):
    """
    Implementation of Conv2DCustom that implements a standard convolution with dense kernel matrix.

    """
    def build(self, input_shape):
        if input_shape[-1] is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')

        input_dim = input_shape[-1]
        self.kernel_shape = self.kernel_size + (input_dim, self.filters) # h x w x channels_in x channels_out

        self.kernel = self.add_weight(shape=self.kernel_shape,
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
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
        """

        :param X: The input 3D tensor.
        :return:
        """

        X_shape = [d.value for d in X.get_shape()]
        W_shape = self.kernel_shape
        sample_size, input_height, input_width, nb_in_channels = X_shape
        filter_height, filter_width, filter_in_channels, filter_nbr = W_shape

        _, output_height, output_width, _ = self._compute_output_shape(X_shape, W_shape, self.padding_height, self.padding_width, self.strides_height, self.strides_width)

        X_flat = self.imagette_flatten(X, filter_height, filter_width, nb_in_channels, output_height, output_width, (self.strides_height, self.strides_width), (self.padding_height, self.padding_width))
        W_flat = K.reshape(self.kernel, [filter_height*filter_width*nb_in_channels, filter_nbr])

        if self.use_bias:
            z = K.dot(X_flat, W_flat) + self.bias  # b: 1 X filter_n
        else:
            z = K.dot(X_flat, W_flat)

        return K.permute_dimensions(K.reshape(z, (output_height, output_width, -1 if sample_size is None else sample_size, filter_nbr)), [2, 0, 1, 3])

    def compute_output_shape(self, input_shape):
        return self._compute_output_shape(input_shape, [d.value for d in self.kernel.get_shape()], self.padding_height, self.padding_width, self.strides_height, self.strides_width)

