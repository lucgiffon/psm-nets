from abc import abstractmethod, ABCMeta

from tensorflow.keras import activations, initializers, regularizers, constraints, backend as K
from tensorflow.keras.layers import Layer
# from tensorflow.keras.engine import Layer
from keras.utils import conv_utils

import tensorflow as tf
from keras.layers import Conv2D

class TFConv2DCustom(Layer, metaclass=ABCMeta):
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
        super(TFConv2DCustom, self).__init__(**kwargs)

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

        self.init_padding_and_stride() # define padding height/width and strides height/width

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


    def get_config(self):
        base_config = super().get_config()

        config = {
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'padding': self.padding,
            'activation': activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint)
        }
        config.update(base_config)

        return config

    @staticmethod
    def imagette_flatten(X, window_h, window_w, window_c, out_h, out_w, stride=(1, 1), padding=(0, 0)):
        """
        From batch input images. For each image in batch: slice it in imagettes of height `imagette_h` and width `imagette_width`, and then stack them.

        The size of the output matrix depends on the padding and strides. But if padding is `same` and the strides is 1, the matrix is in `R^{HW \times hwc}`.

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
        super().build(input_shape)

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


