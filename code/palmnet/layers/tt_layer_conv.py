from keras import backend as K, activations, initializers
from keras.engine.topology import Layer
import numpy as np
import tensorflow as tf
from keras.utils import conv_utils

from palmnet.utils import get_facto_for_channel_and_order

'''
Implementation of the paper 'Ultimate tensorization: compressing convolutional and FC layers alike', Timur Garipov, Dmitry Podoprikhin, Alexander Novikov, Dmitry P. Vetrov, 2016
This layer performs a 2d convolution by decomposing the convolution kernel with Tensor Train method.
'''


class TTLayerConv(Layer):
    '''
    parameters:
        Parameters:
        inp_modes(list): [n_1, n_2, ..., n_k] such that n_1*n_2*...*n_k=N
        out_modes(list): [m_1, m_2, ..., m_k] such that m_1*m_2*...m_k = M
        mat_ranks(list): [1, r_1, r_2, ..., r_k, 1]


    '''

    def __init__(self, window, mat_ranks, nb_filters, inp_modes=None, out_modes=None, mode='auto', bias_initializer='zeros', kernel_initializer='glorot_normal', use_bias=True, activation=None, stride=(1, 1), padding='SAME', **kwargs):
        self.window = conv_utils.normalize_tuple(window, 2, 'kernel_size')
        self.stride = conv_utils.normalize_tuple(stride, 2, 'strides')
        self.padding = conv_utils.normalize_padding(padding)
        self.padding = self.padding.upper() if type(self.padding) is str else self.padding

        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        # self.kernel_regularizer = regularizers.get(kernel_regularizer)
        # self.bias_regularizer = regularizers.get(bias_regularizer)
        # self.kernel_constraint = constraints.get(kernel_constraint)
        # self.bias_constraint = constraints.get(bias_constraint)
        self.mode = mode

        self.mat_ranks = np.array(mat_ranks).astype(int)
        self.order = len(self.mat_ranks) - 1
        self.nb_filters = nb_filters

        if self.mode == "auto":
            self.inp_modes = inp_modes
            self.out_modes = out_modes
        elif self.mode == "manual":
            if inp_modes is None or out_modes is None:
                raise ValueError("inp_modes and out_modes should be specified in mode manual.")
            self.inp_modes = np.array(inp_modes).astype(int)
            self.out_modes = np.array(out_modes).astype(int)

            if np.prod(self.out_modes) != self.nb_filters:
                raise ValueError("out_modes product should equal to filters: {} != {}".format(np.prod(self.out_modes), self.nb_filters))
            if self.inp_modes.shape[0] != self.out_modes.shape[0]:
                raise ValueError("The number of input and output dimensions should be the same.")
            if self.order != self.out_modes.shape[0]:
                raise ValueError("Rank should have one more element than input/output shape")
            for r in self.mat_ranks:
                if isinstance(r, np.integer) != True:
                    raise ValueError("The rank should be an array of integer.")
        else:
            raise ValueError("Unknown mode {}".format(self.mode))
        super(TTLayerConv, self).__init__(**kwargs)




    def build(self, input_shape):
        inp_shape = input_shape[1:]
        inp_h, inp_w, inp_ch = inp_shape[0:3]

        if self.mode == "auto":
            self.inp_modes = get_facto_for_channel_and_order(inp_ch, self.order) if self.inp_modes is None else self.inp_modes
            self.out_modes = get_facto_for_channel_and_order(self.nb_filters, self.order) if self.out_modes is None else self.out_modes

        assert np.prod(self.out_modes) == self.nb_filters, "The product of out_modes should equal to the number of filters."
        assert np.prod(self.inp_modes) == inp_ch, "The product of inp_modes should equal to the number of channel in the last layer."

        filters_shape = [self.window[0], self.window[1], 1, self.mat_ranks[0]]
        self.filters = self.add_weight(name='filters', shape=filters_shape, initializer=self.kernel_initializer, trainable=True)

        out_ch = np.prod(self.out_modes)
        dim = self.order
        self.mat_cores = []
        for i in range(dim):
            self.mat_cores.append(
                self.add_weight(name='mat_core_%d' % (i + 1),
                                shape=[self.out_modes[i] * self.mat_ranks[i + 1], self.mat_ranks[i] * self.inp_modes[i]],
                                initializer=self.kernel_initializer,
                                trainable=True))

        if self.use_bias:
            self.bias = self.add_weight(name="bias", shape=(out_ch,), initializer=self.bias_initializer, trainable=True)

        super(TTLayerConv, self).build(input_shape)

    def call(self, input):
        inp_shape = input.get_shape().as_list()[1:]
        inp_h, inp_w, inp_ch = inp_shape[0:3]
        tmp = tf.reshape(input, [-1, inp_h, inp_w, inp_ch])
        tmp = tf.transpose(tmp, [0, 3, 1, 2])
        tmp = tf.reshape(tmp, [-1, inp_h, inp_w, 1])
        # [1] + self.stride + [1]
        tmp = tf.nn.conv2d(tmp, self.filters, (1, *self.stride, 1), self.padding)
        # tmp shape = [batch_size * inp_ch, h, w, r]
        h, w = tmp.get_shape().as_list()[1:3]
        tmp = tf.reshape(tmp, [-1, inp_ch, h, w, self.mat_ranks[0]])
        tmp = tf.transpose(tmp, [4, 1, 0, 2, 3])
        # tmp shape = [r, c, b, h, w]

        dim = self.order
        # out = tf.reshape(input, [-1, np.prod(self.inp_modes)])
        # out = tf.transpose(out, [1, 0])
        for i in range(dim):
            # print("STEP", i)
            tmp = tf.reshape(tmp, [self.mat_ranks[i] * self.inp_modes[i], -1])
            # print(out.shape)
            # print(self.mat_cores[i].shape)
            tmp = tf.matmul(self.mat_cores[i], tmp)
            tmp = tf.reshape(tmp, [self.out_modes[i], -1])
            tmp = tf.transpose(tmp, [1, 0])

        out = tf.reshape(tmp, [-1, h, w, np.prod(self.out_modes)])
        if self.use_bias:
            out = tf.add(out, self.bias, name='out')

        if self.activation is not None:
            out = self.activation(out)
        return out

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], input_shape[2], np.prod(self.out_modes))

    def get_config(self):
        super_config = super().get_config()
        super_config.update({
            "window": self.window,
            "stride": self.stride,
            "padding": self.padding,
            "inp_modes": self.inp_modes,
            "out_modes": self.out_modes,
            "mat_ranks": self.mat_ranks
        })
        return super_config
