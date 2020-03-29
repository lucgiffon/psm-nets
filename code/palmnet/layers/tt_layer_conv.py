from keras import backend as K
from keras.engine.topology import Layer
import numpy as np
import tensorflow as tf

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

    def __init__(self, window, inp_modes, out_modes, mat_ranks, stride=[1, 1], padding='SAME', **kwargs):
        self.window = window
        self.stride = stride
        self.padding = padding
        self.inp_modes = np.array(inp_modes).astype(int)
        self.out_modes = np.array(out_modes).astype(int)
        self.mat_ranks = np.array(mat_ranks).astype(int)
        self.num_dim = self.inp_modes.shape[0]
        super(TTLayerConv, self).__init__(**kwargs)
        if self.inp_modes.shape[0] != self.out_modes.shape[0]:
            raise ValueError("The number of input and output dimensions should be the same.")
        if self.mat_ranks.shape[0] != self.out_modes.shape[0] + 1:
            raise ValueError("Rank should have one more element than input/output shape")
        for r in self.mat_ranks:
            if isinstance(r, np.integer) != True:
                raise ValueError("The rank should be an array of integer.")

    def build(self, input_shape):
        inp_shape = input_shape[1:]
        inp_h, inp_w, inp_ch = inp_shape[0:3]
        filters_shape = [self.window[0], self.window[1], 1, self.mat_ranks[0]]
        self.filters = self.add_weight(name='filters', shape=filters_shape, initializer='glorot_normal', trainable=True)

        out_ch = np.prod(self.out_modes)
        dim = self.num_dim
        self.mat_cores = []
        for i in range(dim):
            self.mat_cores.append(
                self.add_weight(name='mat_core_%d' % (i + 1), shape=[self.out_modes[i] * self.mat_ranks[i + 1], self.mat_ranks[i] * self.inp_modes[i]], initializer='glorot_normal', trainable=True))
        self.bias = self.add_weight(name="bias", shape=(out_ch,), initializer='zeros', trainable=True)
        super(TTLayerConv, self).build(input_shape)

    def call(self, input):
        inp_shape = input.get_shape().as_list()[1:]
        inp_h, inp_w, inp_ch = inp_shape[0:3]
        tmp = tf.reshape(input, [-1, inp_h, inp_w, inp_ch])
        tmp = tf.transpose(tmp, [0, 3, 1, 2])
        tmp = tf.reshape(tmp, [-1, inp_h, inp_w, 1])
        tmp = tf.nn.conv2d(tmp, self.filters, [1] + self.stride + [1], self.padding)
        # tmp shape = [batch_size * inp_ch, h, w, r]
        h, w = tmp.get_shape().as_list()[1:3]
        tmp = tf.reshape(tmp, [-1, inp_ch, h, w, self.mat_ranks[0]])
        tmp = tf.transpose(tmp, [4, 1, 0, 2, 3])
        # tmp shape = [r, c, b, h, w]

        dim = self.num_dim
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
        out = tf.add(tf.reshape(tmp, [-1, h, w, np.prod(self.out_modes)]), self.bias, name='out')
        return out

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], input_shape[2], np.prod(self.out_modes))
