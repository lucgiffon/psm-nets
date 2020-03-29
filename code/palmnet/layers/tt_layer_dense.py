from keras import backend as K
from keras.engine.topology import Layer
import numpy as np
import tensorflow as tf

'''
Implementation of the paper 'Tensorizing Neural Networks', Alexander Novikov, Dmitry Podoprikhin, Anton Osokin, Dmitry P. Vetrov, NIPS, 2015
to compress a dense layer using Tensor Train factorization.
TTLayer compute y = Wx + b in the compressed form.
'''


class TTLayerDense(Layer):
    """ Given x\in\mathbb{R}^{N}, b\in\mathbb{R}^{M}, W\in\mathbb{R}^{M\times N}, y\in\mathbb{R}^{M}, compute y = Wx + b in the TT-format.

    Parameters:
        inp_modes(list): [n_1, n_2, ..., n_k] such that n_1*n_2*...*n_k=N
        out_modes(list): [m_1, m_2, ..., m_k] such that m_1*m_2*...m_k = M
        mat_ranks(list): [1, r_1, r_2, ..., r_k]

    """

    def __init__(self, inp_modes, out_modes, mat_ranks, **kwargs):
        self.inp_modes = np.array(inp_modes).astype(int)
        self.out_modes = np.array(out_modes).astype(int)
        self.mat_ranks = np.array(mat_ranks).astype(int)
        self.num_dim = self.inp_modes.shape[0]
        super(TTLayerDense, self).__init__(**kwargs)
        if self.inp_modes.shape[0] != self.out_modes.shape[0]:
            raise ValueError("The number of input and output dimensions should be the same.")
        if self.mat_ranks.shape[0] != self.out_modes.shape[0] + 1:
            raise ValueError("Rank should have one more element than input/output shape")
        for r in self.mat_ranks:
            if isinstance(r, np.integer) != True:
                raise ValueError("The rank should be an array of integer.")

    def build(self, input_shape):
        dim = self.num_dim
        self.mat_cores = []
        for i in range(dim):
            self.mat_cores.append(
                self.add_weight(name='mat_core_%d' % (i + 1), shape=[self.out_modes[i] * self.mat_ranks[i + 1], self.mat_ranks[i] * self.inp_modes[i]], initializer='glorot_normal', trainable=True))
        self.bias = self.add_weight(name="bias", shape=(np.prod(self.out_modes),), initializer='zeros', trainable=True)
        super(TTLayerDense, self).build(input_shape)

    def call(self, input):
        dim = self.num_dim
        out = tf.reshape(input, [-1, np.prod(self.inp_modes)])
        out = tf.transpose(out, [1, 0])
        for i in range(dim):
            out = tf.reshape(out, [self.mat_ranks[i] * self.inp_modes[i], -1])
            out = tf.matmul(self.mat_cores[i], out)
            out = tf.reshape(out, [self.out_modes[i], -1])
            out = tf.transpose(out, [1, 0])
        out = tf.add(tf.reshape(out, [-1, np.prod(self.out_modes)]), self.bias, name='out')
        return out

    def compute_output_shape(self, input_shape):
        return (input_shape[0], np.prod(self.out_modes))


