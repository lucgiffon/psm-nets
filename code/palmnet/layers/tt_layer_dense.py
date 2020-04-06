from keras import backend as K, activations, initializers
from keras.engine.topology import Layer
import numpy as np
import tensorflow as tf

from palmnet.utils import get_facto_for_channel_and_order, DCT_CHANNEL_PREDEFINED_FACTORIZATIONS

'''
Implementation of the paper 'Tensorizing Neural Networks', Alexander Novikov, Dmitry Podoprikhin, Anton Osokin, Dmitry P. Vetrov, NIPS, 2015
to compress a dense layer using Tensor Train factorization.
TTLayer compute y = Wx + b in the compressed form.
'''


class TTLayerDense(Layer):
    """ Given x\in\mathbb{R}^{N}, b\in\mathbb{R}^{M}, W\in\mathbb{R}^{M\times N}, y\in\mathbb{R}^{M}, compute y = Wx + b in the TT-format.

    Parameters:
        inp_modes: [n_1, n_2, ..., n_k] such that n_1*n_2*...*n_k=N
        out_modes: [m_1, m_2, ..., m_k] such that m_1*m_2*...m_k = M
        mat_ranks: [1, r_1, r_2, ..., r_k]

    """

    def __init__(self, nb_units, mat_ranks, inp_modes=None, out_modes=None, mode="auto", bias_initializer='zeros', kernel_initializer='glorot_normal', use_bias=True, activation=None, **kwargs):
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)

        self.mode = mode

        self.mat_ranks = np.array(mat_ranks).astype(int)
        self.order = len(self.mat_ranks) - 1
        self.nb_units = nb_units

        if self.mode == "auto":
            self.inp_modes = inp_modes
            self.out_modes = out_modes
        elif self.mode == "manual":
            if inp_modes is None or out_modes is None:
                raise ValueError("inp_modes and out_modes should be specified in mode manual.")
            self.inp_modes = np.array(inp_modes).astype(int)
            self.out_modes = np.array(out_modes).astype(int)
            self.num_dim = self.inp_modes.shape[0]

            if np.prod(self.out_modes) != self.nb_units:
                raise ValueError("out_modes product should equal to nb units: {} != {}".format(np.prod(self.out_modes), self.nb_units))
            if self.inp_modes.shape[0] != self.out_modes.shape[0]:
                raise ValueError("The number of input and output dimensions should be the same.")
            if self.order != self.out_modes.shape[0]:
                raise ValueError("Rank should have one more element than input/output shape")
            for r in self.mat_ranks:
                if isinstance(r, np.integer) != True:
                    raise ValueError("The rank should be an array of integer.")
        else:
            raise ValueError("Unknown mode {}".format(self.mode))

        super(TTLayerDense, self).__init__(**kwargs)

    def build(self, input_shape):
        inp_ch = input_shape[-1]
        if self.mode == "auto":
            self.inp_modes = get_facto_for_channel_and_order(inp_ch, self.order, dct_predefined_facto=DCT_CHANNEL_PREDEFINED_FACTORIZATIONS) if self.inp_modes is None else self.inp_modes
            self.out_modes = get_facto_for_channel_and_order(self.nb_units, self.order, dct_predefined_facto=DCT_CHANNEL_PREDEFINED_FACTORIZATIONS) if self.out_modes is None else self.out_modes

        assert np.prod(self.out_modes) == self.nb_units, "The product of out_modes should equal to the number of output units."
        assert np.prod(self.inp_modes) == inp_ch, "The product of inp_modes should equal to the input dimension."

        dim = self.order
        self.mat_cores = []
        for i in range(dim):
            self.mat_cores.append(
                self.add_weight(name='mat_core_%d' % (i + 1), shape=[self.out_modes[i] * self.mat_ranks[i + 1], self.mat_ranks[i] * self.inp_modes[i]], initializer=self.kernel_initializer, trainable=True))

        if self.use_bias:
            self.bias = self.add_weight(name="bias", shape=(np.prod(self.out_modes),), initializer=self.bias_initializer, trainable=True)

        super(TTLayerDense, self).build(input_shape)

    def call(self, input):
        dim = self.order
        out = tf.reshape(input, [-1, np.prod(self.inp_modes)])
        out = tf.transpose(out, [1, 0])
        for i in range(dim):
            out = tf.reshape(out, [self.mat_ranks[i] * self.inp_modes[i], -1])
            out = tf.matmul(self.mat_cores[i], out)
            out = tf.reshape(out, [self.out_modes[i], -1])
            out = tf.transpose(out, [1, 0])

        out = tf.reshape(out, [-1, np.prod(self.out_modes)])

        if self.use_bias:
            out = tf.add(out, self.bias, name='out')

        if self.activation is not None:
            out = self.activation(out)
        return out

    def compute_output_shape(self, input_shape):
        return (input_shape[0], np.prod(self.out_modes))

    def get_config(self):
        super_config = super().get_config()
        super_config.update({
            "inp_modes": self.inp_modes,
            "out_modes": self.out_modes,
            "mat_ranks": self.mat_ranks,
            "mode": self.mode,
        })
        return super_config
