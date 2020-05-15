import keras
import numpy as np
import keras.backend as K
from keras import initializers, activations

from palmnet.layers import Conv2DCustom
from palmnet.layers.utils import G_variable, B_variable, P_variable, H_variable, S_variable
from skluc.utils.datautils import dimensionality_constraints
import tensorflow as tf


class FastFoodLayerConv(Conv2DCustom):
    def __init__(self, sigma=1., **kwargs):
        self.sigma = sigma

        self.init_dim = None
        self.final_dim = None

        super().__init__(**kwargs)

    def build(self, input_shape):
        inp_shape = input_shape[1:]
        inp_h, inp_w, inp_ch = inp_shape[0:3]

        with K.name_scope("conv_fastfood" + "_sigma-" + str(self.sigma)):
            self.init_dim = np.prod(self.kernel_size) * inp_ch
            self.final_dim = int(dimensionality_constraints(self.init_dim))
            self.num_outputs = None

            nbr_stack = int(np.ceil(self.filters / self.final_dim))
            self.nbr_stack = nbr_stack

            B = B_variable((nbr_stack, self.init_dim))
            self.B = self.add_weight(
                name="B",
                shape=(nbr_stack, self.init_dim),
                initializer=lambda *args, **kwargs: B,
                trainable=True
            )

            G, G_norm = G_variable((nbr_stack, self.final_dim))
            G = np.reshape(G, (nbr_stack* self.final_dim,))  # this reshape to simplify subsequent calculations
            self.G = self.add_weight(
                name="G",
                shape=(nbr_stack* self.final_dim,),
                initializer=lambda *args, **kwargs: G,
                trainable=True
            )

            H = H_variable(self.final_dim)  # constant actually
            # self.H = self.add_weight(
            #     name="H",
            #     shape=(self.final_dim, self.final_dim),
            #     initializer=lambda *args, **kwargs: H,
            #     trainable=False
            # )
            self.H = H

            P = P_variable(self.final_dim, nbr_stack)  # constant also
            self.P = P
            # self.P = self.add_weight(
            #     name="P",
            #     shape=(self.final_dim * nbr_stack, self.final_dim * nbr_stack),
            #     initializer=lambda *args, **kwargs: P,
            #     trainable=False
            # )

            S = S_variable((nbr_stack, self.final_dim), G_norm)

            # self.S = self.add_weight(
            #     name="S",
            #     shape=(nbr_stack, self.final_dim),
            #     initializer=lambda *args, **kwargs: S,
            #     trainable=self.trainable
            # )

            dim_S = self.filters
            S = np.reshape(S, (nbr_stack * self.final_dim,))[:dim_S] # this reshape to simplify subsequent calculations
            self.S = self.add_weight(
                name="S",
                shape=(dim_S,),
                initializer=lambda *args, **kwargs: S,
                trainable=True
            )

        self.num_outputs = self.filters
        input_dim = input_shape[-1]
        self.kernel_shape = self.kernel_size + (input_dim, self.filters)  # h x w x channels_in x channels_out

        if self.use_bias:
            self.bias = self.add_weight(name="bias", shape=(self.num_outputs,), initializer=self.bias_initializer, trainable=True)
        else:
            self.bias = None

        super().build(input_shape)

    def convolution(self, input):

        # this is cool shit
        # reconstruct the full kernel from the fastfood decomposition
        reconstructed_kernel = K.reshape(tf.einsum("ij,kj->kij", self.H[:, :self.init_dim], self.B), (self.nbr_stack*self.final_dim, self.init_dim))  # sd x hwc, c nbr input channel
        reconstructed_kernel = K.dot(self.P, reconstructed_kernel)  # sd x hwc
        reconstructed_kernel = K.reshape(self.G, (-1, 1)) * reconstructed_kernel  # sd x hwc
        reconstructed_kernel = K.reshape(reconstructed_kernel, (self.nbr_stack, self.final_dim, self.init_dim))  # s x d x hwc
        reconstructed_kernel = K.reshape(tf.einsum("ij,kjm->kim", self.H, reconstructed_kernel), (self.nbr_stack*self.final_dim, self.init_dim))  # sd x hwc
        reconstructed_kernel = reconstructed_kernel[:self.filters, :]  # f x hwc , f nbr filter
        reconstructed_kernel = ((1 / (self.sigma * np.sqrt(self.final_dim))) * K.reshape(self.S, (-1, 1)) * reconstructed_kernel)  # f x hwc
        reconstructed_kernel = K.transpose(reconstructed_kernel)  # hwc x f
        reconstructed_kernel = K.reshape(reconstructed_kernel, self.kernel_shape)  # h x w x c x f

        output = K.conv2d(
            input,
            reconstructed_kernel,
            strides=self.strides,
            padding=self.padding)

        if self.use_bias:
            out = K.bias_add(output, self.bias)
        else:
            out = output

        return out

    def compute_output_shape(self, input_shape):
        return self._compute_output_shape(input_shape, self.kernel_shape, self.padding_height, self.padding_width, self.strides_height, self.strides_width)

    def get_config(self):
        base_config = super().get_config()
        base_config.update({
            'sigma': self.sigma,
        })
        return base_config