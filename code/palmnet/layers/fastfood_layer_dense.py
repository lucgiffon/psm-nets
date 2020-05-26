import keras
import numpy as np
import keras.backend as K
from keras import initializers, activations

from palmnet.layers.utils import G_variable, B_variable, P_variable, H_variable, S_variable
from skluc.utils.datautils import dimensionality_constraints
import tensorflow as tf


class FastFoodLayerDense(keras.layers.Layer):
    def __init__(self, nbr_stack=None, seed=None, nb_units=None, use_bias=True, activation=None, bias_initializer="zeros", sigma=1., cos_sin_act=False, trainable=True, **kwargs):
        super().__init__(**kwargs)
        self.sigma = sigma

        if seed is None:
            seed = np.random.randint(2**16)
        self.random_state = np.random.RandomState(seed)
        self.seed = seed

        self.nbr_stack = nbr_stack
        self.nb_units = nb_units

        assert not (self.nbr_stack is None and self.nb_units is None), "nbr_stack or nb_units should be set but both are None"

        self.trainable = trainable
        self.cos_sin_act = cos_sin_act

        self.use_bias = use_bias
        self.activation = activations.get(activation)
        self.bias_initializer = initializers.get(bias_initializer)

        self.init_dim = None
        self.final_dim = None

        super().__init__(**kwargs)

    def build(self, input_shape):
        with K.name_scope("fastfood" + "_sigma-" + str(self.sigma)):
            self.init_dim = np.prod([s for s in input_shape if s is not None])
            self.final_dim = int(dimensionality_constraints(self.init_dim))
            self.num_outputs = None

            if self.nbr_stack is not None:
                nbr_stack = self.nbr_stack
            else:
                nbr_stack = int(np.ceil(self.nb_units / self.final_dim))
                if self.nbr_stack is not None:
                    assert self.nbr_stack == nbr_stack
                else:
                    self.nbr_stack = nbr_stack

            B = B_variable((nbr_stack, self.final_dim))
            self.B = self.add_weight(
                name="B",
                shape=(nbr_stack, self.final_dim),
                initializer=lambda *args, **kwargs: B,
                trainable=self.trainable
            )

            G, G_norm = G_variable((nbr_stack, self.final_dim))
            G = np.reshape(G, (nbr_stack* self.final_dim,))  # this reshape to simplify subsequent calculations
            self.G = self.add_weight(
                name="G",
                shape=(nbr_stack* self.final_dim,),
                initializer=lambda *args, **kwargs: G,
                trainable=self.trainable
            )

            H = H_variable(self.final_dim)  # constant actually
            # self.H = self.add_weight(
            #     name="H",
            #     shape=(self.final_dim, self.final_dim),
            #     initializer=lambda *args, **kwargs: H,
            #     trainable=False
            # )
            self.H = H

            P = P_variable(self.final_dim, nbr_stack, self.random_state)  # constant also
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

            if self.nb_units is not None:
                dim_S = self.nb_units
            else:
                dim_S = nbr_stack * self.final_dim
            S = np.reshape(S, (nbr_stack * self.final_dim,))[:dim_S] # this reshape to simplify subsequent calculations
            self.S = self.add_weight(
                name="S",
                shape=(dim_S,),
                initializer=lambda *args, **kwargs: S,
                trainable=self.trainable
            )

        if self.nb_units is None:
            self.num_outputs = self.final_dim * nbr_stack
        else:
            self.num_outputs = self.nb_units

        if self.cos_sin_act:
            self.num_outputs *= 2

        if self.use_bias:
            self.bias = self.add_weight(name="bias", shape=(self.num_outputs,), initializer=self.bias_initializer, trainable=True)
        else:
            self.bias = None

        super().build(input_shape)

    def call(self, input, **kwargs):
        padding = self.final_dim - self.init_dim
        conv_out2 = K.reshape(input, [-1, self.init_dim])
        paddings = K.constant([[0, 0], [0, padding]], dtype=np.int32)
        conv_out2 = tf.pad(conv_out2, paddings, "CONSTANT")
        conv_out2 = K.reshape(conv_out2, (1, -1, 1, self.final_dim))
        h_ff1 = conv_out2 * self.B
        h_ff1 = K.reshape(h_ff1, (-1, self.final_dim))
        h_ff2 = K.dot(h_ff1, self.H)
        h_ff2 = K.reshape(h_ff2, (-1, self.final_dim * self.nbr_stack))
        h_ff3 = tf.gather(h_ff2, self.P, axis=1)
        # h_ff3 = K.dot(h_ff2, self.P)
        h_ff4 = K.reshape(h_ff3, (-1, self.final_dim * self.nbr_stack)) * self.G  # all the diagonals are represented as a single vector with concatenated diagonals
        h_ff4 = K.reshape(h_ff4, (-1, self.final_dim))
        h_ff5 = K.dot(h_ff4, self.H)

        h_ff5 = K.reshape(h_ff5, (-1, self.final_dim * self.nbr_stack))

        if self.nb_units is not None:
            h_ff5 = h_ff5[:, :self.nb_units]  # prevent useless computation

        h_ff6 = (1 / (self.sigma * np.sqrt(self.final_dim))) * h_ff5 * self.S  # all the diagonals are represented as a single vector with concatenated diagonals
        # h_ff6 = (1 / (self.sigma * np.sqrt(self.final_dim))) * h_ff5 * K.reshape(self.S, (-1, self.final_dim * self.nbr_stack))
        if self.cos_sin_act:
            h_ff7_1 = K.cos(h_ff6)
            h_ff7_2 = K.sin(h_ff6)
            h_ff7 = np.sqrt(float(1 / self.final_dim)) * K.concatenate([h_ff7_1, h_ff7_2], axis=1)
        else:
            h_ff7 = h_ff6

        if self.use_bias:
            out = K.bias_add(h_ff7, self.bias)
        else:
            out = h_ff7

        if self.activation is not None:
            out = self.activation(out)

        return out

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.num_outputs)

    def get_config(self):
        base_config = super().get_config()
        base_config.update({
            "use_bias": self.use_bias,
            "nbr_stack": self.nbr_stack,
            "nb_units": self.nb_units,
            'trainable': self.trainable,
            'activation': activations.serialize(self.activation),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'sigma': self.sigma,
            "cos_sin_act": self.cos_sin_act,
            'seed': self.seed
        })
        return base_config