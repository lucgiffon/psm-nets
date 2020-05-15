from abc import abstractmethod, ABCMeta
import numpy as np

from palmnet.core.layer_replacer import LayerReplacer
from palmnet.data import Cifar100
from palmnet.layers.fastfood_layer_conv import FastFoodLayerConv
from palmnet.layers.fastfood_layer_dense import FastFoodLayerDense
from palmnet.layers.tt_layer_conv import TTLayerConv
from palmnet.layers.tt_layer_dense import TTLayerDense
from collections import defaultdict
from keras.layers import Dense, Conv2D

from palmnet.utils import build_dct_tt_ranks


class LayerReplacerDeepFried(LayerReplacer):
    def __init__(self, nb_stack=None, *args, **kwargs):
        self.nb_stack = nb_stack
        super().__init__(*args, **kwargs)

    ##################################
    # LayerReplacer abstract methods #
    ##################################
    def _apply_replacement(self, layer):
        dct_replacement = dict()
        if isinstance(layer, Conv2D) or isinstance(layer, Dense):
            dct_replacement["nb_stack"] = self.nb_stack
        else:
            dct_replacement = None
        return dct_replacement

    def _replace_conv2D(self, layer, dct_compression):
        nb_filters = layer.filters
        strides = layer.strides
        kernel_size = layer.kernel_size
        activation = layer.activation
        padding = layer.padding
        use_bias = layer.use_bias

        replacing_layer = FastFoodLayerConv(filters=nb_filters, use_bias=use_bias,
                                            kernel_size=kernel_size, strides=strides,
                                            padding=padding, activation=activation)
        replacing_weights = None

        return replacing_layer, replacing_weights, True

    def _replace_dense(self, layer, dct_compression):
        hidden_layer_dim = layer.units
        activation = layer.activation
        use_bias = layer.use_bias
        nb_stack = dct_compression["nb_stack"]

        replacing_layer = FastFoodLayerDense(nbr_stack=nb_stack, nb_units=hidden_layer_dim, use_bias=use_bias, activation=activation)
        replacing_weights = None

        return replacing_layer, replacing_weights, True

    def _set_weights_to_layer(self, replacing_layer, replacing_weights):
        if replacing_weights is None:
            return
        else:
            raise NotImplementedError("No weight expected")
