from abc import abstractmethod, ABCMeta
import numpy as np

from palmnet.layers.multi_conv2D import MultiConv2D
from palmnet.layers.multi_dense import MultiDense
from skluc.utils import logger
from tensorly.decomposition import matrix_product_state

from palmnet.core.layer_replacer import LayerReplacer
from palmnet.data import Cifar100
from palmnet.layers.tt_layer_conv import TTLayerConv
from palmnet.layers.tt_layer_dense import TTLayerDense
from collections import defaultdict
# from keras.layers import Dense, Conv2D

from palmnet.utils import build_dct_tt_ranks, get_facto_for_channel_and_order, DCT_CHANNEL_PREDEFINED_FACTORIZATIONS, TensortrainBadRankException
from tensorflow_model_optimization.sparsity import keras as sparsity


class LayerReplacerMulti(LayerReplacer):
    def __init__(self, nb_factors, final_sparsity=0, end_step=None, frequency=100, init_sparsity=0.5, begin_step=0, *args, **kwargs):
        self.nb_factors = nb_factors

        self.final_sparsity = final_sparsity
        if self.final_sparsity != 0:
            self.pruning_params = {
                'pruning_schedule': sparsity.PolynomialDecay(initial_sparsity=init_sparsity,
                                                             final_sparsity=final_sparsity,
                                                             begin_step=begin_step,
                                                             end_step=end_step,
                                                             frequency=frequency)
            }

        super().__init__(*args, **kwargs)


    ##################################
    # LayerReplacer abstract methods #
    ##################################
    def _apply_replacement(self, layer):
        if isinstance(layer, self.keras_module.layers.Conv2D) or isinstance(layer, self.keras_module.layers.Dense):
            dct_replacement = dict()
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

        replacing_layer = MultiConv2D(nb_factors=self.nb_factors, filters=nb_filters, use_bias=use_bias, kernel_size=kernel_size, strides=strides, padding=padding, activation=activation)
        if self.final_sparsity != 0:
            replacing_layer = sparsity.prune_low_magnitude(replacing_layer, **self.pruning_params)
        replacing_weights = None

        return replacing_layer, replacing_weights, True


    def _replace_dense(self, layer, dct_compression):
        hidden_layer_dim = layer.units
        activation = layer.activation
        use_bias = layer.use_bias

        replacing_layer = MultiDense(units=hidden_layer_dim, nb_factors=self.nb_factors, use_bias=use_bias, activation=activation)
        if self.final_sparsity != 0:
            replacing_layer = sparsity.prune_low_magnitude(replacing_layer, **self.pruning_params)
        replacing_weights = None

        return replacing_layer, replacing_weights, True

    def _set_weights_to_layer(self, replacing_layer, replacing_weights):
        if replacing_weights is not None:
            raise ValueError("Shouldn't have any weight for replacement.")
        else:
            return


