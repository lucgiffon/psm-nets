from abc import abstractmethod, ABCMeta
import numpy as np

from palmnet.core.layer_replacer import LayerReplacer
from palmnet.data import Cifar100
from palmnet.layers.fastfood_layer_conv import FastFoodLayerConv
from palmnet.layers.fastfood_layer_dense import FastFoodLayerDense
from palmnet.layers.sparse_conv2D_masked import SparseConv2D
from palmnet.layers.sparse_dense_masked import SparseDense
from palmnet.layers.tt_layer_conv import TTLayerConv
from palmnet.layers.tt_layer_dense import TTLayerDense
from collections import defaultdict
from keras.layers import Dense, Conv2D

from palmnet.layers.utils import proj_percent_greatest
from palmnet.utils import build_dct_tt_ranks, get_sparsity_pattern, NAME_INIT_SPARSE_FACTO
from tensorflow_model_optimization.sparsity import keras as sparsity
import tensorflow as tf

class LayerReplacerMagnitudePruning(LayerReplacer):
    def __init__(self, final_sparsity, end_step=None, frequency=100, init_sparsity=0.5, begin_step=0, hard=False, *args, **kwargs):
        self.hard = hard
        if self.hard is False:
            self.pruning_params = {
                'pruning_schedule': sparsity.PolynomialDecay(initial_sparsity=init_sparsity,
                                                       final_sparsity=final_sparsity,
                                                       begin_step=begin_step,
                                                       end_step=end_step,
                                                       frequency=frequency)
            }

        self.final_sparsity = final_sparsity
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

    def __replace_prune_low_magnitude(self, layer):
        if isinstance(layer, self.keras_module.layers.Conv2D):
            new_fresh_layer = tf.keras.layers.Conv2D(**layer.get_config())
        elif isinstance(layer, self.keras_module.layers.Dense):
            new_fresh_layer = tf.keras.layers.Dense(**layer.get_config())
        else:
            raise NotImplementedError
        replacing_layer = sparsity.prune_low_magnitude(new_fresh_layer, **self.pruning_params)

        replacing_weights = layer.get_weights()

        return replacing_layer, replacing_weights, True


    def __replace_prune_low_magnitude_hard(self, layer):
        if isinstance(layer, self.keras_module.layers.Conv2D):
            conv_kernel = layer.get_weights()[0]
            conv_matrix = np.reshape(conv_kernel, (np.prod(conv_kernel.shape[:3]),conv_kernel.shape[-1]))
            replacing_weights = proj_percent_greatest(conv_matrix, 1 - self.final_sparsity)
            sparsity_pattern = get_sparsity_pattern(replacing_weights)

            nb_filters = layer.filters
            strides = layer.strides
            kernel_size = layer.kernel_size
            activation = layer.activation
            padding = layer.padding
            regularizer = layer.kernel_regularizer

            new_fresh_layer = SparseConv2D(strides=strides, filters=nb_filters, kernel_size=kernel_size,
                                                        sparsity_pattern=sparsity_pattern, use_bias=layer.use_bias, activation=activation, padding=padding,
                                                        kernel_regularizer=regularizer, kernel_initializer=NAME_INIT_SPARSE_FACTO)

            replacing_weights = [replacing_weights] + ([layer.get_weights()[-1]] if layer.use_bias else [])

        elif isinstance(layer, self.keras_module.layers.Dense):
            replacing_weights = proj_percent_greatest(layer.get_weights()[0], 1 - self.final_sparsity)
            sparsity_pattern = get_sparsity_pattern(replacing_weights)

            hidden_layer_dim = layer.units
            activation = layer.activation
            regularizer = layer.kernel_regularizer

            new_fresh_layer = SparseDense(units=hidden_layer_dim, sparsity_pattern=sparsity_pattern, use_bias=layer.use_bias,
                                                       activation=activation, kernel_regularizer=regularizer, kernel_initializer=NAME_INIT_SPARSE_FACTO)
            replacing_weights = [replacing_weights] + [layer.get_weights()[-1]] if layer.use_bias else []
        else:
            raise NotImplementedError

        return new_fresh_layer, replacing_weights, True

    def _replace_conv2D(self, layer, dct_compression):
        if self.hard is False:
            return self.__replace_prune_low_magnitude(layer)
        else:
            return self.__replace_prune_low_magnitude_hard(layer)

    def _replace_dense(self, layer, dct_compression):
        if self.hard is False:
            return self.__replace_prune_low_magnitude(layer)
        else:
            return self.__replace_prune_low_magnitude_hard(layer)

    def _set_weights_to_layer(self, replacing_layer, replacing_weights):
        replacing_layer.set_weights(replacing_weights)
        
