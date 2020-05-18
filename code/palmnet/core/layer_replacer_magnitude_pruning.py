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
from tensorflow_model_optimization.sparsity import keras as sparsity
import tensorflow as tf

class LayerReplacerMagnitudePruning(LayerReplacer):
    def __init__(self,final_sparsity, end_step, frequency=100, init_sparsity=0.5, begin_step=0, *args, **kwargs):
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

    def _replace_conv2D(self, layer, dct_compression):
        return self.__replace_prune_low_magnitude(layer)

    def _replace_dense(self, layer, dct_compression):
        return self.__replace_prune_low_magnitude(layer)

    def _set_weights_to_layer(self, replacing_layer, replacing_weights):
        replacing_layer.set_weights(replacing_weights)
        
