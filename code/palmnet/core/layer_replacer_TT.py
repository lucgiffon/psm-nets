from abc import abstractmethod, ABCMeta
import numpy as np

from palmnet.core.layer_replacer import LayerReplacer
from palmnet.core.palminize import Palminizer, Palminizable
from palmnet.data import Cifar100
from keras.models import Model, Sequential
from keras.layers import InputLayer
from palmnet.layers.sparse_masked import SparseFactorisationDense, SparseFactorisationConv2DDensify
from palmnet.layers.tt_layer_conv import TTLayerConv
from palmnet.layers.tt_layer_dense import TTLayerDense
from palmnet.utils import get_sparsity_pattern, get_idx_last_dense_layer
from skluc.utils import log_memory_usage, logger
from collections import defaultdict
from keras.layers import Dense, Conv2D


class LayerReplacerTT(LayerReplacer):

    ##################################
    # LayerReplacer abstract methods #
    ##################################
    def _replace_conv2D(self, layer, sparse_factorization):

        nb_filters = layer.filters
        strides = layer.strides
        kernel_size = layer.kernel_size
        activation = layer.activation
        padding = layer.padding

        tt_ranks = self.dct_name_compression[layer.name]["tt_ranks"]

        replacing_layer = TTLayerConv(nb_filters=nb_filters, mat_ranks=tt_ranks, window=kernel_size, strides=strides, padding=padding, activation=activation, mode="auto")
        replacing_weights = None

        return replacing_layer, replacing_weights, True

    def _replace_dense(self, layer, sparse_factorization):
        hidden_layer_dim = layer.units
        activation = layer.activation

        tt_ranks = self.dct_name_compression[layer.name]["tt_ranks"]
        replacing_layer = TTLayerDense(nb_units=hidden_layer_dim, mat_ranks=tt_ranks, activation=activation, mode="auto")
        replacing_weights = None

        return replacing_layer, replacing_weights, True

    def _set_weights_to_layer(self, replacing_layer, replacing_weights):
        if replacing_weights is None:
            return
        else:
            raise NotImplementedError("No weight expected")



if __name__ == "__main__":
    model1 = Sequential()
    old_layer =  Dense(10, input_shape=(10,))
    model1.add(old_layer)

    model2 = Sequential()
    new_layer = old_layer.__class__(**old_layer.get_config())
    model2.add(new_layer)
    new_layer.set_weights(old_layer.get_weights())

    assert (new_layer.get_weights()[0] == old_layer.get_weights()[0]).all()
    assert (new_layer.get_weights()[1] == old_layer.get_weights()[1]).all()


    exit()
    from pprint import pprint
    # base_model = Cifar10.load_model("cifar10_tensortrain_base")
    base_model = Cifar100.load_model("cifar100-resnet20")
    palminizer = Palminizer(sparsity_fac=2,
                            nb_factor=2,
                            nb_iter=2,
                            delta_threshold_palm=1e-6,
                            hierarchical=False,
                            fast_unstable_proj=True)

    palminizable = Palminizable(base_model, palminizer)
    palminizable.palminize()
    pprint(palminizable.sparsely_factorized_layers)
    keep_last_layer, only_mask, dct_name_facto = False, True, palminizable.sparsely_factorized_layers
    model_transformer = LayerReplacer(keep_last_layer, only_mask, dct_name_facto)
    new_model = model_transformer.fit_transform(base_model)
    for l in new_model.layers:
        layer_w = l.get_weights()
        print(l.name)
        pprint([w for w in layer_w if len(w.shape)>1])

