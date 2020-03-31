from abc import abstractmethod, ABCMeta
import numpy as np

from palmnet.core.layer_replacer import LayerReplacer
from palmnet.data import Cifar100
from palmnet.layers.tt_layer_conv import TTLayerConv
from palmnet.layers.tt_layer_dense import TTLayerDense
from collections import defaultdict
from keras.layers import Dense, Conv2D

from palmnet.utils import build_dct_tt_ranks


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

        replacing_layer = TTLayerConv(nb_filters=nb_filters, mat_ranks=tt_ranks, window=kernel_size, stride=strides, padding=padding, activation=activation, mode="auto")
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

    from pprint import pprint
    # base_model = Cifar10.load_model("cifar10_tensortrain_base")
    base_model = Cifar100.load_model("cifar100_vgg19_2048x2048")

    dct_layer_params = build_dct_tt_ranks(base_model)

    keep_last_layer = True
    model_transformer = LayerReplacerTT(dct_name_compression=dct_layer_params, keep_last_layer=keep_last_layer, keep_first_layer=True)
    new_model = model_transformer.fit_transform(base_model)
    for l in new_model.layers:
        layer_w = l.get_weights()
        print(l.name, l.__class__.__name__)
        # pprint([w for w in layer_w if len(w.shape)>1])

