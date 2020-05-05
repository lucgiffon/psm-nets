from abc import abstractmethod, ABCMeta
import numpy as np

from palmnet.core.layer_replacer import LayerReplacer
from palmnet.data import Cifar100
from palmnet.layers.fastfood_layer import FastFoodLayer
from palmnet.layers.tt_layer_conv import TTLayerConv
from palmnet.layers.tt_layer_dense import TTLayerDense
from collections import defaultdict
from keras.layers import Dense, Conv2D

from palmnet.utils import build_dct_tt_ranks


class LayerReplacerDeepFried(LayerReplacer):
    def __init__(self, nb_stack, *args, **kwargs):
        self.nb_stack = nb_stack
        super().__init__(*args, **kwargs)

    ##################################
    # LayerReplacer abstract methods #
    ##################################
    def _apply_replacement(self, layer):
        dct_replacement = dict()
        if isinstance(layer, Dense):
            dct_replacement["nb_stack"] = self.nb_stack
        else:
            dct_replacement = None

        return dct_replacement

    def _replace_conv2D(self, layer, dct_compression):
        return None, None, False

    def _replace_dense(self, layer, dct_compression):
        hidden_layer_dim = layer.units
        activation = layer.activation
        use_bias = layer.use_bias
        nb_stack = dct_compression["nb_stack"]

        replacing_layer = FastFoodLayer(nbr_stack=nb_stack, nb_units=hidden_layer_dim, use_bias=use_bias, activation=activation)
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

    # dct_layer_params = build_dct_tt_ranks(base_model)

    keep_last_layer = True
    model_transformer = LayerReplacerTT(rank_value=2, order=4, keep_last_layer=keep_last_layer, keep_first_layer=True)
    new_model = model_transformer.fit_transform(base_model)
    for l in new_model.layers:
        layer_w = l.get_weights()
        print(l.name, l.__class__.__name__)
        # pprint([w for w in layer_w if len(w.shape)>1])

