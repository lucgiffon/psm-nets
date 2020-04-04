from abc import abstractmethod, ABCMeta
import numpy as np
import tensorly
from tensorly.decomposition import partial_tucker

from palmnet import VBMF
from palmnet.core.layer_replacer import LayerReplacer
from palmnet.data import Cifar100
from palmnet.layers.tt_layer_conv import TTLayerConv
from palmnet.layers.tt_layer_dense import TTLayerDense
from collections import defaultdict
from keras.layers import Dense, Conv2D

from palmnet.layers.tucker_layer import TuckerLayerConv
from palmnet.utils import build_dct_tt_ranks


class LayerReplacerTucker(LayerReplacer):

    @staticmethod
    def get_rank_layer(weights):
        """ Unfold the 2 modes of the Tensor the decomposition will
        be performed on, and estimates the ranks of the matrices using VBMF.
        Taken from: https://jacobgil.github.io/deeplearning/tensor-decompositions-deep-learning
        """
        unfold_0 = tensorly.base.unfold(weights, 2) # channel in first
        unfold_1 = tensorly.base.unfold(weights, 3) # channel out
        _, diag_0, _, _ = VBMF.EVBMF(unfold_0)
        _, diag_1, _, _ = VBMF.EVBMF(unfold_1)
        ranks = (max(1, diag_0.shape[0]), max(1, diag_1.shape[1]))
        return ranks

    @staticmethod
    def get_tucker_decomposition(layer_weights, in_rank, out_rank):
        core, (first, last) = partial_tucker(layer_weights, modes=(2, 3), ranks=(in_rank, out_rank), init='svd', n_iter_max=500, tol=10e-10)
        return first, core, last

    ##################################
    # LayerReplacer abstract methods #
    ##################################
    def _apply_replacement(self, layer):
        dct_replacement = dict()
        if isinstance(layer, Conv2D):
            layer_weights = layer.get_weights()[0]  # h, w, c_in, c_out
            assert len(layer_weights.shape) == 4, "Shape of convolution kernel should be of size 4"
            assert layer.data_format == "channels_last", "filter dimension should be last"
            in_rank, out_rank = self.get_rank_layer(layer_weights)
            dct_replacement["in_rank"] = in_rank
            dct_replacement["out_rank"] = out_rank
            first, core, last = self.get_tucker_decomposition(layer_weights, in_rank, out_rank)
            first = first[np.newaxis, np.newaxis, :]
            last = last.T
            last = last[np.newaxis, np.newaxis, :]
            dct_replacement["first_conv_weights"] = first
            dct_replacement["core_conv_weights"] = core
            dct_replacement["last_conv_weights"] = last
        elif isinstance(layer, Dense):
            dct_replacement = None
        else:
            dct_replacement = None

        return dct_replacement

    def _replace_conv2D(self, layer, sparse_factorization):
        nb_filters = layer.filters
        strides = layer.strides
        kernel_size = layer.kernel_size
        activation = layer.activation
        padding = layer.padding
        kernel_regularizer = layer.kernel_regularizer
        bias_regularizer = layer.bias_regularizer

        in_rank = self.dct_name_compression[layer.name]["in_rank"]
        out_rank = self.dct_name_compression[layer.name]["out_rank"]

        replacing_layer = TuckerLayerConv(in_rank=in_rank, out_rank=out_rank, filters=nb_filters,
                                          kernel_size=kernel_size, strides=strides, padding=padding, activation=activation,
                                          kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer)

        replacing_weights = [self.dct_name_compression[layer.name]["first_conv_weights"]] \
            + [self.dct_name_compression[layer.name]["core_conv_weights"]] \
            + [self.dct_name_compression[layer.name]["last_conv_weights"]] \
            + [layer.get_weights()[-1]] if layer.use_bias else []

        return replacing_layer, replacing_weights, True

    def _replace_dense(self, layer, sparse_factorization):
        """Dense layers are not replaced by tucker decomposition"""
        return None, None, False

    def _set_weights_to_layer(self, replacing_layer, replacing_weights):
        replacing_layer.set_weights(replacing_weights)




# if __name__ == "__main__":
#
#     from pprint import pprint
#     # base_model = Cifar10.load_model("cifar10_tensortrain_base")
#     base_model = Cifar100.load_model("cifar100_vgg19_2048x2048")
#
#     # dct_layer_params = build_dct_tt_ranks(base_model)
#
#     keep_last_layer = True
#     model_transformer = LayerReplacerTT(rank_value=2, order=4, keep_last_layer=keep_last_layer, keep_first_layer=True)
#     new_model = model_transformer.fit_transform(base_model)
#     for l in new_model.layers:
#         layer_w = l.get_weights()
#         print(l.name, l.__class__.__name__)
#         # pprint([w for w in layer_w if len(w.shape)>1])
#
