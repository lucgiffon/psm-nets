from abc import abstractmethod, ABCMeta
import numpy as np
import tensorly

from palmnet.core.layer_replacer_sparse_facto import LayerReplacerSparseFacto
from skluc.utils import logger
from tensorly.decomposition import partial_tucker

from palmnet import VBMF
from palmnet.core.layer_replacer import LayerReplacer
from palmnet.core.layer_replacer_tucker import LayerReplacerTucker
from palmnet.data import Cifar100
from palmnet.layers.tt_layer_conv import TTLayerConv
from palmnet.layers.tt_layer_dense import TTLayerDense
from collections import defaultdict
from keras.layers import Dense, Conv2D

from palmnet.layers.tucker_layer import TuckerLayerConv
from palmnet.layers.tucker_layer_sparse_facto import TuckerSparseFactoLayerConv
from palmnet.utils import build_dct_tt_ranks


class LayerReplacerSparseFactoTucker(LayerReplacer):
    lst_tucker_weights = [
        "first_conv_weights",
        "core_conv_weights",
        "last_conv_weights"
    ]

    def __init__(self, sparse_factorizer=None,  *args, **kwargs):
        self.sparse_factorizer = sparse_factorizer
        super().__init__(*args, **kwargs)

    ##################################
    # LayerReplacer abstract methods #
    ##################################
    def _apply_replacement(self, layer):

        if isinstance(layer, Conv2D):
            dct_tucker_replacement = LayerReplacerTucker.apply_tucker_or_low_rank_decomposition_to_layer(layer)

            dct_replacement = dict()
            dct_replacement["in_rank"] = dct_tucker_replacement["in_rank"]
            dct_replacement["out_rank"] = dct_tucker_replacement["out_rank"]

            for tucker_part_name in self.lst_tucker_weights:
                tucker_part_weights = dct_tucker_replacement[tucker_part_name]
                _lambda, op_sparse_factors, new_layer_weights = self.sparse_factorizer.factorize_conv2D_weights(tucker_part_weights)
                dct_tucker_part_sparse_facto = {
                    "lambda": _lambda,
                    "sparse_factors": op_sparse_factors,
                    "base_weights": tucker_part_weights
                }
                dct_replacement[tucker_part_name] = dct_tucker_part_sparse_facto

        elif isinstance(layer, Dense):
            _lambda, op_sparse_factors, new_layer_weights = self.sparse_factorizer.factorize_layer(layer, apply_weights=False)
            dct_replacement = {
                "lambda": _lambda,
                "sparse_factors": op_sparse_factors
            }
        else:
            dct_replacement = None

        return dct_replacement

    def _replace_conv2D(self, layer, dct_compression):
        nb_filters = layer.filters
        strides = layer.strides
        kernel_size = layer.kernel_size
        activation = layer.activation
        padding = layer.padding
        kernel_regularizer = layer.kernel_regularizer
        bias_regularizer = layer.bias_regularizer

        in_rank = dct_compression["in_rank"]
        out_rank = dct_compression["out_rank"]


        lst_sparsity_patterns_by_tucker_part = list()
        lst_scaling = list()
        lst_factor_data = list()
        for tucker_part_name in self.lst_tucker_weights:
            tucker_part_weights = dct_compression[tucker_part_name]
            scaling, factor_data, sparsity_patterns = self.sparse_factorizer.get_weights_from_sparse_facto(tucker_part_weights,return_scaling=True)
            lst_sparsity_patterns_by_tucker_part.append(sparsity_patterns)
            lst_factor_data.append(factor_data)
            lst_scaling.append(scaling)
        less_values_than_base_tucker, less_values_tucker_than_dense = self.check_facto_less_values_than_tucker_base(layer, lst_sparsity_patterns_by_tucker_part)

        lst_bias = [layer.get_weights()[-1]] if layer.use_bias else []

        if less_values_than_base_tucker:
            replacing_layer = TuckerSparseFactoLayerConv(lst_sparsity_patterns_by_tucker_part=lst_sparsity_patterns_by_tucker_part,
                                                         in_rank=in_rank, out_rank=out_rank, filters=nb_filters,
                                                         kernel_size=kernel_size, strides=strides, padding=padding, activation=activation,
                                                         kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer)
            replacing_weights = list()
            for it in range(len(lst_factor_data)):
                replacing_weights += lst_scaling[it]
                replacing_weights += lst_sparsity_patterns_by_tucker_part[it]
            replacing_weights += lst_bias

        elif less_values_tucker_than_dense:
            replacing_layer = TuckerLayerConv(in_rank=in_rank, out_rank=out_rank, filters=nb_filters,
                                              kernel_size=kernel_size, strides=strides, padding=padding, activation=activation,
                                              kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer)

            replacing_weights = [dct_compression["first_conv_weights"]["base_weights"]] \
                                + [dct_compression["core_conv_weights"]["base_weights"]] \
                                + [dct_compression["last_conv_weights"]["base_weights"]] \
                                + lst_bias
        else:
            replacing_layer, replacing_weights = None, None

        return replacing_layer, replacing_weights, less_values_than_base_tucker or less_values_tucker_than_dense

    def _replace_dense(self, layer, dct_compression):
        """Dense layers are not replaced by tucker decomposition"""
        return LayerReplacerSparseFacto.replace_dense_static(layer, dct_compression,
                                                             fct_get_weights_from_sparse_facto=self.sparse_factorizer.get_weights_from_sparse_facto,
                                                             use_scaling=True)

    def _set_weights_to_layer(self, replacing_layer, replacing_weights):
        replacing_layer.set_weights(replacing_weights)

    @staticmethod
    def check_facto_less_values_than_tucker_base(layer, lst_sparsity_patterns):
        """
        Check if there is actually less values in the compressed layer than base layer.

        :param layer:
        :param sparsity_patterns:
        :return:
        """
        layer_weights = layer.get_weights()
        nb_val_full_layer = np.sum(np.prod(w.shape) for w in layer_weights)

        nb_val_sparse_factors = 0
        nb_val_tucker = 0
        for sparsity_patterns in lst_sparsity_patterns:
            nb_val_sparse_factors += np.sum([np.sum(fac) for fac in sparsity_patterns])
            nb_val_tucker += np.prod((sparsity_patterns[0].shape[0], sparsity_patterns[-1].shape[-1]))
        nb_val_sparse_factors += len(lst_sparsity_patterns) # one scaling factor for each sparsity pattern

        less_value_base_than_tucker = nb_val_full_layer <= nb_val_tucker
        less_values_tucker_than_sparse_facto = nb_val_tucker <= nb_val_sparse_factors
        if less_value_base_than_tucker :
            logger.info("Less values in full matrix than tucker in layer {}. Keep full matrix. {} <= {}".format(layer.name, nb_val_full_layer, nb_val_tucker))
        elif less_values_tucker_than_sparse_facto:
            logger.info("Less values in tucker than sparse facto in layer {}. Keep tucker. {} <= {}".format(layer.name, nb_val_tucker, nb_val_sparse_factors))

        return not less_values_tucker_than_sparse_facto, not less_value_base_than_tucker

