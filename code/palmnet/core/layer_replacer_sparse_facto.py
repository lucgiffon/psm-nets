from abc import abstractmethod, ABCMeta
import numpy as np

from palmnet.core.layer_replacer import LayerReplacer
from palmnet.core.palminizer import Palminizer
from palmnet.data import Cifar100
from palmnet.layers.sparse_masked import SparseFactorisationDense, SparseFactorisationConv2DDensify
from palmnet.utils import get_sparsity_pattern
from skluc.utils import logger


class LayerReplacerSparseFacto(LayerReplacer):
    def __init__(self, only_mask, sparse_factorizer=None,  *args, **kwargs):
        self.only_mask = only_mask
        self.sparse_factorizer = sparse_factorizer
        super().__init__(*args, **kwargs)

    @staticmethod
    @abstractmethod
    def _get_factors_from_op_sparsefacto(op_sparse_facto):
        pass

    ##################################
    # LayerReplacer abstract methods #
    ##################################
    def _apply_replacement(self, layer):
        _lambda, op_sparse_factors = self._get_facto(layer)
        dct_replacement = {
            "lambda": _lambda,
            "sparse_factors": op_sparse_factors
        }
        if _lambda is None and op_sparse_factors is None:
            return None
        else:
            return dct_replacement

    def _replace_conv2D(self, layer, sparse_factorization):
        scaling, factor_data, sparsity_patterns = self._get_weights_from_sparse_facto(sparse_factorization)
        less_values_than_base = self.__check_facto_less_values_than_base(layer, sparsity_patterns)

        if not less_values_than_base:
            replacing_weights = None
            replacing_layer = layer
        else:
            nb_filters = layer.filters
            strides = layer.strides
            kernel_size = layer.kernel_size
            activation = layer.activation
            padding = layer.padding
            regularizer = layer.kernel_regularizer
            replacing_layer = SparseFactorisationConv2DDensify(use_scaling=not self.only_mask, strides=strides, filters=nb_filters, kernel_size=kernel_size,
                                                               sparsity_patterns=sparsity_patterns, use_bias=layer.use_bias, activation=activation, padding=padding,
                                                               kernel_regularizer=regularizer)
            replacing_weights = scaling + factor_data + [layer.get_weights()[-1]] if layer.use_bias else []

        return replacing_layer, replacing_weights, less_values_than_base

    def _replace_dense(self, layer, sparse_factorization):
        scaling, factor_data, sparsity_patterns = self._get_weights_from_sparse_facto(sparse_factorization)

        less_values_than_base = self.__check_facto_less_values_than_base(layer, sparsity_patterns)

        if not less_values_than_base:
            replacing_weights = None
            replacing_layer = layer
        else:

            hidden_layer_dim = layer.units
            activation = layer.activation
            regularizer = layer.kernel_regularizer
            replacing_layer = SparseFactorisationDense(use_scaling=not self.only_mask, units=hidden_layer_dim, sparsity_patterns=sparsity_patterns, use_bias=layer.use_bias,
                                                       activation=activation, kernel_regularizer=regularizer)
            replacing_weights = scaling + factor_data + [layer.get_weights()[-1]] if layer.use_bias else []

        return replacing_layer, replacing_weights, less_values_than_base

    def _set_weights_to_layer(self, replacing_layer, replacing_weights):
        if self.only_mask:
            masked_weights = []
            i = 0
            for w in replacing_layer.get_weights():
                if len(w.shape) > 1:  # if not bias vector then apply sparsity
                    new_weight = w * get_sparsity_pattern(replacing_weights[i])
                    i += 1
                else:
                    new_weight = w # case bias
                masked_weights.append(new_weight)
            replacing_weights = masked_weights

        replacing_layer.set_weights(replacing_weights)


    ####################################
    # LayerReplacerSparseFacto methods #
    ####################################
    def _get_facto(self, layer):
        _lambda, op_sparse_factors, _ = self.sparse_factorizer.factorize_layer(layer, apply_weights=False)
        return _lambda, op_sparse_factors

    def _get_weights_from_sparse_facto(self, sparse_factorization):
        # backward compatibility
        if type(sparse_factorization) == tuple:
            sparse_factorization = {
                "lambda":sparse_factorization[0],
                "sparse_factors": sparse_factorization[1]
            }

        if self.only_mask:
            scaling = []
        else:
            scaling = [np.array(sparse_factorization["lambda"])[None]]

        factors = self._get_factors_from_op_sparsefacto(sparse_factorization["sparse_factors"])
        sparsity_patterns = [get_sparsity_pattern(w) for w in factors]

        return scaling, factors, sparsity_patterns

    def __check_facto_less_values_than_base(self, layer, sparsity_patterns):
        """
        Check if there is actually less values in the compressed layer than base layer.

        :param layer:
        :param sparsity_patterns:
        :return:
        """
        layer_weights = layer.get_weights()
        nb_val_full_layer = np.sum(np.prod(w.shape) for w in layer_weights)

        nb_val_sparse_factors = np.sum([np.sum(fac) for fac in sparsity_patterns])

        if nb_val_full_layer <= nb_val_sparse_factors:
            logger.info("Less values in full matrix than factorization in layer {}. Keep full matrix. {} <= {}".format(layer.name, nb_val_full_layer, nb_val_sparse_factors))
            return False

        return True
