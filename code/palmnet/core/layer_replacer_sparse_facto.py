from abc import abstractmethod, ABCMeta
import numpy as np

from palmnet.core.layer_replacer import LayerReplacer
from palmnet.core.palminizer import Palminizer
from palmnet.data import Cifar100
from palmnet.layers.sparse_facto_conv2D_masked import SparseFactorisationConv2D
from palmnet.layers.sparse_facto_dense_masked import SparseFactorisationDense
from palmnet.utils import get_sparsity_pattern
from skluc.utils import logger


class LayerReplacerSparseFacto(LayerReplacer):
    """
    Replace layers Conv and Dense of NN by their sparse factorized version.
    """
    def __init__(self, only_mask, sparse_factorizer=None,  intertwine_batchnorm=False, *args, **kwargs):
        self.only_mask = only_mask
        self.sparse_factorizer = sparse_factorizer
        self.intertwine_batchnorm = intertwine_batchnorm
        super().__init__(*args, **kwargs)


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

    def _replace_conv2D(self, layer, dct_compression):
        scaling, factor_data, sparsity_patterns = self.sparse_factorizer.get_weights_from_sparse_facto(dct_compression, return_scaling=not self.only_mask)

        less_values_than_base = self.check_facto_less_values_than_base(layer, sparsity_patterns)

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
            replacing_layer = SparseFactorisationConv2D(use_scaling=not self.only_mask, strides=strides, filters=nb_filters, kernel_size=kernel_size,
                                                        sparsity_patterns=sparsity_patterns, use_bias=layer.use_bias, activation=activation, padding=padding,
                                                        kernel_regularizer=regularizer)
            replacing_weights = scaling + factor_data + [layer.get_weights()[-1]] if layer.use_bias else []

        return replacing_layer, replacing_weights, less_values_than_base

    @staticmethod
    def replace_dense_static(layer, dct_compression, fct_get_weights_from_sparse_facto, use_scaling=True, itnertwine_batchnorm=False):
        scaling, factor_data, sparsity_patterns = fct_get_weights_from_sparse_facto (dct_compression, return_scaling=use_scaling)

        less_values_than_base = LayerReplacerSparseFacto.check_facto_less_values_than_base(layer, sparsity_patterns)

        if not less_values_than_base:
            replacing_weights = None
            replacing_layer = layer
        else:

            hidden_layer_dim = layer.units
            activation = layer.activation
            regularizer = layer.kernel_regularizer
            replacing_layer = SparseFactorisationDense(use_scaling=use_scaling, units=hidden_layer_dim, sparsity_patterns=sparsity_patterns, use_bias=layer.use_bias,
                                                       activation=activation, kernel_regularizer=regularizer, intertwine_batchnorm=itnertwine_batchnorm)
            replacing_weights = scaling + factor_data + [layer.get_weights()[-1]] if layer.use_bias else []

        return replacing_layer, replacing_weights, less_values_than_base

    def _replace_dense(self, layer, dct_compression):
        return self.replace_dense_static(layer, dct_compression, fct_get_weights_from_sparse_facto=self.sparse_factorizer.get_weights_from_sparse_facto,
                                         use_scaling=not self.only_mask, itnertwine_batchnorm=self.intertwine_batchnorm)

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

    @staticmethod
    def check_facto_less_values_than_base(layer, sparsity_patterns):
        """
        Check if there is actually less values in the compressed layer than base layer.

        :param layer:
        :param sparsity_patterns:
        :return:
        """
        layer_weights = layer.get_weights()
        nb_val_full_layer = np.sum(np.prod(w.shape) for w in layer_weights)

        nb_val_sparse_factors = np.sum([np.sum(fac) for fac in sparsity_patterns])
        nb_val_sparse_factors += 1# +1 for the scaling factor
        if nb_val_full_layer <= nb_val_sparse_factors:
            logger.info("Less values in full matrix than factorization in layer {}. Keep full matrix. {} <= {}".format(layer.name, nb_val_full_layer, nb_val_sparse_factors))
            return False

        return True
