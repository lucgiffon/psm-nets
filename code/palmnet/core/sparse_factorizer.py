import logging
from abc import ABCMeta, abstractmethod

from keras.layers import Conv2D, Dense
from qkmeans.core.utils import build_constraint_set_smart
from qkmeans.palm.palm_fast import hierarchical_palm4msa, palm4msa

from palmnet.utils import get_sparsity_pattern
from skluc.utils import logger

import numpy as np



class SparseFactorizer(metaclass=ABCMeta):
    def __init__(self, sparsity_fac=2, nb_factor=None, nb_iter=300, hierarchical=True):
        self.sparsity_fac = sparsity_fac
        self.nb_iter = nb_iter
        self.hierarchical = hierarchical
        self.nb_factor = nb_factor

    @staticmethod
    @abstractmethod
    def get_factors_from_op_sparsefacto(op_sparse_factorization):
        pass

    def get_weights_from_sparse_facto(self, sparse_factorization, return_scaling):
        # backward compatibility
        if type(sparse_factorization) == tuple:
            sparse_factorization = {
                "lambda": sparse_factorization[0],
                "sparse_factors": sparse_factorization[1]
            }

        if not return_scaling:
            scaling = []
        else:
            scaling = [np.array(sparse_factorization["lambda"])[None]]

        factors = self.get_factors_from_op_sparsefacto(sparse_factorization["sparse_factors"])
        sparsity_patterns = [get_sparsity_pattern(w) for w in factors]

        return scaling, factors, sparsity_patterns

    @abstractmethod
    def apply_factorization(self, matrix):
        pass

    def factorize_layer(self, layer_obj, apply_weights=True):
        """
        Takes a keras layer object as entry and modify its weights as reconstructed by the palm approximation. (works with conv2D and dense layers)

        The layer can be modified in place with apply_weights=True and the inner weight tensor is returned modifed.

        :param layer_obj: The layer object to which modify weights
        :return: The new weights
        """
        if isinstance(layer_obj, Conv2D):
            logger.info("Find {}".format(layer_obj.__class__.__name__))
            if layer_obj.use_bias :
                layer_weights, layer_bias = layer_obj.get_weights()
            else:
                layer_weights = layer_obj.get_weights()[0]
                layer_bias = []
            _lambda, op_sparse_factors, new_layer_weights = self.factorize_conv2D_weights(layer_weights)
            if apply_weights:
                layer_obj.set_weights([new_layer_weights] + [layer_bias])
            return _lambda, op_sparse_factors, new_layer_weights
        elif isinstance(layer_obj, Dense):
            logger.info("Find {}".format(layer_obj.__class__.__name__))
            if layer_obj.use_bias:
                layer_weights, layer_bias = layer_obj.get_weights()
            else:
                layer_weights = layer_obj.get_weights()
                layer_bias = []
            _lambda, op_sparse_factors, new_layer_weights = self.factorize_dense_weights(layer_weights)
            if apply_weights:
                layer_obj.set_weights([new_layer_weights] + [layer_bias])
            return _lambda, op_sparse_factors, new_layer_weights
        else:
            logger.debug("Find {}. Can't Palminize this. Pass.".format(layer_obj.__class__.__name__))
            return None, None, None

    def factorize_conv2D_weights(self, layer_weights):
        filter_height, filter_width, in_chan, out_chan = layer_weights.shape
        filter_matrix = layer_weights.reshape(filter_height * filter_width * in_chan, out_chan)
        _lambda, op_sparse_factors, reconstructed_filter_matrix = self.apply_factorization(filter_matrix)
        new_layer_weights = reconstructed_filter_matrix.reshape(filter_height, filter_width, in_chan, out_chan)
        return _lambda, op_sparse_factors, new_layer_weights

    def factorize_dense_weights(self, layer_weights):
        _lambda, op_sparse_factors, reconstructed_dense_matrix = self.apply_factorization(layer_weights)
        new_layer_weights = reconstructed_dense_matrix
        return _lambda, op_sparse_factors, new_layer_weights