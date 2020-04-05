import logging
from abc import ABCMeta, abstractmethod

from keras.layers import Conv2D, Dense
from qkmeans.core.utils import build_constraint_set_smart
from qkmeans.palm.palm_fast import hierarchical_palm4msa, palm4msa

from skluc.utils import logger

import numpy as np



class SparseFactorizer(metaclass=ABCMeta):
    def __init__(self, sparsity_fac=2, nb_factor=None, nb_iter=300, hierarchical=True):
        self.sparsity_fac = sparsity_fac
        self.nb_iter = nb_iter
        self.hierarchical = hierarchical
        self.nb_factor = nb_factor

    @abstractmethod
    def apply_factorization(self, matrix):
        pass

    def factorize_layer(self, layer_obj, apply_weights=True):
        """
        Takes a keras layer object as entry and modify its weights as reconstructed by the palm approximation.

        The layer is modifed in place but the inner weight tensor is returned modifed.

        :param layer_obj: The layer object to which modify weights
        :return: The new weights
        """
        if isinstance(layer_obj, Conv2D):
            logger.info("Find {}".format(layer_obj.__class__.__name__))
            layer_weights, layer_bias = layer_obj.get_weights()
            filter_height, filter_width, in_chan, out_chan = layer_weights.shape
            filter_matrix = layer_weights.reshape(filter_height*filter_width*in_chan, out_chan)
            _lambda, op_sparse_factors, reconstructed_filter_matrix = self.apply_factorization(filter_matrix)
            new_layer_weights = reconstructed_filter_matrix.reshape(filter_height, filter_width, in_chan, out_chan)
            if apply_weights:
                layer_obj.set_weights((new_layer_weights, layer_bias))
            return _lambda, op_sparse_factors, new_layer_weights
        elif isinstance(layer_obj, Dense):
            logger.info("Find {}".format(layer_obj.__class__.__name__))
            layer_weights, layer_bias = layer_obj.get_weights()
            _lambda, op_sparse_factors, reconstructed_dense_matrix = self.apply_factorization(layer_weights)
            new_layer_weights = reconstructed_dense_matrix
            if apply_weights:
                layer_obj.set_weights((new_layer_weights, layer_bias))
            return _lambda, op_sparse_factors, new_layer_weights
        else:
            logger.debug("Find {}. Can't Palminize this. Pass.".format(layer_obj.__class__.__name__))
            return None, None, None