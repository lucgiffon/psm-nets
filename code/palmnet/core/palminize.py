import logging
from copy import deepcopy

from keras.layers import Conv2D, Dense
from qkmeans.core.utils import build_constraint_set_smart
from qkmeans.palm.palm_fast import hierarchical_palm4msa

from skluc.utils import logger

import numpy as np


class Palminizable:
    def __init__(self, keras_model, palminizer):
        self.base_model = keras_model
        self.compressed_model = deepcopy(keras_model)
        self.sparsely_factorized_layers = {} # tuples: (base model, compressed model)
        self.param_by_layer = {} # tuples: (base model, compressed model)
        self.flop_by_layer = {}
        self.is_palminized = False
        self.palminizer = palminizer

        self.total_nb_param_base = None
        self.total_nb_param_compressed = None
        self.total_nb_flop_base = None
        self.total_nb_flop_compressed = None


    def palminize(self):
        """
        Takes a keras model object as entry and returns a version of it with all weights matrix palminized.

        Modifications are in-place but the model is still returned.

        :param model: Keras model
        :return: The same, model object with new weights.
        """
        for layer in self.compressed_model.layers:
            _lambda, op_sparse_factors, _ = self.palminizer.palminize_layer(layer)
            self.sparsely_factorized_layers[layer.name] = (_lambda, op_sparse_factors)

        self.is_palminized = True
        return self.compressed_model

    def count_nb_param_layer(self, layer):
        nb_param_layer = (np.prod(layer.kernel.shape) + np.prod(layer.bias.shape)).value
        nb_param_compressed_layer = self.sparsely_factorized_layers[layer.name][1].get_nb_param() + 1 + np.prod(layer.bias.shape).value  # +1 for lambda
        return nb_param_layer, nb_param_compressed_layer

    @staticmethod
    def count_nb_flop_conv_layer(layer, nb_param_layer, nb_param_compressed_layer):
        if layer.padding == "valid":
            padding_horizontal = 0
            padding_vertical = 0
        elif layer.padding == "same":
            padding_horizontal = np.floor(layer.kernel.shape[0])
            padding_vertical = np.floor(layer.kernel.shape[1])
        else:
            raise ValueError("Unknown padding value for convolutional layer {}.".format(layer.name))

        nb_patch_horizontal = (layer.input_shape[1] + 2 * padding_horizontal - layer.kernel.shape[0]) // layer.strides[0]
        nb_patch_vertical = (layer.input_shape[2] + 2 * padding_vertical - layer.kernel.shape[1]) // layer.strides[1]

        imagette_matrix_size = (nb_patch_horizontal * nb_patch_vertical).value

        nb_flop_layer_for_one_imagette = nb_param_layer
        nb_flop_compressed_layer_for_one_imagette = nb_param_compressed_layer

        # *2 for the multiplcations and then sum
        nb_flop_layer = imagette_matrix_size * nb_flop_layer_for_one_imagette * 2
        nb_flop_compressed_layer = imagette_matrix_size * nb_flop_compressed_layer_for_one_imagette * 2

        return nb_flop_layer, nb_flop_compressed_layer

    @staticmethod
    def count_nb_flop_dense_layer(layer, nb_param_layer, nb_param_compressed_layer):
        # *2 for the multiplcations and then sum
        nb_flop_layer = nb_param_layer * 2
        nb_flop_compressed_layer = nb_param_compressed_layer * 2
        return nb_flop_layer, nb_flop_compressed_layer

    def count_model_param_and_flops(self):
        """
        Return the number of params and the number of flops of 2DConvolutional Layers and Dense Layers for both the base model and the compressed model.

        :return:
        """
        nb_param_base, nb_param_compressed, nb_flop_base, nb_flop_compressed = 0, 0, 0, 0

        for layer in self.base_model.layers:

            if isinstance(layer, Conv2D):
                nb_param_layer, nb_param_compressed_layer = self.count_nb_param_layer(layer)
                nb_flop_layer, nb_flop_compressed_layer = self.count_nb_flop_conv_layer(layer, nb_param_layer, nb_param_compressed_layer)

            elif isinstance(layer, Dense):
                nb_param_layer, nb_param_compressed_layer = self.count_nb_param_layer(layer)
                nb_flop_layer, nb_flop_compressed_layer = self.count_nb_flop_dense_layer(layer, nb_param_layer, nb_param_compressed_layer)

            else:
                logger.warning("Layer {}, class {}, hasn't been compressed".format(layer.name, layer.__class__.__name__))
                nb_param_compressed_layer, nb_param_layer, nb_flop_layer, nb_flop_compressed_layer = 0, 0, 0, 0

            self.param_by_layer[layer.name] = (nb_param_layer, nb_param_compressed_layer)
            self.flop_by_layer[layer.name] = (nb_flop_layer, nb_flop_compressed_layer)

            nb_param_base += nb_param_layer
            nb_param_compressed += nb_param_compressed_layer
            nb_flop_base += nb_flop_layer
            nb_flop_compressed += nb_flop_compressed_layer

        self.total_nb_param_base = nb_param_base
        self.total_nb_param_compressed = nb_param_compressed
        self.total_nb_flop_base = nb_flop_base
        self.total_nb_flop_compressed = nb_flop_compressed

        return nb_param_base, nb_param_compressed, nb_flop_base, nb_flop_compressed

    @staticmethod
    def compile_model(model):
        model.compile(loss='categorical_crossentropy',
                       optimizer="adam",
                       metrics=['categorical_accuracy'])

    def evaluate(self, x_test, y_test):

        self.compile_model(self.compressed_model)
        self.compile_model(self.base_model)

        score_base, acc_base = self.base_model.evaluate(x_test, y_test)
        score_compressed, acc_compressed = self.compressed_model.evaluate(x_test, y_test)

        return score_base, acc_base, score_compressed, acc_compressed

class Palminizer:
    def __init__(self, sparsity_fac=2, nb_iter=300, delta_threshold_palm=1e-6):
        self.sparsity_fac = sparsity_fac
        self.nb_iter = nb_iter
        self.delta_threshold_palm = delta_threshold_palm

    def apply_palm(self, matrix):
        """
        Apply Hierarchical-PALM4MSA algorithm to the input matrix and return the reconstructed approximation from
        the sparse factorisation.

        :param matrix: The matrix to apply PALM to.
        :param sparsity_fac: The sparsity factor for PALM.
        :return:
        """
        logging.info("Applying palm function to matrix with shape {}".format(matrix.shape))
        transposed = False

        if matrix.shape[0] > matrix.shape[1]:
            # we want the bigger dimension to be on right due to the residual computation that should remain big
            matrix = matrix.T
            transposed = True

        left_dim, right_dim = matrix.shape
        A = min(left_dim, right_dim)
        B = max(left_dim, right_dim)
        assert A == left_dim and B == right_dim, "Dimensionality problem: left dim should be higher than right dim before palm"

        nb_factors = int(np.log2(A))

        lst_factors = [np.eye(A) for _ in range(nb_factors + 1)]
        lst_factors[-1] = np.zeros((A, B))
        _lambda = 1.  # init the scaling factor at 1

        lst_proj_op_by_fac_step, lst_proj_op_by_fac_step_desc = build_constraint_set_smart(left_dim=left_dim,
                                                                                           right_dim=right_dim,
                                                                                           nb_factors=nb_factors + 1, # this is due to constant as first factor (so will be identity)
                                                                                           sparsity_factor=self.sparsity_fac,
                                                                                           residual_on_right=True,
                                                                                           fast_unstable_proj=False)

        final_lambda, final_factors, final_X, _, _ = hierarchical_palm4msa(
            arr_X_target=matrix,
            lst_S_init=lst_factors,
            lst_dct_projection_function=lst_proj_op_by_fac_step,
            f_lambda_init=_lambda,
            nb_iter=self.nb_iter,
            update_right_to_left=True,
            residual_on_right=True,
            delta_objective_error_threshold_palm=self.delta_threshold_palm)

        if transposed:
            return final_lambda, final_factors.transpose(), final_X.T
        else:
            return final_lambda, final_factors, final_X

    def palminize_layer(self, layer_obj):
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
            _lambda, op_sparse_factors, reconstructed_filter_matrix = self.apply_palm(filter_matrix)
            new_layer_weights = reconstructed_filter_matrix.reshape(filter_height, filter_width, in_chan, out_chan)
            layer_obj.set_weights((new_layer_weights, layer_bias))
            return _lambda, op_sparse_factors, new_layer_weights
        elif isinstance(layer_obj, Dense):
            logger.info("Find {}".format(layer_obj.__class__.__name__))
            layer_weights, layer_bias = layer_obj.get_weights()
            _lambda, op_sparse_factors, reconstructed_dense_matrix = self.apply_palm(layer_weights)
            new_layer_weights = reconstructed_dense_matrix
            layer_obj.set_weights((new_layer_weights, layer_bias))
            return _lambda, op_sparse_factors, new_layer_weights
        else:
            logger.debug("Find {}. Can't Palminize this. Pass.".format(layer_obj.__class__.__name__))
            return None, None, None