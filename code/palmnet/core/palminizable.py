from copy import deepcopy

import numpy as np

from skluc.utils import logger
# from palmnet.core.palminizer import Palminizer

class Palminizable:
    def __init__(self, keras_model, palminizer):
        self.base_model = keras_model
        # self.compressed_model = deepcopy(keras_model)
        self.sparsely_factorized_layers = {}  # tuples: (base model, compressed model)
        self.param_by_layer = {}  # tuples: (base model, compressed model)
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
            _lambda, op_sparse_factors, _ = self.palminizer.factorize_layer(layer)
            self.sparsely_factorized_layers[layer.name] = (_lambda, op_sparse_factors)

        self.is_palminized = True
        return self.compressed_model

    @staticmethod
    def count_nb_param_layer(layer, dct_layer_sparse_facto_op=None):
        params = layer.get_weights()
        if layer.bias is not None:
            assert len(params[-1].shape) == 1, "Last weight matrix in get_weights of layer {} should be of len(shape) == 1. Shape is length {}".format(layer.name, len(layer.shape))
            nb_param_layer_bias = params[-1].size
            params = params[:-1]
        else:
            nb_param_layer_bias = 0

        nb_param_layer = 0
        for w in params:
            nb_param_layer += w.size

        if dct_layer_sparse_facto_op is not None:
            nb_param_compressed_layer = int(dct_layer_sparse_facto_op[layer.name][1].get_nb_param() + 1 + np.prod(layer.bias.shape))  # +1 for lambda
            return nb_param_layer, nb_param_compressed_layer, nb_param_layer_bias
        else:
            return nb_param_layer, 0, nb_param_layer_bias

    @staticmethod
    def count_nb_flop_conv_layer(layer, nb_param_layer, nb_param_layer_bias, nb_param_compressed_layer=None):
        if layer.padding == "valid":
            padding_horizontal = 0
            padding_vertical = 0
        elif layer.padding == "same":
            padding_horizontal = int(layer.kernel_size[0]) // 2
            padding_vertical = int(layer.kernel_size[1]) // 2
        else:
            raise ValueError("Unknown padding value for convolutional layer {}.".format(layer.name))

        nb_patch_horizontal = (layer.input_shape[1] + 2 * padding_horizontal - layer.kernel_size[0]) // layer.strides[0]
        nb_patch_vertical = (layer.input_shape[2] + 2 * padding_vertical - layer.kernel_size[1]) // layer.strides[1]

        imagette_matrix_size = int(nb_patch_horizontal * nb_patch_vertical)

        nb_flop_layer_for_one_imagette = nb_param_layer * 2 + nb_param_layer_bias
        # *2 for the multiplcations and then sum
        nb_flop_layer = imagette_matrix_size * nb_flop_layer_for_one_imagette

        if nb_param_compressed_layer is not None:
            nb_flop_compressed_layer_for_one_imagette = nb_param_compressed_layer
            nb_flop_compressed_layer = imagette_matrix_size * nb_flop_compressed_layer_for_one_imagette * 2

            return nb_flop_layer, nb_flop_compressed_layer
        else:
            return nb_flop_layer, 0

    @staticmethod
    def count_nb_flop_dense_layer(layer, nb_param_layer, nb_param_layer_bias, nb_param_compressed_layer=None):
        # *2 for the multiplcations and then sum
        nb_flop_layer = nb_param_layer * 2
        if nb_param_compressed_layer is not None:
            nb_flop_compressed_layer = nb_param_compressed_layer * 2 + nb_param_layer_bias
            return nb_flop_layer, nb_flop_compressed_layer
        else:
            return nb_flop_layer, 0

    @staticmethod
    def count_model_param_and_flops_(model, dct_layer_sparse_facto_op=None):
        """
        Return the number of params and the number of flops of 2DConvolutional Layers and Dense Layers for both the base model and the compressed model.

        :return:
        """
        from keras.layers import Conv2D, Dense

        from palmnet.layers import Conv2DCustom
        from palmnet.layers.sparse_facto_sparse_tensor_deprecated import SparseFactorisationDense

        nb_param_base, nb_param_compressed, nb_flop_base, nb_flop_compressed = 0, 0, 0, 0

        param_by_layer = {}
        flop_by_layer = {}

        for layer in model.layers:
            logger.warning("Process layer {}".format(layer.name))
            if isinstance(layer, Conv2D) or isinstance(layer, Conv2DCustom):
                nb_param_layer, nb_param_compressed_layer, nb_param_layer_bias = Palminizable.count_nb_param_layer(layer, dct_layer_sparse_facto_op)
                nb_flop_layer, nb_flop_compressed_layer = Palminizable.count_nb_flop_conv_layer(layer, nb_param_layer, nb_param_layer_bias, nb_param_compressed_layer)

            elif isinstance(layer, Dense) or isinstance(layer, SparseFactorisationDense):
                nb_param_layer, nb_param_compressed_layer, nb_param_layer_bias = Palminizable.count_nb_param_layer(layer, dct_layer_sparse_facto_op)
                nb_flop_layer, nb_flop_compressed_layer = Palminizable.count_nb_flop_dense_layer(layer, nb_param_layer, nb_param_layer_bias, nb_param_compressed_layer)

            else:
                logger.warning("Layer {}, class {}, hasn't been compressed".format(layer.name, layer.__class__.__name__))
                nb_param_compressed_layer, nb_param_layer, nb_flop_layer, nb_flop_compressed_layer = 0, 0, 0, 0

            param_by_layer[layer.name] = nb_param_layer
            flop_by_layer[layer.name] = nb_flop_layer

            nb_param_base += nb_param_layer
            nb_param_compressed += nb_param_compressed_layer
            nb_flop_base += nb_flop_layer
            nb_flop_compressed += nb_flop_compressed_layer

        return nb_param_base, nb_param_compressed, nb_flop_base, nb_flop_compressed, param_by_layer, flop_by_layer

    def count_model_param_and_flops(self):
        """
        Return the number of params and the number of flops of 2DConvolutional Layers and Dense Layers for both the base model and the compressed model.

        :return:
        """
        self.total_nb_param_base, self.total_nb_param_compressed, self.total_nb_flop_base, self.total_nb_flop_compressed, self.param_by_layer, self.flop_by_layer = self.count_model_param_and_flops_(
            self.base_model, self.sparsely_factorized_layers)
        #
        # nb_param_base, nb_param_compressed, nb_flop_base, nb_flop_compressed = 0, 0, 0, 0
        #
        # for layer in self.base_model.layers:
        #     logger.warning("Process layer {}".format(layer.name))
        #     if isinstance(layer, Conv2D):
        #         nb_param_layer, nb_param_compressed_layer = self.count_nb_param_layer(layer, self.sparsely_factorized_layers)
        #         nb_flop_layer, nb_flop_compressed_layer = self.count_nb_flop_conv_layer(layer, nb_param_layer, nb_param_compressed_layer)
        #
        #     elif isinstance(layer, Dense):
        #         nb_param_layer, nb_param_compressed_layer = self.count_nb_param_layer(layer, self.sparsely_factorized_layers)
        #         nb_flop_layer, nb_flop_compressed_layer = self.count_nb_flop_dense_layer(layer, nb_param_layer, nb_param_compressed_layer)
        #
        #     else:
        #         logger.warning("Layer {}, class {}, hasn't been compressed".format(layer.name, layer.__class__.__name__))
        #         nb_param_compressed_layer, nb_param_layer, nb_flop_layer, nb_flop_compressed_layer = 0, 0, 0, 0
        #
        #     self.param_by_layer[layer.name] = (nb_param_layer, nb_param_compressed_layer)
        #     self.flop_by_layer[layer.name] = (nb_flop_layer, nb_flop_compressed_layer)
        #
        #     nb_param_base += nb_param_layer
        #     nb_param_compressed += nb_param_compressed_layer
        #     nb_flop_base += nb_flop_layer
        #     nb_flop_compressed += nb_flop_compressed_layer

        return self.total_nb_param_base, self.total_nb_param_compressed, self.total_nb_flop_base, self.total_nb_flop_compressed

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
