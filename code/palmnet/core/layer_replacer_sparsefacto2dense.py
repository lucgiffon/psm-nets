import numpy as np
import tensorly
from keras.layers import Dense, Conv2D
from tensorly.decomposition import partial_tucker

from palmnet import VBMF
from palmnet.core.layer_replacer import LayerReplacer
from palmnet.layers.low_rank_dense_layer import LowRankDense
from palmnet.layers.sparse_facto_conv2D_masked import SparseFactorisationConv2D
from palmnet.layers.sparse_facto_dense_masked import SparseFactorisationDense
from palmnet.layers.tucker_layer import TuckerLayerConv


class LayerReplacerSparseFacto2Dense(LayerReplacer):
    # def __init__(self, *args, **kwargs):
    #     super().__init__(*args, **kwargs)


    def _init_layer_classes(self):
        self.dense_layer_class = SparseFactorisationDense
        self.conv_layer_class = SparseFactorisationConv2D

    ##################################
    # LayerReplacer abstract methods #
    ##################################
    def _apply_replacement(self, layer):
        # returns the object necessary to build the new layer (a dict for instance)
        dct_replacement = dict()
        if isinstance(layer, SparseFactorisationConv2D):
            layer_weights = layer.get_weights()
            lambda_value = layer_weights[0]
            if layer.use_bias:
                bias = layer_weights[-1]
                reconstructed = lambda_value * np.linalg.multi_dot(layer_weights[1:-1])
                assert len(bias.shape) == 1 and bias.shape[0] == layer.output_shape[-1]
            else:
                reconstructed = lambda_value * np.linalg.multi_dot(layer_weights[1:])

            assert np.prod(reconstructed.shape) == np.prod(layer.kernel_size) * layer.input_shape[-1] * layer.output_shape[-1]

            dct_replacement["kernel"] = reconstructed

        elif isinstance(layer, SparseFactorisationDense):
            layer_weights = layer.get_weights()
            lambda_value = layer_weights[0]
            if layer.use_bias:
                bias = layer_weights[-1]
                reconstructed = lambda_value * np.linalg.multi_dot(layer_weights[1:-1])
                assert len(bias.shape) == 1 and bias.shape[0] == layer.output_shape[-1]
            else:
                reconstructed = lambda_value * np.linalg.multi_dot(layer_weights[1:])

            assert np.prod(reconstructed.shape) == layer.input_shape[-1] * layer.output_shape[-1]

            dct_replacement["kernel"] = reconstructed

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

        replacing_layer = self.keras_module.layers.Conv2D(filters=nb_filters, use_bias=layer.use_bias,
                                          kernel_size=kernel_size, strides=strides, padding=padding, activation=activation,
                                          kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer)

        kernel_reshaped = np.reshape(dct_compression["kernel"], (*kernel_size, -1, nb_filters))

        replacing_weights = [kernel_reshaped] + ([layer.get_weights()[-1]] if layer.use_bias else [])

        return replacing_layer, replacing_weights, True

    def _replace_dense(self, layer, dct_compression):
        """Dense layers are not replaced by tucker decomposition"""
        if dct_compression is not None:
            hidden_layer_dim = layer.units
            activation = layer.activation
            regularizer = layer.kernel_regularizer
            bias_regularizer = layer.bias_regularizer

            replacing_layer =  self.keras_module.layers.Dense(units=hidden_layer_dim,
                                           activation=activation, kernel_regularizer=regularizer,
                                           bias_regularizer=bias_regularizer, use_bias=layer.use_bias)
            replacing_weights = [dct_compression["kernel"]] + ([layer.get_weights()[-1]] if layer.use_bias else [])

            return replacing_layer, replacing_weights, True
        else:
            return None, None, False

    def _set_weights_to_layer(self, replacing_layer, replacing_weights):
        replacing_layer.set_weights(replacing_weights)

