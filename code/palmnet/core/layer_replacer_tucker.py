import numpy as np
import tensorly
from keras.layers import Dense, Conv2D
from tensorly.decomposition import partial_tucker

from palmnet import VBMF
from palmnet.core.layer_replacer import LayerReplacer
from palmnet.layers.low_rank_dense_layer import LowRankDense
from palmnet.layers.tucker_layer import TuckerLayerConv


class LayerReplacerTucker(LayerReplacer):
    def __init__(self, rank_percentage_conv=None, rank_dense=None, rank_percentage_dense=None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.rank_dense = rank_dense
        self.rank_percentage_conv = rank_percentage_conv
        assert self.rank_percentage_conv is None or 0 < self.rank_percentage_conv < 1
        self.rank_percentage_dense = rank_percentage_dense
        assert self.rank_percentage_dense is None or 0 < self.rank_percentage_dense < 1

        if self.rank_dense is not None and self.rank_percentage_dense is not None:
            raise ValueError("rank_dense and rank_percentage_dense can't be set together. Ambiguous.")

        self.int_or_flt_rank = self.rank_dense if self.rank_dense is not None else self.rank_percentage_dense

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

    @staticmethod
    def apply_tucker_or_low_rank_decomposition_to_layer(layer, rank_dense=None, rank_percentage_conv=None):
        dct_replacement = dict()
        if isinstance(layer, Conv2D):
            layer_weights = layer.get_weights()[0]  # h, w, c_in, c_out
            assert len(layer_weights.shape) == 4, "Shape of convolution kernel should be of size 4"
            assert layer.data_format == "channels_last", "filter dimension should be last"
            if rank_percentage_conv is None:
                in_rank, out_rank = LayerReplacerTucker.get_rank_layer(layer_weights)
            else:
                in_rank = int(np.ceil(rank_percentage_conv * layer_weights.shape[2]))  # c_in
                out_rank = int(np.ceil(rank_percentage_conv * layer_weights.shape[3]))  # c_out
            dct_replacement["in_rank"] = in_rank
            dct_replacement["out_rank"] = out_rank
            first, core, last = LayerReplacerTucker.get_tucker_decomposition(layer_weights, in_rank, out_rank)
            first = first[np.newaxis, np.newaxis, :]
            last = last.T
            last = last[np.newaxis, np.newaxis, :]
            dct_replacement["first_conv_weights"] = first
            dct_replacement["core_conv_weights"] = core
            dct_replacement["last_conv_weights"] = last
        elif isinstance(layer, Dense) and rank_dense is not None:
            weight_matrix, bias = layer.get_weights()
            U, S, V = np.linalg.svd(weight_matrix)
            if 0 < rank_dense < 1:
                rank = int(np.ceil(rank_dense * len(S)))
                assert rank >= 1, f"{rank}"
            else:  # type int
                assert type(rank_dense)==int
                rank = rank_dense
            U = U[:, :rank]
            S = S[:rank]
            V = V[:rank, :]

            U = U * np.sqrt(S).reshape(1, -1)
            V = np.sqrt(S).reshape(-1, 1) * V

            dct_replacement["dense_in"] = U
            dct_replacement["dense_out"] = V
            dct_replacement["rank"] = rank

        else:
            dct_replacement = None

        return dct_replacement

    ##################################
    # LayerReplacer abstract methods #
    ##################################
    def _apply_replacement(self, layer):
        return self.apply_tucker_or_low_rank_decomposition_to_layer(layer, rank_dense=self.int_or_flt_rank,
                                                                    rank_percentage_conv=self.rank_percentage_conv)

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

        replacing_layer = TuckerLayerConv(in_rank=in_rank, out_rank=out_rank, filters=nb_filters,
                                          kernel_size=kernel_size, strides=strides, padding=padding, activation=activation,
                                          kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer)

        replacing_weights = [dct_compression["first_conv_weights"]] \
            + [dct_compression["core_conv_weights"]] \
            + [dct_compression["last_conv_weights"]] \
            + [layer.get_weights()[-1]] if layer.use_bias else []

        return replacing_layer, replacing_weights, True

    def _replace_dense(self, layer, dct_compression):
        """Dense layers are not replaced by tucker decomposition"""
        if dct_compression is not None:
            hidden_layer_dim = layer.units
            activation = layer.activation
            regularizer = layer.kernel_regularizer
            bias_regularizer = layer.bias_regularizer

            rank = dct_compression["rank"]
            replacing_layer = LowRankDense(units=hidden_layer_dim, rank=rank, activation=activation, kernel_regularizer=regularizer, bias_regularizer=bias_regularizer)
            replacing_weights = [
                dct_compression["dense_in"], dct_compression["dense_out"]
            ] + [layer.get_weights()[-1]] if layer.use_bias else []
            return replacing_layer, replacing_weights, True
        else:
            return None, None, False

    def _set_weights_to_layer(self, replacing_layer, replacing_weights):
        replacing_layer.set_weights(replacing_weights)

