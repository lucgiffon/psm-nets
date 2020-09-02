from abc import abstractmethod, ABCMeta
import numpy as np
from skluc.utils import logger
from tensorly.decomposition import matrix_product_state

from palmnet.core.layer_replacer import LayerReplacer
from palmnet.data import Cifar100
from palmnet.layers.tt_layer_conv import TTLayerConv
from palmnet.layers.tt_layer_dense import TTLayerDense
from collections import defaultdict
from keras.layers import Dense, Conv2D

from palmnet.utils import build_dct_tt_ranks, get_facto_for_channel_and_order, DCT_CHANNEL_PREDEFINED_FACTORIZATIONS, TensortrainBadRankException


class LayerReplacerTT(LayerReplacer):
    def __init__(self, rank_value, order, tt_rank0_conv_1=False, use_pretrained=False, *args, **kwargs):
        self.order = order
        self.rank_value = rank_value
        self.use_pretrained = use_pretrained

        self.tt_rank0_conv_1 = tt_rank0_conv_1
        self.tt_ranks_conv = [rank_value] * order + [1]
        self.tt_ranks_dense = [rank_value] * order + [1]
        self.tt_ranks_dense[0] = 1
        if tt_rank0_conv_1:
            self.tt_ranks_conv[0] = 1

        self.tt_ranks_dense = tuple(self.tt_ranks_dense)
        self.tt_ranks_conv = tuple(self.tt_ranks_conv)
        super().__init__(*args, **kwargs)


    def get_mps_decompostion(self, layer):
        matrix_layer = layer.get_weights()[0]

        if isinstance(layer, Conv2D):
            inp_modes_tmp = get_facto_for_channel_and_order(matrix_layer.shape[-2], self.order, dct_predefined_facto=DCT_CHANNEL_PREDEFINED_FACTORIZATIONS)
            out_modes_tmp = get_facto_for_channel_and_order(matrix_layer.shape[-1], self.order, dct_predefined_facto=DCT_CHANNEL_PREDEFINED_FACTORIZATIONS)
            inp_modes = tuple([np.prod(layer.kernel_size)] + list(inp_modes_tmp))
            out_modes = tuple([1] + list(out_modes_tmp))
            tt_ranks = tuple([1] + list(self.tt_ranks_conv))
            # in_out_mod_products = tuple([np.prod(layer.kernel_size)] + [inp_modes[i] * out_modes[i] for i in range(len(out_modes))])

        else:
            inp_modes_tmp = get_facto_for_channel_and_order(matrix_layer.shape[0], self.order, dct_predefined_facto=DCT_CHANNEL_PREDEFINED_FACTORIZATIONS)
            out_modes_tmp = get_facto_for_channel_and_order(matrix_layer.shape[-1], self.order, dct_predefined_facto=DCT_CHANNEL_PREDEFINED_FACTORIZATIONS)
            inp_modes = inp_modes_tmp
            out_modes = out_modes_tmp
            tt_ranks = self.tt_ranks_dense

        in_out_mod_products = tuple(inp_modes[i]*out_modes[i] for i in range(len(out_modes)))


        matrix_layer_tensor_form = np.reshape(matrix_layer, in_out_mod_products)
        res = matrix_product_state(matrix_layer_tensor_form, tt_ranks)

        if not all((core.shape[0] == tt_ranks[idx_core] and  core.shape[-1] == tt_ranks[idx_core+1]) for idx_core, core in enumerate(res)):
            obtained_ranks = tuple([core.shape[0] for core in res] + [res[-1].shape[-1]])
            logger.warning(f"{layer.name}: " + str(TensortrainBadRankException(expected_ranks=tt_ranks, obtained_ranks=obtained_ranks)))
            tt_ranks = obtained_ranks

        lst_shapes = list()
        for i in range(len(inp_modes)):
            lst_shapes.append(tuple([out_modes[i] * tt_ranks[i + 1], tt_ranks[i] * inp_modes[i]]))

        for idx_core, shape_core in enumerate(lst_shapes):
            res[idx_core] = np.reshape(res[idx_core], tuple(shape_core))

        if isinstance(layer, Conv2D):
            res[0] = np.reshape(res[0], (*layer.kernel_size, 1, tt_ranks[1]))
            tt_ranks = tt_ranks[1:]

        return inp_modes_tmp, out_modes_tmp, tt_ranks, res


    ##################################
    # LayerReplacer abstract methods #
    ##################################
    def _apply_replacement(self, layer):
        dct_replacement = dict()
        if isinstance(layer, Conv2D):
            if self.use_pretrained:
                inp_modes, out_modes, tt_ranks, res = self.get_mps_decompostion(layer)
                dct_replacement["lst_weights_cores"] = res
                dct_replacement["inp_modes"] = inp_modes
                dct_replacement["out_modes"] = out_modes
            else:
                tt_ranks = self.tt_ranks_conv
            dct_replacement["tt_ranks"] = tt_ranks
        elif isinstance(layer, Dense):
            if self.use_pretrained:
                inp_modes, out_modes, tt_ranks, res = self.get_mps_decompostion(layer)
                dct_replacement["lst_weights_cores"] = res
                dct_replacement["inp_modes"] = inp_modes
                dct_replacement["out_modes"] = out_modes
            else:
                tt_ranks = self.tt_ranks_dense
            dct_replacement["tt_ranks"] = tt_ranks
        else:
            dct_replacement = None

        return dct_replacement

    def _replace_conv2D(self, layer, dct_compression):

        nb_filters = layer.filters
        strides = layer.strides
        kernel_size = layer.kernel_size
        activation = layer.activation
        padding = layer.padding
        use_bias = layer.use_bias

        if layer.use_bias:
            bias = [layer.get_weights()[-1]]
        else:
            bias = []

        tt_ranks = dct_compression["tt_ranks"]

        if self.use_pretrained:
        # noinspection PyUnreachableCode
        # if False:
            replacing_weights = dct_compression["lst_weights_cores"] + bias
            inp_modes = dct_compression["inp_modes"]
            out_modes = dct_compression["out_modes"]
            replacing_layer = TTLayerConv(filters=nb_filters, use_bias=use_bias,
                                          inp_modes=inp_modes, out_modes=out_modes, mat_ranks=tt_ranks,
                                          kernel_size=kernel_size, strides=strides, padding=padding, activation=activation,
                                          mode="manual")
        else:
            replacing_layer = TTLayerConv(filters=nb_filters, use_bias=use_bias, mat_ranks=tt_ranks, kernel_size=kernel_size, strides=strides, padding=padding, activation=activation, mode="auto")
            replacing_weights = None

        return replacing_layer, replacing_weights, True

    def _replace_dense(self, layer, dct_compression):
        hidden_layer_dim = layer.units
        activation = layer.activation
        use_bias = layer.use_bias

        if layer.use_bias:
            bias = [layer.get_weights()[-1]]
        else:
            bias = []

        tt_ranks = dct_compression["tt_ranks"]
        if self.use_pretrained:
            replacing_weights = dct_compression["lst_weights_cores"] + bias
            inp_modes = dct_compression["inp_modes"]
            out_modes = dct_compression["out_modes"]
            replacing_layer = TTLayerDense(nb_units=hidden_layer_dim, use_bias=use_bias,
                                           inp_modes=inp_modes, out_modes=out_modes, mat_ranks=tt_ranks, activation=activation,
                                           mode="manual")
        else:
            replacing_layer = TTLayerDense(nb_units=hidden_layer_dim, use_bias=use_bias, mat_ranks=tt_ranks, activation=activation, mode="auto")
            replacing_weights = None

        return replacing_layer, replacing_weights, True

    def _set_weights_to_layer(self, replacing_layer, replacing_weights):
        if replacing_weights is not None:
            assert self.use_pretrained
            replacing_layer.set_weights(replacing_weights)
        else:
            return


if __name__ == "__main__":

    from pprint import pprint
    # base_model = Cifar10.load_model("cifar10_tensortrain_base")
    base_model = Cifar100.load_model("cifar100_vgg19_2048x2048")

    # dct_layer_params = build_dct_tt_ranks(base_model)

    keep_last_layer = True
    model_transformer = LayerReplacerTT(rank_value=2, order=4, keep_last_layer=keep_last_layer, keep_first_layer=True)
    new_model = model_transformer.fit_transform(base_model)
    for l in new_model.layers:
        layer_w = l.get_weights()
        print(l.name, l.__class__.__name__)
        # pprint([w for w in layer_w if len(w.shape)>1])

