from abc import abstractmethod, ABCMeta
import numpy as np

from palmnet.core.palminize import Palminizer, Palminizable
from palmnet.data import Cifar100
from keras.models import Model, Sequential
from keras.layers import InputLayer
from palmnet.layers.sparse_masked import SparseFactorisationDense, SparseFactorisationConv2DDensify
from palmnet.utils import get_sparsity_pattern, get_idx_last_dense_layer
from skluc.utils import log_memory_usage, logger
from collections import defaultdict
from keras.layers import Dense, Conv2D


class LayerReplacer(metaclass=ABCMeta):
    def __init__(self, keep_last_layer, dct_name_compression):
        self.keep_last_layer = keep_last_layer
        self.dct_name_facto = dct_name_compression
        self.dct_bool_replaced_layers = defaultdict(lambda: False)
        self.dct_old_name_new_name = defaultdict(lambda: None)
        self.dct_new_name_old_name = defaultdict(lambda: None)

    def __refresh_and_apply_layer_to_input(self, layer, layer_inputs):
        new_fresh_layer = layer.__class__(**layer.get_config())
        old_layer_weights = layer.get_weights()
        x = new_fresh_layer(layer_inputs)
        new_fresh_layer.set_weights(old_layer_weights)
        return x, new_fresh_layer

    def fit_transform(self, model):

        if not isinstance(model.layers[0], InputLayer):
            model = Model(input=model.input, output=model.output)

        network_dict = {'input_layers_of': defaultdict(lambda: []), 'new_output_tensor_of': defaultdict(lambda: [])}

        # Set the output tensor of the input layer
        network_dict['new_output_tensor_of'].update(
            {model.layers[0].name: model.input})


        for i, layer in enumerate(model.layers):
            # each layer is set as `input` layer of all its outbound layers
            for node in layer._outbound_nodes:
                outbound_layer_name = node.outbound_layer.name
                network_dict['input_layers_of'][outbound_layer_name].append(layer.name)

        idx_last_dense_layer = get_idx_last_dense_layer(model) if self.keep_last_layer else -1
        idx_last_dense_layer -= 1

        for i, layer in enumerate(model.layers[1:]):
            log_memory_usage("Before layer {}".format(layer.name))

            # get all layers input
            layer_inputs = [network_dict['new_output_tensor_of'][curr_layer_input] for curr_layer_input in network_dict['input_layers_of'][layer.name]]
            if len(layer_inputs) == 1:
                layer_inputs = layer_inputs[0]

            sparse_factorization = self.dct_name_facto[layer.name]
            logger.info('Prepare layer {}'.format(layer.name))
            if sparse_factorization != (None, None) and not (i == idx_last_dense_layer and self.keep_last_layer):
                # if there is a replacement available and not (it is the last layer and we want to keep it as is)
                # create new layer
                if isinstance(layer, Dense):
                    logger.debug("Dense layer treatment")
                    replacing_layer, replacing_weights, less_values_than_base = self._replace_dense(layer, sparse_factorization)
                elif isinstance(layer, Conv2D):
                    logger.debug("Conv2D layer treatment")
                    replacing_layer, replacing_weights, less_values_than_base = self._replace_conv2D(layer, sparse_factorization)
                else:
                    raise ValueError("Unsupported layer class")

                if less_values_than_base: # then replace layer with compressed layer
                    replacing_layer.name = '{}_-_{}'.format(layer.name, replacing_layer.name)

                    x = replacing_layer(layer_inputs)

                    self.dct_old_name_new_name[layer.name] = replacing_layer.name
                    self.dct_new_name_old_name[replacing_layer.name] = layer.name
                    self.dct_bool_replaced_layers[layer.name] = True

                    self._set_weights_to_layer(replacing_layer, replacing_weights)

                    logger.info('Layer {} modified into {}'.format(layer.name, replacing_layer.name))
                else:
                    x, new_fresh_layer = self.__refresh_and_apply_layer_to_input(layer, layer_inputs)
                    logger.info('Layer {} unmodified'.format(new_fresh_layer.name))
            else:
                x, new_fresh_layer = self.__refresh_and_apply_layer_to_input(layer, layer_inputs)
                # x = layer(layer_inputs)
                logger.info('Layer {} unmodified'.format(new_fresh_layer.name))

            network_dict['new_output_tensor_of'].update({layer.name: x})

        model = Model(inputs=model.inputs, outputs=x)

        return model

    def have_been_replaced(self, layer_name):
        return self.dct_bool_replaced_layers[layer_name]

    def get_replaced_layer_name(self, new_layer_name):
        return self.dct_new_name_old_name[new_layer_name]

    def get_replacing_layer_name(self, old_layer_name):
        return self.dct_old_name_new_name[old_layer_name]

    @abstractmethod
    def _replace_conv2D(self, layer, sparse_factorization):
        pass

    @abstractmethod
    def _replace_dense(self, layer, sparse_factorization):
        pass

    @abstractmethod
    def _set_weights_to_layer(self, replacing_layer, replacing_weights):
        pass


class LayerReplacerPalm(LayerReplacer):
    def __init__(self, keep_last_layer, only_mask, dct_name_compression):
        self.only_mask = only_mask
        super().__init__(keep_last_layer, dct_name_compression)

    ##################################
    # LayerReplacer abstract methods #
    ##################################
    def _replace_conv2D(self, layer, sparse_factorization):
        scaling, factor_data, sparsity_patterns = self.__get_weights_from_sparse_facto(sparse_factorization)
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
        scaling, factor_data, sparsity_patterns = self.__get_weights_from_sparse_facto(sparse_factorization)

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
                    new_weight = w
                masked_weights.append(new_weight)
            replacing_weights = masked_weights

        replacing_layer.set_weights(replacing_weights)


    #############################
    # LayerReplacerPalm methods #
    #############################
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

    def __get_weights_from_sparse_facto(self, sparse_factorization):
        # scaling = 1.
        if self.only_mask:
            scaling = []
        else:
            scaling = [np.array(sparse_factorization[0])[None]]

        factors = [fac.toarray() for fac in sparse_factorization[1].get_list_of_factors()]
        sparsity_patterns = [get_sparsity_pattern(w) for w in factors]

        return scaling, factors, sparsity_patterns


if __name__ == "__main__":
    model1 = Sequential()
    old_layer =  Dense(10, input_shape=(10,))
    model1.add(old_layer)

    model2 = Sequential()
    new_layer = old_layer.__class__(**old_layer.get_config())
    model2.add(new_layer)
    new_layer.set_weights(old_layer.get_weights())

    assert (new_layer.get_weights()[0] == old_layer.get_weights()[0]).all()
    assert (new_layer.get_weights()[1] == old_layer.get_weights()[1]).all()


    exit()
    from pprint import pprint
    # base_model = Cifar10.load_model("cifar10_tensortrain_base")
    base_model = Cifar100.load_model("cifar100-resnet20")
    palminizer = Palminizer(sparsity_fac=2,
                            nb_factor=2,
                            nb_iter=2,
                            delta_threshold_palm=1e-6,
                            hierarchical=False,
                            fast_unstable_proj=True)

    palminizable = Palminizable(base_model, palminizer)
    palminizable.palminize()
    pprint(palminizable.sparsely_factorized_layers)
    keep_last_layer, only_mask, dct_name_facto = False, True, palminizable.sparsely_factorized_layers
    model_transformer = LayerReplacer(keep_last_layer, only_mask, dct_name_facto)
    new_model = model_transformer.fit_transform(base_model)
    for l in new_model.layers:
        layer_w = l.get_weights()
        print(l.name)
        pprint([w for w in layer_w if len(w.shape)>1])

