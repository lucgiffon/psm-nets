from abc import abstractmethod, ABCMeta
import numpy as np

from palmnet.core.palminizer import Palminizer
from palmnet.core.palminizable import Palminizable
from palmnet.data import Cifar100
from keras.models import Model, Sequential
from keras.layers import InputLayer
from palmnet.layers.sparse_masked import SparseFactorisationDense, SparseFactorisationConv2DDensify
from palmnet.utils import get_sparsity_pattern, get_idx_last_layer_of_class, get_idx_first_layer_of_class
from skluc.utils import log_memory_usage, logger
from collections import defaultdict
from keras.layers import Dense, Conv2D


class LayerReplacer(metaclass=ABCMeta):
    def __init__(self, keep_last_layer=False, keep_first_layer=False, dct_name_compression=None):
        self.keep_last_layer = keep_last_layer
        self.keep_first_layer = keep_first_layer
        self.dct_name_compression = dct_name_compression
        self.dct_bool_replaced_layers = defaultdict(lambda: False)
        self.dct_old_name_new_name = defaultdict(lambda: None)
        self.dct_new_name_old_name = defaultdict(lambda: None)

    def __refresh_and_apply_layer_to_input(self, layer, layer_inputs):
        new_fresh_layer = layer.__class__(**layer.get_config())
        old_layer_weights = layer.get_weights()
        x = new_fresh_layer(layer_inputs)
        new_fresh_layer.set_weights(old_layer_weights)
        return x, new_fresh_layer

    @abstractmethod
    def _apply_replacement(self, layer):
        pass

    def fit_transform(self, model):
        self.fit(model)
        return self.transform(model)

    def fit(self, model):
        if self.dct_name_compression is not None:
            raise ValueError("{} has already been fit.".format(self.__class__.__name__))

        self.dct_name_compression = dict()
        for layer in model.layers:
            dct_replacement = self._apply_replacement(layer)
            # should return dict in most case but need to be backward compatible with older implementation of PALM
            self.dct_name_compression[layer.name] = dct_replacement

    def transform(self, model):

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

        idx_last_dense_layer = get_idx_last_layer_of_class(model, Dense) if self.keep_last_layer else -1
        idx_last_dense_layer -= 1
        idx_first_conv_layer = get_idx_first_layer_of_class(model, Conv2D) if self.keep_first_layer else -1
        idx_first_conv_layer -= 1

        for i, layer in enumerate(model.layers[1:]):
            log_memory_usage("Before layer {}".format(layer.name))

            # get all layers input
            layer_inputs = [network_dict['new_output_tensor_of'][curr_layer_input] for curr_layer_input in network_dict['input_layers_of'][layer.name]]
            if len(layer_inputs) == 1:
                layer_inputs = layer_inputs[0]

            sparse_factorization = self.dct_name_compression[layer.name]
            # adapted to the palminized case... not very clean but OK
            bool_find_modif = (sparse_factorization != None and sparse_factorization != (None, None))
            logger.info('Prepare layer {}'.format(layer.name))
            keep_this_layer = (i == idx_last_dense_layer and self.keep_last_layer) or (i == idx_first_conv_layer and self.keep_first_layer)
            if bool_find_modif and not keep_this_layer:
                # if there is a replacement available and not (it is the last layer and we want to keep it as is)
                # create new layer
                if isinstance(layer, Dense):
                    logger.debug("Dense layer treatment")
                    replacing_layer, replacing_weights, bool_modified = self._replace_dense(layer, sparse_factorization)
                elif isinstance(layer, Conv2D):
                    logger.debug("Conv2D layer treatment")
                    replacing_layer, replacing_weights, bool_modified = self._replace_conv2D(layer, sparse_factorization)
                else:
                    raise ValueError("Unsupported layer class")

                if bool_modified: # then replace layer with compressed layer
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
    new_model = model_transformer.transform(base_model)
    for l in new_model.layers:
        layer_w = l.get_weights()
        print(l.name)
        pprint([w for w in layer_w if len(w.shape)>1])

