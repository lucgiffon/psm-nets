from abc import abstractmethod, ABCMeta
import pickle
import keras
# from self.keras_module.models import Model
# from self.keras_module.layers import InputLayer
# from self.keras_module.layers import Dense, Conv2D

from palmnet.core.palminizable import Palminizable
from palmnet.utils import get_idx_last_layer_of_class, get_idx_first_layer_of_class
from skluc.utils import log_memory_usage, logger
from collections import defaultdict
import pathlib


class LayerReplacer(metaclass=ABCMeta):
    def __init__(self, keep_last_layer=False, keep_first_layer=False, dct_name_compression=None, path_checkpoint_file=None, only_dense=False, keras_module=keras):
        self.keras_module = keras_module
        self.keep_last_layer = keep_last_layer
        self.keep_first_layer = keep_first_layer
        self.only_dense = only_dense
        self.dct_name_compression = dct_name_compression if dct_name_compression is not None else dict()
        self.path_checkpoint_file = path_checkpoint_file  # type: pathlib.Path
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

    def load_dct_name_compression(self):
        with open(str(self.path_checkpoint_file), 'rb') as rb_file:
            self.dct_name_compression = pickle.load(rb_file)

        if type(self.dct_name_compression) == Palminizable:
            self.dct_name_compression = self.dct_name_compression.sparsely_factorized_layers

    def save_dct_name_compression(self):
        if self.path_checkpoint_file is None:
            return

        with open(str(self.path_checkpoint_file), 'wb') as wb_file:
            pickle.dump(self.dct_name_compression, wb_file)

    def fit_transform(self, model):
        self.fit(model)
        return self.transform(model)

    def fit_one_layer(self, layer):
        if layer.name not in self.dct_name_compression:
            dct_replacement = self._apply_replacement(layer)
            # should return dict in most case but need to be backward compatible with older implementation of PALM
            self.dct_name_compression[layer.name] = dct_replacement
            self.save_dct_name_compression()
        else:
            logger.warning("skip layer {} because already in dict".format(layer.name))

    def fit(self, model):
        for layer in model.layers:
            self.fit_one_layer(layer)


    def transform_one_layer(self, layer, idx_layer, layer_inputs):
        sparse_factorization = self.dct_name_compression[layer.name]
        # adapted to the palminized case... not very clean but OK
        bool_find_modif = (sparse_factorization != None and sparse_factorization != (None, None))
        logger.info('Prepare layer {}'.format(layer.name))
        bool_only_dense = not isinstance(layer, self.keras_module.layers.Dense) and self.only_dense
        bool_last_layer = idx_layer == self.idx_last_dense_layer and self.keep_last_layer
        bool_first_layer = idx_layer == self.idx_first_conv_layer and self.keep_first_layer
        keep_this_layer = bool_only_dense or bool_last_layer or bool_first_layer
        if bool_find_modif and not keep_this_layer:
            # if there is a replacement available and not (it is the last layer and we want to keep it as is)
            # create new layer
            if isinstance(layer, self.keras_module.layers.Dense):
                logger.debug("Dense layer treatment")
                replacing_layer, replacing_weights, bool_modified = self._replace_dense(layer, sparse_factorization)
            elif isinstance(layer, self.keras_module.layers.Conv2D):
                logger.debug("Conv2D layer treatment")
                replacing_layer, replacing_weights, bool_modified = self._replace_conv2D(layer, sparse_factorization)
            else:
                raise ValueError("Unsupported layer class")

            if bool_modified:  # then replace layer with compressed layer
                try:
                    replacing_layer.name = '{}_-_{}'.format(layer.name, replacing_layer.name)
                except AttributeError:
                    logger.warning("Found layer with property name unsettable. try _name instead.")
                    replacing_layer._name = '{}_-_{}'.format(layer.name, replacing_layer.name)

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

        return x


    def transform(self, model):

        if not isinstance(model.layers[0], self.keras_module.layers.InputLayer):
            model = self.keras_module.models.Model(input=model.input, output=model.output)

        network_dict = {'input_layers_of': defaultdict(lambda: []), 'new_output_tensor_of': defaultdict(lambda: [])}

        # Set the output tensor of the input layer
        network_dict['new_output_tensor_of'].update(
            {model.layers[0].name: model.input})

        for i, layer in enumerate(model.layers):
            # each layer is set as `input` layer of all its outbound layers
            for node in layer._outbound_nodes:
                outbound_layer_name = node.outbound_layer.name
                network_dict['input_layers_of'][outbound_layer_name].append(layer.name)

        self.idx_last_dense_layer = get_idx_last_layer_of_class(model, self.keras_module.layers.Dense) if self.keep_last_layer else -1
        self.idx_last_dense_layer -= 1
        self.idx_first_conv_layer = get_idx_first_layer_of_class(model, self.keras_module.layers.Conv2D) if self.keep_first_layer else -1
        self.idx_first_conv_layer -= 1

        for i, layer in enumerate(model.layers[1:]):
            log_memory_usage("Before layer {}".format(layer.name))

            # get all layers input
            layer_inputs = [network_dict['new_output_tensor_of'][curr_layer_input] for curr_layer_input in network_dict['input_layers_of'][layer.name]]
            if len(layer_inputs) == 1:
                layer_inputs = layer_inputs[0]

            x = self.transform_one_layer(layer, i, layer_inputs)

            network_dict['new_output_tensor_of'].update({layer.name: x})

        model = self.keras_module.models.Model(inputs=model.inputs, outputs=x)

        return model

    def have_been_replaced(self, layer_name):
        return self.dct_bool_replaced_layers[layer_name]

    def get_replaced_layer_name(self, new_layer_name):
        return self.dct_new_name_old_name[new_layer_name]

    def get_replacing_layer_name(self, old_layer_name):
        return self.dct_old_name_new_name[old_layer_name]

    @abstractmethod
    def _replace_conv2D(self, layer, dct_compression):
        """
        Implementation of this method should return the triplet:

        replacing_weights: list of np.ndarray
        replacing_layer: self.keras_module.layers.Layer
        bool_replaced: tells if the layer should be replaced

        :param layer:
        :param dct_compression:
        :return:
        """
        pass

    @abstractmethod
    def _replace_dense(self, layer, dct_compression):
        """
        Implementation of this method should return the triplet:

        replacing_weights: list of np.ndarray
        replacing_layer: self.keras_module.layers.Layer
        bool_replaced: tells if the layer should be replaced

        :param layer:
        :param dct_compression:
        :return:
        """
        pass

    @abstractmethod
    def _set_weights_to_layer(self, replacing_layer, replacing_weights):
        pass
