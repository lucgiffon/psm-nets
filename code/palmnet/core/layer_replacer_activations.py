from abc import abstractmethod, ABCMeta
import pickle
import keras
# from self.keras_module.models import Model
# from self.keras_module.layers import InputLayer
# from self.keras_module.layers import Dense, Conv2D
from palmnet.core.layer_replacer import LayerReplacer
from palmnet.core.layer_replacer_sparse_facto import LayerReplacerSparseFacto
from palmnet.core.palminizable import Palminizable
from palmnet.utils import get_idx_last_layer_of_class, get_idx_first_layer_of_class
from skluc.utils import log_memory_usage, logger
from collections import defaultdict
import pathlib


class LayerReplacerActivations(LayerReplacerSparseFacto, metaclass=ABCMeta):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


    def fit(self, model):
        raise NotImplementedError("Fit and transform are intertwinned. Call fit_transform instead.")

    def transform(self, model):
        raise NotImplementedError("Fit and transform are intertwinned. Call fit_transform instead.")

    def fit_transform(self, model):

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

        idx_last_dense_layer = get_idx_last_layer_of_class(model, self.keras_module.layers.Dense) if self.keep_last_layer else -1
        idx_last_dense_layer -= 1
        idx_first_conv_layer = get_idx_first_layer_of_class(model, self.keras_module.layers.Conv2D) if self.keep_first_layer else -1
        idx_first_conv_layer -= 1

        new_model = None

        for i, layer in enumerate(model.layers[1:]):
            log_memory_usage("Before layer {}".format(layer.name))

            # get all layers input
            layer_inputs = [network_dict['new_output_tensor_of'][curr_layer_input] for curr_layer_input in network_dict['input_layers_of'][layer.name]]
            if len(layer_inputs) == 1:
                layer_inputs = layer_inputs[0]

            self.sparse_factorizer.set_preprocessing_model(new_model)
            # the fit method from before
            # {
            self.fit_one_layer(layer)
            # }

            # the transform method from before: doesn't change either
            # {
            x = self.transform_one_layer(layer, i, layer_inputs)
            # }

            network_dict['new_output_tensor_of'].update({layer.name: x})

            new_model = self.keras_module.models.Model(inputs=model.inputs, outputs=x)

        return new_model
