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


class LayerReplacerSparseFactoActivations(LayerReplacerSparseFacto, metaclass=ABCMeta):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


    def fit(self, model):
        raise NotImplementedError("Fit and transform are intertwinned. Call fit_transform instead.")

    def transform(self, model):
        raise NotImplementedError("Fit and transform are intertwinned. Call fit_transform instead.")

    def fit_transform(self, model):

        model, network_dict = self.prepare_transform(model)

        new_model = None

        for i, layer in enumerate(model.layers[1:]):
            log_memory_usage("Before layer {}".format(layer.name))

            # get all layers input
            layer_inputs = [network_dict['new_output_tensor_of'][curr_layer_input] for curr_layer_input in network_dict['input_layers_of'][layer.name]]
            if len(layer_inputs) == 1:
                layer_inputs = layer_inputs[0]

            self.sparse_factorizer.set_preprocessing_model(new_model)
            self.sparse_factorizer.set_layer_to_factorize_name(layer.name)

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
