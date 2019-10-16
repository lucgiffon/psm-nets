import random
import re
import time

from keras.models import Model
from pathlib import Path
import keras.backend as K
import numpy as np
import os

from palmnet.data import Mnist

root_dir = Path(__file__).parent.parent.parent

def insert_layer_nonseq(model, layer_regex, insert_layer_factory,
                        insert_layer_name=None, position='after'):
    """
    Replace, put after or before the layer containing the regex `layer_regex` the layer given by `insert_layer_factory`.

    :param model: The model to modify.
    :param layer_regex: The regex layer to locate.
    :param insert_layer_factory: Factory function that creates keras layer
    :param insert_layer_name: Name of keras layer
    :param position: `before`, `after` or `replace`
    :return:
    """

    # Auxiliary dictionary to describe the network graph
    network_dict = {'input_layers_of': {}, 'new_output_tensor_of': {}}

    # Set the input layers of each layer
    for layer in model.layers:
        for node in layer.outbound_nodes:
            layer_name = node.outbound_layer.name
            if layer_name not in network_dict['input_layers_of']:
                network_dict['input_layers_of'].update(
                        {layer_name: [layer.name]})
            else:
                network_dict['input_layers_of'][layer_name].append(layer.name)

    # Set the output tensor of the input layer
    network_dict['new_output_tensor_of'].update(
            {model.layers[0].name: model.input})

    # Iterate over all layers after the input
    for layer in model.layers[1:]:

        # Determine input tensors
        layer_input = [network_dict['new_output_tensor_of'][layer_aux]
                for layer_aux in network_dict['input_layers_of'][layer.name]]
        if len(layer_input) == 1:
            layer_input = layer_input[0]

        # Insert layer if name matches the regular expression
        if re.match(layer_regex, layer.name):
            if position == 'replace':
                x = layer_input
            elif position == 'after':
                x = layer(layer_input)
            elif position == 'before':
                pass
            else:
                raise ValueError('position must be: before, after or replace')

            new_layer = insert_layer_factory()
            if insert_layer_name:
                new_layer.name = insert_layer_name
            else:
                new_layer.name = '{}_{}'.format(layer.name,
                                                new_layer.name)
            x = new_layer(x)
            print('Layer {} inserted after layer {}'.format(new_layer.name,
                                                            layer.name))
            if position == 'before':
                x = layer(x)
        else:
            x = layer(layer_input)

        # Set new output tensor (the original one, or the one of the inserted
        # layer)
        network_dict['new_output_tensor_of'].update({layer.name: x})

    return Model(inputs=model.inputs, outputs=x)


def imagette_flatten(X, window_h, window_w, window_c, out_h, out_w, stride=1, padding=0):
    X_padded = K.pad(X, [[0, 0], [padding, padding], [padding, padding], [0, 0]])

    windows = []
    for y in range(out_h):
        for x in range(out_w):
            window = K.slice(X_padded, [0, y * stride, x * stride, 0], [-1, window_h, window_w, -1])
            windows.append(window)
    stacked = K.stack(windows)  # shape : [out_h, out_w, n, filter_h, filter_w, c]

    return K.reshape(stacked, [-1, window_c * window_w * window_h])

def convolution(X, W, b, padding, stride):
    """

    :param X: The input 3D tensor.
    :param W: The 4D tensor of filters.
    :param b: The bias for each filter.
    :param padding: The padding size in both dimension.
    :param stride: The stride size in both dimension.
    :return:
    """
    # todo padding "half" on the basis of W
    sample_size, input_height, input_width, nb_in_channels = map(lambda d: d.value, X.get_shape())
    filter_height, filter_width, filter_in_channels, filter_nbr = [d.value for d in W.get_shape()]

    output_height = (input_height + 2*padding - filter_height) // stride + 1
    output_width = (input_width + 2*padding - filter_width) // stride + 1

    X_flat = imagette_flatten(X, filter_height, filter_width, nb_in_channels, output_height, output_width, stride, padding)
    W_flat = K.reshape(W, [filter_height*filter_width*nb_in_channels, filter_nbr])

    z = K.dot(X_flat, W_flat) + b     # b: 1 X filter_n

    return K.permute_dimensions(K.reshape(z, [output_height, output_width, sample_size, filter_nbr]), [2, 0, 1, 3])


class ParameterManager(dict):
    def __init__(self, dct_params, **kwargs):
        super().__init__(self, **dct_params, **kwargs)
        self["--sparsity-factor"] = int(self["--sparsity-factor"]) if self["--sparsity-factor"] is not None else None
        self["--nb-iteration-palm"] = int(self["--nb-iteration-palm"]) if self["--nb-iteration-palm"] is not None else None
        self["--delta-threshold"] = float(self["--delta-threshold"]) if self["--delta-threshold"] is not None else None

        self.__init_identifier()
        self.__init_output_file()

    def __init_identifier(self):
        job_id = os.environ.get('OAR_JOB_ID')  # in case it is running with oarsub job scheduler
        if job_id is None:
            job_id = str(int(time.time()))
            job_id = int(job_id) + random.randint(0, 10 ** len(job_id))
        else:
            job_id = int(job_id)

        self["identifier"] = str(job_id)

    def __init_output_file(self):
        self["output_file_resprinter"] = Path(self["identifier"] + "_results.csv")
        self["output_file_modelprinter"] = Path(self["identifier"] + "_model_layers.npz")

    def __init_seed(self):
        self["--seed"] = int(self["--seed"])
        if self["--seed"] is not None:
            np.random.seed(self["--seed"])
        else:
            self["--seed"] = int(self["--seed"])

    def get_dataset(self):
        """
        Return dataset in shape n x d.

        n: number of observations.
        d: dimensionality of observations.

        :return:
        """
        if self["--mnist"]:
            (x_train, y_train), (x_test, y_test) =  Mnist.load_data()
            return (x_train, y_train), (x_test, y_test)
        else:
            raise NotImplementedError("No dataset specified.")

    def get_model(self):
        if self["--mnist"]:
            return Mnist.load_model() # for now there is only one model for each dataset... it may change
        else:
            raise NotImplementedError("No dataset specified.")


class ResultPrinter:
    """
    Class that handles 1-level dictionnaries and is able to print/write their values in a csv like format.
    """
    def __init__(self, *args, header=True, output_file=None):
        """
        :param args: the dictionnaries objects you want to print.
        :param header: tells if you want to print the header
        :param output_file: path to the outputfile. If None, no outputfile is written on ResultPrinter.print()
        """
        self.__dict = dict()
        self.__header = header
        self.__output_file = output_file

    def add(self, d):
        """
        Add dictionnary after initialisation.

        :param d: the dictionnary object you want to add.
        :return:
        """
        self.__dict.update(d)

    def _get_ordered_items(self):
        all_keys, all_values = zip(*self.__dict.items())
        arr_keys, arr_values = np.array(all_keys), np.array(all_values)
        indexes_sort = np.argsort(arr_keys)
        return list(arr_keys[indexes_sort]), list(arr_values[indexes_sort])

    def print(self):
        """
        Call this function whener you want to print/write to file the content of the dictionnaires.
        :return:
        """
        headers, values = self._get_ordered_items()
        headers = [str(h) for h in headers]
        s_headers = ",".join(headers)
        values = [str(v) for v in values]
        s_values = ",".join(values)
        if self.__header:
            print(s_headers)
        print(s_values)
        if self.__output_file is not None:
            with open(self.__output_file, "w+") as out_f:
                if self.__header:
                    out_f.write(s_headers + "\n")
                out_f.write(s_values + "\n")
