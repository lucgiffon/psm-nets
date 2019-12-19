import pickle
import zlib
import pathlib
import random
import re
import time
from keras import Input, Sequential
from keras.engine import InputLayer
from keras.layers import Input
from keras.models import Model
from pathlib import Path
import keras.backend as K
import numpy as np
import os
from collections import defaultdict
from palmnet.visualization.utils import get_dct_result_files_by_root, build_df

from skluc.utils import logger
import scipy

from palmnet.data import Mnist, Test, Cifar10, Cifar100, Svhn

root_dir = Path(__file__).parent.parent.parent

def replace_intermediate_layer_in_keras(model, layer_name, new_layer):
    raise NotImplementedError("Doesn't work for bizarre layers")
    from keras.models import Model

    if not isinstance(model .layers[0], InputLayer):
        model = Model(input=model.input, output=model.output)

    layers = [l for l in model.layers]

    x = layers[0].output
    for layer in layers:
        if layer_name == layer.name:
            x = new_layer(x)
        else:
            x = layers(x)

    new_model = Model(input=layers[0].input, output=x)
    return new_model

def insert_intermediate_layer_in_keras(model, layer_id, new_layer):
    raise NotImplementedError("Doesn't work for bizarre layers")
    from keras.models import Model

    layers = [l for l in model.layers]

    x = layers[0].output
    for i in range(1, len(layers)):
        if i == layer_id:
            x = new_layer(x)
        x = layers[i](x)

    new_model = Model(input=layers[0].input, output=x)
    return new_model

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
    # raise NotImplementedError("Doesn't work")
    # Auxiliary dictionary to describe the network graph
    network_dict = {'input_layers_of': defaultdict(lambda: []), 'new_output_tensor_of': defaultdict(lambda: [])}

    if not isinstance(model .layers[0], InputLayer):
        model = Model(input=model.input, output=model.output)

    # Set the input layers of each layer
    for layer in model.layers:
        # each layer is set as `input` layer of all its outbound layers
        for node in layer._outbound_nodes:
            outbound_layer_name = node.outbound_layer.name
            network_dict['input_layers_of'].update({outbound_layer_name: [layer.name]})

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
            logger.debug('Layer {} inserted after layer {}'.format(new_layer.name,
                                                            layer.name))
            if position == 'before':
                x = layer(x)
        else:
            x = layer(layer_input)

        # Set new output tensor (the original one, or the one of the inserted
        # layer)
        network_dict['new_output_tensor_of'].update({layer.name: x})

    newModel = Model(inputs=model.inputs, outputs=x)

    return newModel


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
        self["output_file_modelprinter"] = Path(self["identifier"] + "_model_layers.pckle")

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
        elif self["--cifar10"]:
            (x_train, y_train), (x_test, y_test) = Cifar10.load_data()
            return (x_train, y_train), (x_test, y_test)
        elif self["--cifar100"]:
            (x_train, y_train), (x_test, y_test) = Cifar100.load_data()
            return (x_train, y_train), (x_test, y_test)
        elif self["--svhn"]:
            (x_train, y_train), (x_test, y_test) = Svhn.load_data()
            return (x_train, y_train), (x_test, y_test)
        elif self["--test-data"]:
            (x_train, y_train), (x_test, y_test) = Test.load_data()
            return (x_train, y_train), (x_test, y_test)

        else:
            raise NotImplementedError("No dataset specified.")

    def get_model(self):
        if self["--mnist-lenet"]:
            return Mnist.load_model() # for now there is only one model for each dataset... it may change
        elif self["--cifar10-vgg19"]:
            return Cifar10.load_model()
        elif self["--cifar100-vgg19"]:
            return Cifar100.load_model()
        elif self["--svhn-vgg19"]:
            return Svhn.load_model()
        elif self["--test-model"]:
            return Test.load_model()
        else:
            raise NotImplementedError("No dataset specified.")


def timeout_signal_handler(signum, frame):
    raise TimeoutError("TOO MUCH TIME FORCE EXIT")

class ParameterManagerFinetune(ParameterManager):
    def __init__(self, dct_params, **kwargs):
        super().__init__(dct_params, **kwargs)
        self.__init_hash_expe()

        self["--input-dir"] = pathlib.Path(self["--input-dir"])
        self["--walltime"] = int(self["--walltime"])

        self.__init_model_path()
        self.__init_output_file()

    def __init_hash_expe(self):
        lst_elem_to_remove_for_hash = [
            'output_file_modelprinter',
            'identifier',
            'output_file_resprinter',
            '-v',
            '--help',
            '--input-dir',
            "--walltime"
        ]
        keys_expe = sorted(self.keys())
        any(keys_expe.remove(item) for item in lst_elem_to_remove_for_hash)
        val_expe = [self[k] for k in keys_expe]
        str_expe = [str(val) for pair in zip(keys_expe, val_expe) for val in pair]
        self["hash"] = hex(zlib.crc32(str.encode("".join(str_expe))))


    def __init_output_file(self):
        self["output_file_resprinter"] = Path(self["hash"] + "_results.csv")
        self["output_file_modelprinter"] = Path(self["hash"] + "_model.h5")
        self["output_file_notfinishedprinter"] = Path(self["hash"] + ".notfinished")
        self["output_file_finishedprinter"] = Path(self["hash"] + ".finished")
        self["output_file_tensorboardprinter"] = Path(self["hash"] + ".tb")
        self["output_file_csvcbprinter"] = Path(self["hash"] + "_history.csv")

    def __init_model_path(self):
        df = get_df(self["--input-dir"])
        keys_of_interest = ['--cifar10',
                            '--cifar10-vgg19',
                            '--cifar100',
                            '--cifar100-vgg19',
                            '--delta-threshold',
                            '--hierarchical',
                            '--mnist',
                            '--mnist-lenet',
                            '--nb-iteration-palm',
                            '--sparsity-factor',
                            '--svhn',
                            '--svhn-vgg19',
                            '--test-data',
                            '--test-model',
                            ]
        queries = []
        for k in keys_of_interest:
            query = "(df['{}']=={})".format(k, self[k])
            queries.append(query)

        s_query = " & ".join(queries)
        s_eval = "df[({})]".format(s_query)
        line_of_interest = eval(s_eval)

        assert len(line_of_interest) == 1, "The parameters doesn't allow to discriminate only one pre-trained model in directory"

        self["input_model_path"] = self["--input-dir"] / line_of_interest["output_file_modelprinter"][0]

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

def get_palminized_model_and_df(path):
    src_result_dir = pathlib.Path(path)
    dct_output_files_by_root = get_dct_result_files_by_root(src_results_dir=src_result_dir, old_filename_objective=True)

    col_to_delete = []

    dct_oarid_palminized_model = {}
    for root_name, job_files in dct_output_files_by_root.items():
        objective_file_path = src_result_dir / job_files["palminized_model"]
        loaded_model = pickle.load(open(objective_file_path, 'rb'))
        dct_oarid_palminized_model[root_name] = loaded_model

    df_results = build_df(src_result_dir, dct_output_files_by_root, col_to_delete)
    return dct_oarid_palminized_model, df_results

def get_df(path):
    src_result_dir = pathlib.Path(path)
    dct_output_files_by_root = get_dct_result_files_by_root(src_results_dir=src_result_dir, old_filename_objective=True)

    col_to_delete = []

    df_results = build_df(src_result_dir, dct_output_files_by_root, col_to_delete)
    return df_results


def get_sparsity_pattern(arr):
    non_zero = arr != 0
    sparsity_pattern = np.zeros_like(arr)
    sparsity_pattern[non_zero] = 1
    return sparsity_pattern


def create_random_block_diag(dim1, dim2, block_size, mask=False, greedy=True):
    """
    Create a random block diagonal matrix.

    :param dim1: Number of lines
    :param dim2: Number of cols
    :param block_size: Size of each square block (square if possible)
    :param mask: Return a mask for block diagonal matrix (ones instead of random values)
    :param greedy: Always try to put the biggest possible blocks: result in a matrix with less values (and more concentrated)
    :return:
    """
    min_dim = min(dim1, dim2)
    max_dim = max(dim1, dim2)

    # block create function will be used like: block_creation_function(block_shape)
    if mask:
        block_creation_function = np.ones
    else:
        block_creation_function = lambda shape: np.random.rand(*shape)

    block_diagonals = []
    remaining_out = max_dim
    while remaining_out > 0:

        blocks = []  # blocks = [np.random.rand(block_size, block_size) for _ in range(min_dim//block_size + 1)]
        remaining_in = min_dim
        while remaining_in > 0:
            block = block_creation_function((block_size, block_size))
            blocks.append(block)
            remaining_in -= block_size

        if remaining_in != 0:  # case min_dim % block_size != 0
            if max_dim == dim1 and greedy:
                blocks[-1] = blocks[-1][:, :remaining_in]
            elif max_dim == dim2 and greedy:
                blocks[-1] = blocks[-1][:remaining_in, :]
            else:
                blocks[-1] = blocks[-1][:remaining_in, :remaining_in]

        block_diag = scipy.linalg.block_diag(*blocks)
        block_diagonals.append(block_diag)

        if max_dim == dim1:
            remaining_out -= block_diag.shape[0]
        else:
            remaining_out -= block_diag.shape[1]

    if remaining_out != 0:  # case max_dim % min_dim != 0
        if max_dim == dim1:
            block_diagonals[-1] = block_diagonals[-1][:remaining_out, :]
        else:
            block_diagonals[-1] = block_diagonals[-1][:, :remaining_out]

    if max_dim == dim1:
        final_matrix = np.vstack(block_diagonals)
    else:
        final_matrix = np.hstack(block_diagonals)

    return final_matrix

def create_sparse_matrix_pattern(shape, block_size):
    """
    Create a random mask for a sparse matrix with (+/- 1) `block_size` value in each line and each col.

    If you want exactly `block_size` value in each line and each col, shape must be square and block_size can divide shape
    :param shape: The shape of the matrix
    :param block_size:
    :return:
    """
    base_block_diag = create_random_block_diag(shape[0], shape[1], block_size, mask=True)
    lines_permutation = np.random.permutation(shape[0])
    column_permutation = np.random.permutation(shape[1])
    final_sparse_matrix = base_block_diag[lines_permutation]
    final_sparse_matrix = final_sparse_matrix[:, column_permutation]
    return final_sparse_matrix

def create_sparse_factorization_pattern(shape, block_size, nb_factors):
    min_dim = min(*shape)
    sparse_factors = [create_sparse_matrix_pattern((shape[0], min_dim), block_size)]
    for _ in range(nb_factors-2):
        sparse_factors.append(create_sparse_matrix_pattern((min_dim, min_dim), block_size))
    sparse_factors.append(create_sparse_matrix_pattern((min_dim, shape[1]), block_size))
    return sparse_factors