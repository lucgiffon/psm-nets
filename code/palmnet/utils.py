import re
from keras.engine import InputLayer
from keras.models import Model
from pathlib import Path
import numpy as np
from collections import defaultdict

from palmnet.core.palminize import Palminizable

from skluc.utils import logger
import scipy
import scipy.special

root_dir = Path(__file__).parent.parent.parent

def np_create_permutation_from_weight_matrix(weight_matrix, threshold):
    softmax_1 = scipy.special.softmax(weight_matrix, axis=1)
    softmax_0 = scipy.special.softmax(weight_matrix, axis=0)
    soft_permutation_mat = np.multiply(softmax_1, softmax_0)
    soft_permutation_mat[soft_permutation_mat < threshold] = 0
    soft_permutation_mat[soft_permutation_mat >= threshold] = 1
    return soft_permutation_mat

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


def timeout_signal_handler(signum, frame):
    raise TimeoutError("TOO MUCH TIME FORCE EXIT")


def count_model_param_and_flops(model, dct_layer_sparse_facto_op=None):
    """
    Return the number of params and the number of flops of 2DConvolutional Layers and Dense Layers for both the base model and the compressed model.

    :return:
    """
    from keras.layers import Conv2D, Dense

    from palmnet.layers import Conv2DCustom
    from palmnet.layers.sparse_tensor import SparseFactorisationDense

    nb_param_base, nb_param_compressed, nb_flop_base, nb_flop_compressed = 0, 0, 0, 0

    param_by_layer = {}
    flop_by_layer = {}

    for layer in model.layers:
        logger.warning("Process layer {}".format(layer.name))
        if isinstance(layer, Conv2D) or isinstance(layer, Conv2DCustom):
            nb_param_layer, nb_param_compressed_layer  = Palminizable.count_nb_param_layer(layer, dct_layer_sparse_facto_op)
            nb_flop_layer, nb_flop_compressed_layer = Palminizable.count_nb_flop_conv_layer(layer, nb_param_layer, dct_layer_sparse_facto_op)

        elif isinstance(layer, Dense) or isinstance(layer, SparseFactorisationDense):
            nb_param_layer, nb_param_compressed_layer  = Palminizable.count_nb_param_layer(layer, dct_layer_sparse_facto_op)
            nb_flop_layer, nb_flop_compressed_layer = Palminizable.count_nb_flop_dense_layer(layer, nb_param_layer, dct_layer_sparse_facto_op)

        else:
            logger.warning("Layer {}, class {}, hasn't been compressed".format(layer.name, layer.__class__.__name__))
            nb_param_compressed_layer, nb_param_layer, nb_flop_layer, nb_flop_compressed_layer = 0, 0, 0, 0

        param_by_layer[layer.name] = nb_param_layer
        flop_by_layer[layer.name] = nb_flop_layer

        nb_param_base += nb_param_layer
        nb_param_compressed += nb_param_compressed_layer
        nb_flop_base += nb_flop_layer
        nb_flop_compressed += nb_flop_compressed_layer

    return nb_param_base, nb_param_compressed, nb_flop_base, nb_flop_compressed, param_by_layer, flop_by_layer


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

def create_permutation_matrix(d, dtype=float):
    lines_permutation = np.random.permutation(d)
    column_permutation = np.random.permutation(d)
    final_sparse_matrix = np.eye(d, dtype=dtype)[lines_permutation]
    final_sparse_matrix = final_sparse_matrix[:, column_permutation]
    return final_sparse_matrix

def create_sparse_matrix_pattern(shape, block_size, permutation=True):
    """
    Create a random mask for a sparse matrix with (+/- 1) `block_size` value in each line and each col.

    If you want exactly `block_size` value in each line and each col, shape must be square and block_size can divide shape
    :param shape: The shape of the matrix
    :param block_size:
    :return:
    """
    base_block_diag = create_random_block_diag(shape[0], shape[1], block_size, mask=True)
    if permutation:
        lines_permutation = np.random.permutation(shape[0])
        column_permutation = np.random.permutation(shape[1])
        final_sparse_matrix = base_block_diag[lines_permutation]
        final_sparse_matrix = final_sparse_matrix[:, column_permutation]
        return final_sparse_matrix
    else:
        return base_block_diag

def create_sparse_factorization_pattern(shape, block_size, nb_factors, permutation=True):
    min_dim = min(*shape)
    sparse_factors = [create_sparse_matrix_pattern((shape[0], min_dim), block_size, permutation)]
    for _ in range(nb_factors-2):
        sparse_factors.append(create_sparse_matrix_pattern((min_dim, min_dim), block_size, permutation))
    sparse_factors.append(create_sparse_matrix_pattern((min_dim, shape[1]), block_size, permutation))
    return sparse_factors


def cast_sparsity_pattern(sparsity_pattern):
    try:
        return np.array(sparsity_pattern)
    except:
        raise ValueError("Sparsity pattern isn't well formed")