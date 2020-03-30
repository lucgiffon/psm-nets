import csv

from typing import Iterable
import io
from collections import OrderedDict

import re
import six
import os
from keras.callbacks import Callback
from keras.engine import InputLayer
from keras.layers import Dense
from keras.models import Model
from pathlib import Path
import numpy as np
import keras.backend as K
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
    return arr.astype(bool).astype(float)


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

def get_lr_metric(optimizer):
    def lr(y_true, y_pred):
        return optimizer.lr
    return lr

class CSVLoggerByBatch(Callback):
    """Callback that streams epoch results to a csv file.

    Supports all values that can be represented as a string,
    including 1D iterables such as np.ndarray.

    # Example

    ```python
    csv_logger = CSVLogger('training.log')
    model.fit(X_train, Y_train, callbacks=[csv_logger])
    ```

    # Arguments
        filename: filename of the csv file, e.g. 'run/log.csv'.
        separator: string used to separate elements in the csv file.
        append: True: append if file exists (useful for continuing
            training). False: overwrite existing file,
    """

    def __init__(self, filename, n_batch_between_display=1, separator=',', append=False):
        self.sep = separator
        self.filename = filename
        self.append = append
        self.writer = None
        self.keys = None
        self.append_header = True
        self.n_batch_between_display = n_batch_between_display
        if six.PY2:
            self.file_flags = 'b'
            self._open_args = {}
        else:
            self.file_flags = ''
            self._open_args = {'newline': '\n'}
        super(CSVLoggerByBatch, self).__init__()
        self.epoch = 1
        self.global_step = 0
        self.last_batch = -1

    def on_train_begin(self, logs=None):
        if self.append:
            if os.path.exists(self.filename):
                with open(self.filename, 'r' + self.file_flags) as f:
                    self.append_header = not bool(len(f.readline()))
            mode = 'a'
        else:
            mode = 'w'
        self.csv_file = io.open(self.filename,
                                mode + self.file_flags,
                                **self._open_args)

    def on_batch_end(self, batch, logs=None):
        logs = logs or {}
        self.global_step += 1
        if batch < self.last_batch:
            self.epoch += 1

        def handle_value(k):
            is_zero_dim_ndarray = isinstance(k, np.ndarray) and k.ndim == 0
            if isinstance(k, six.string_types):
                return k
            elif isinstance(k, Iterable) and not is_zero_dim_ndarray:
                return '"[%s]"' % (', '.join(map(str, k)))
            else:
                return k

        if self.keys is None:
            self.keys = sorted(logs.keys())

        if self.model.stop_training:
            # We set NA so that csv parsers do not fail for this last epoch.
            logs = dict([(k, logs[k] if k in logs else 'NA') for k in self.keys])

        if not self.writer:
            class CustomDialect(csv.excel):
                delimiter = self.sep
            fieldnames = ['epoch'] + self.keys
            self.writer = csv.DictWriter(self.csv_file,
                                         fieldnames=fieldnames,
                                         dialect=CustomDialect)
            if self.append_header:
                self.writer.writeheader()

        row_dict = OrderedDict({'epoch': self.epoch})
        row_dict.update((key, handle_value(logs[key])) for key in self.keys)
        if self.global_step % self.n_batch_between_display == 0:
            self.writer.writerow(row_dict)
            self.csv_file.flush()

        self.last_batch = batch

    def on_train_end(self, logs=None):
        self.csv_file.close()
        self.writer = None

    def __del__(self):
        if hasattr(self, 'csv_file') and not self.csv_file.closed:
            self.csv_file.close()



class CyclicLR(Callback):
    """This callback implements a cyclical learning rate policy (CLR).
    The method cycles the learning rate between two boundaries with
    some constant frequency.
    # Arguments
        base_lr: initial learning rate which is the
            lower boundary in the cycle.
        max_lr: upper boundary in the cycle. Functionally,
            it defines the cycle amplitude (max_lr - base_lr).
            The lr at any cycle is the sum of base_lr
            and some scaling of the amplitude; therefore
            max_lr may not actually be reached depending on
            scaling function.
        step_size: number of training iterations per
            half cycle. Authors suggest setting step_size
            2-8 x training iterations in epoch.
        mode: one of {triangular, triangular2, exp_range}.
            Default 'triangular'.
            Values correspond to policies detailed above.
            If scale_fn is not None, this argument is ignored.
        gamma: constant in 'exp_range' scaling function:
            gamma**(cycle iterations)
        scale_fn: Custom scaling policy defined by a single
            argument lambda function, where
            0 <= scale_fn(x) <= 1 for all x >= 0.
            mode paramater is ignored
        scale_mode: {'cycle', 'iterations'}.
            Defines whether scale_fn is evaluated on
            cycle number or cycle iterations (training
            iterations since start of cycle). Default is 'cycle'.

    The amplitude of the cycle can be scaled on a per-iteration or
    per-cycle basis.
    This class has three built-in policies, as put forth in the paper.
    "triangular":
        A basic triangular cycle w/ no amplitude scaling.
    "triangular2":
        A basic triangular cycle that scales initial amplitude by half each cycle.
    "exp_range":
        A cycle that scales initial amplitude by gamma**(cycle iterations) at each
        cycle iteration.
    For more detail, please see paper.

    # Example for CIFAR-10 w/ batch size 100:
        ```python
            clr = CyclicLR(base_lr=0.001, max_lr=0.006,
                                step_size=2000., mode='triangular')
            model.fit(X_train, Y_train, callbacks=[clr])
        ```

    Class also supports custom scaling functions:
        ```python
            clr_fn = lambda x: 0.5*(1+np.sin(x*np.pi/2.))
            clr = CyclicLR(base_lr=0.001, max_lr=0.006,
                                step_size=2000., scale_fn=clr_fn,
                                scale_mode='cycle')
            model.fit(X_train, Y_train, callbacks=[clr])
        ```

    # References

      - [Cyclical Learning Rates for Training Neural Networks](
      https://arxiv.org/abs/1506.01186)
    """

    def __init__(
            self,
            base_lr=0.001,
            max_lr=0.006,
            step_size=2000.,
            mode='triangular',
            gamma=1.,
            scale_fn=None,
            scale_mode='cycle',
            logrange=False):
        super(CyclicLR, self).__init__()

        if mode not in ['triangular', 'triangular2',
                        'exp_range']:
            raise KeyError("mode must be one of 'triangular', "
                           "'triangular2', or 'exp_range'")
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = step_size
        self.mode = mode
        self.gamma = gamma
        self.logrange = logrange

        if logrange:
            self.range = np.geomspace(self.base_lr, self.max_lr, num=int(self.step_size))
        else:
            self.range = np.linspace(self.base_lr, self.max_lr, num=int(self.step_size))

        if scale_fn is None:
            if self.mode == 'triangular':
                self.scale_fn = lambda x: 1.
                self.scale_mode = 'cycle'
            elif self.mode == 'triangular2':
                self.scale_fn = lambda x: 1 / (2.**(x - 1))
                self.scale_mode = 'cycle'
            elif self.mode == 'exp_range':
                self.scale_fn = lambda x: gamma ** x
                self.scale_mode = 'iterations'
        else:
            self.scale_fn = scale_fn
            self.scale_mode = scale_mode
        self.clr_iterations = 0.
        self.trn_iterations = 0.
        self.history = {}

        self._reset()

    def _reset(self, new_base_lr=None, new_max_lr=None,
               new_step_size=None):
        """Resets cycle iterations.
        Optional boundary/step size adjustment.
        """
        if new_base_lr is not None:
            self.base_lr = new_base_lr
        if new_max_lr is not None:
            self.max_lr = new_max_lr
        if new_step_size is not None:
            self.step_size = new_step_size
        self.clr_iterations = 0.

    def clr(self):
        cycle = np.floor(1 + self.clr_iterations / (2 * self.step_size))
        x = np.abs(self.clr_iterations / self.step_size - 2 * cycle + 1)
        # x = np.abs(self.clr_iterations / self.step_size - 1)
        # x = np.exp(x - 1) - 1
        # x = self.range[int(self.clr_iterations % (self.step_size)) -1]
        if self.clr_iterations % (2*self.step_size) < self.step_size:
            idx = int(self.clr_iterations % (self.step_size))
        else:
            idx = -int(self.clr_iterations % (self.step_size) + 1)

        lr_val = self.range[idx]

        if self.scale_mode == 'cycle':
            # print(idx)
            # val = self.base_lr + (self.max_lr - self.base_lr) * \
            #     np.maximum(0, (1 - x)) * self.scale_fn(cycle)
            return lr_val * self.scale_fn(cycle)
        else:
            return lr_val * self.scale_fn(self.clr_iterations)
            # return self.base_lr + (self.max_lr - self.base_lr) * \
            #     np.maximum(0, (1 - x)) * self.scale_fn(self.clr_iterations)

    def on_train_begin(self, logs={}):
        logs = logs or {}

        if self.clr_iterations == 0:
            K.set_value(self.model.optimizer.lr, self.base_lr)
        else:
            K.set_value(self.model.optimizer.lr, self.clr())

    def on_batch_end(self, epoch, logs=None):

        logs = logs or {}
        logs['lr'] = K.get_value(self.model.optimizer.lr)

        self.trn_iterations += 1
        self.clr_iterations += 1
        val = self.clr()
        K.set_value(self.model.optimizer.lr, val)

        # print(val, K.get_value(self.model.optimizer.lr))

        self.history.setdefault(
            'lr', []).append(
            K.get_value(
                self.model.optimizer.lr))
        self.history.setdefault('iterations', []).append(self.trn_iterations)

        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)



    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs['lr'] = K.get_value(self.model.optimizer.lr)


def get_idx_last_dense_layer(model):
    idx_last_dense_layer = -1
    for i, layer in enumerate(model.layers):
        if isinstance(layer, Dense):
            idx_last_dense_layer = i
    if idx_last_dense_layer == -1:
        logger.warning("No dense layer found")
    return idx_last_dense_layer