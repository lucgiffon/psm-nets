import csv
import warnings

from typing import Iterable
import io
from collections import OrderedDict

import re
import six
import os
from keras.callbacks import Callback
from keras.engine import InputLayer
from keras.layers import Dense, Conv2D, MaxPooling2D, BatchNormalization, Flatten, Add, Activation, Dropout
from keras.models import Model
from pathlib import Path
import numpy as np
import keras.backend as K
from collections import defaultdict
import tensorflow as tf

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

class SafeModelCheckpoint(Callback):
    """Save the model after every epoch.

    `filepath` can contain named formatting options,
    which will be filled with the values of `epoch` and
    keys in `logs` (passed in `on_epoch_end`).

    For example: if `filepath` is `weights.{epoch:02d}-{val_loss:.2f}.hdf5`,
    then the model checkpoints will be saved with the epoch number and
    the validation loss in the filename.

    # Arguments
        filepath: string, path to save the model file.
        monitor: quantity to monitor.
        verbose: verbosity mode, 0 or 1.
        save_best_only: if `save_best_only=True`,
            the latest best model according to
            the quantity monitored will not be overwritten.
        save_weights_only: if True, then only the model's weights will be
            saved (`model.save_weights(filepath)`), else the full model
            is saved (`model.save(filepath)`).
        mode: one of {auto, min, max}.
            If `save_best_only=True`, the decision
            to overwrite the current save file is made
            based on either the maximization or the
            minimization of the monitored quantity. For `val_acc`,
            this should be `max`, for `val_loss` this should
            be `min`, etc. In `auto` mode, the direction is
            automatically inferred from the name of the monitored quantity.
        period: Interval (number of epochs) between checkpoints.
    """

    def __init__(self, filepath, monitor='val_loss', verbose=0,
                 save_best_only=False, save_weights_only=False,
                 mode='auto', period=1):
        super(SafeModelCheckpoint, self).__init__()
        self.monitor = monitor
        self.verbose = verbose
        self.filepath = filepath
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.period = period
        self.epochs_since_last_save = 0

        if mode not in ['auto', 'min', 'max']:
            warnings.warn('ModelCheckpoint mode %s is unknown, '
                          'fallback to auto mode.' % (mode),
                          RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            filepath = self.filepath.format(epoch=epoch + 1, **logs)
            if self.save_best_only:
                current = logs.get(self.monitor)
                if current is None:
                    warnings.warn('Can save best model only with %s available, '
                                  'skipping.' % (self.monitor), RuntimeWarning)
                else:
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s improved from %0.5f to %0.5f,'
                                  ' saving model to %s'
                                  % (epoch + 1, self.monitor, self.best,
                                     current, filepath))
                        self.best = current
                        self.__save_model(filepath)
                    else:
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s did not improve from %0.5f' %
                                  (epoch + 1, self.monitor, self.best))
            else:
                if self.verbose > 0:
                    print('\nEpoch %05d: saving model to %s' % (epoch + 1, filepath))
                self.__save_model(filepath)

    def __save_model(self, filepath):
        tmp_filepath_name = filepath + ".tmp"
        if self.save_weights_only:
            self.model.save_weights(tmp_filepath_name, overwrite=True)
            os.replace(tmp_filepath_name, filepath)
        else:
            self.model.save(tmp_filepath_name, overwrite=True)
            os.replace(tmp_filepath_name, filepath)

def get_idx_last_layer_of_class(model, class_=Dense):
    idx_last_layer_of_class = -1
    for i, layer in enumerate(model.layers):
        if isinstance(layer, class_):
            idx_last_layer_of_class = i
    if idx_last_layer_of_class == -1:
        logger.warning(f"No layer of class {class_.__name__} found")
    return idx_last_layer_of_class

def get_idx_first_layer_of_class(model, class_=Conv2D):
    idx_first_layer_of_class = -1
    for i, layer in enumerate(model.layers):
        if isinstance(layer, class_):
            idx_first_layer_of_class = i
            break
    if idx_first_layer_of_class == -1:
        logger.warning(f"No layer of class {class_.__name__} found")
    return idx_first_layer_of_class


DCT_CHANNEL_PREDEFINED_FACTORIZATIONS = {
    10: [2, 5],
    100: [2, 5, 10],
    6: [2, 3],
    16: [2, 2, 4],
    3: [3]
}

def get_facto_for_channel_and_order(channel, order, dct_predefined_facto=None):
    """
    Return a factorisation of channel in `order` elements (the number of factors)

    :param channel:
    :param order:
    :param dct_predefined_facto:
    :return: the list of factors whose product equal to channel and of len  `order`
    """
    if dct_predefined_facto is not None and channel in dct_predefined_facto:
        predef_facto = dct_predefined_facto[channel]
        missing_elm = order - len(predef_facto)
        facto = [1]*missing_elm + predef_facto
        return np.array(facto)
    elif int(np.log2(channel)) != np.log2(channel):
        raise ValueError("Channel must be a power of two. {}".format(channel))
    else:
        base = [1] * order
        base = np.array(base)
        _prod = np.prod(base)  # == 1
        next_idx_to_upscale = 0
        while _prod < channel:
            base[next_idx_to_upscale] *= 2
            _prod = np.prod(base)
            next_idx_to_upscale = int((next_idx_to_upscale + 1) % order)

        return base[::-1]  # biggest values at the end for no reason :)

def build_dct_tt_ranks(model, rank_value=2, order=4):
    tt_ranks_conv = [rank_value] * order + [1]
    tt_ranks_dense = [rank_value] * order + [1]
    tt_ranks_dense[0] = 1

    dct_layer_params = defaultdict(lambda: dict())
    for layer in model.layers:
        if isinstance(layer, Conv2D):
            dct_layer_params[layer.name]["tt_ranks"] = tt_ranks_conv
        elif isinstance(layer, Dense):
            dct_layer_params[layer.name]["tt_ranks"] = tt_ranks_dense
        else:
            dct_layer_params[layer.name] = None

    return dct_layer_params

def get_nb_non_zero_values(lst_weights):
    count = 0
    for w in lst_weights:
        count += np.sum(w.astype(bool))
    return count

def get_cumsum_size_weights(lst_weights, nnz=False):
    if not nnz:
        count = 0
        for w in lst_weights:
                count += w.size
        return count
    else:
        return get_nb_non_zero_values(lst_weights)

def get_nb_learnable_weights(layer, nnz=False):
    from palmnet.layers.sparse_facto_conv2D_masked import SparseFactorisationConv2D
    from palmnet.layers.sparse_facto_dense_masked import SparseFactorisationDense
    from palmnet.layers.tucker_layer_sparse_facto import TuckerSparseFactoLayerConv
    from palmnet.layers.fastfood_layer_dense import FastFoodLayerDense

    if isinstance(layer, SparseFactorisationConv2D) or isinstance(layer, SparseFactorisationDense):
        sp_patterns = layer.sparsity_patterns
        assert sp_patterns is not None, f"No sparsity pattern found in layer {layer.name}"
        count_sparsity_patterns = get_nb_non_zero_values(sp_patterns)
        if layer.use_bias:
            count_bias = layer.filters if isinstance(layer, SparseFactorisationConv2D) else layer.units
        else:
            count_bias = 0
        if layer.use_scaling:
            count_scaling = 1
        else:
            count_scaling = 0
        return count_sparsity_patterns + count_bias + count_scaling
    elif isinstance(layer, TuckerSparseFactoLayerConv):
        # reuse code above
        return get_nb_learnable_weights(layer.in_factor, nnz) + get_nb_learnable_weights(layer.core, nnz) + get_nb_learnable_weights(layer.out_factor, nnz)

    # elif isinstance(layer, FastFoodLayer):  # this is not necessary because the Hadamard matrix isn't a weight but a constant
    #     total_nb_weights = get_cumsum_size_weights(layer.get_weights())
    #     hadamard_weights = layer.final_dim ** 2
    #     actual_nb_weights = total_nb_weights - hadamard_weights
    #     return actual_nb_weights

    else:
        return get_cumsum_size_weights(layer.get_weights(), nnz)

def get_nb_learnable_weights_from_model(model, nnz=False):
    count_learnable_weights = 0
    for layer in model.layers:
        count_learnable_weights += get_nb_learnable_weights(layer, nnz)
    return count_learnable_weights

class TensortrainBadRankException(Exception):
    def __init__(self, expected_ranks, obtained_ranks, *args):
        super().__init__(*args)
        self.expected_ranks = expected_ranks
        self.obtained_ranks = obtained_ranks

    def __str__(self):
        return f"{self.__class__.__name__}: Got rank {self.obtained_ranks} when expected was {self.expected_ranks}"

def sparse_facto_init(shape, idx_fac, sparsity_pattern, multiply_left=False):
    if idx_fac == 0:
        # sum of sparsity_pattern[i] is norm zero of sparsity pattern
        # we use 2 on the numerator because the input has gone through relu and then is not 0 mean -> he initialisation
        kernel_init = lambda *args, **kwargs: np.random.randn(*shape) * np.sqrt(2 / np.sum(sparsity_pattern, axis=int(multiply_left))) * sparsity_pattern
    else:
        # we use 1 on the numerator because between factor the "activations" (linear) are expected to have 0 mean -> xavier initialisation
        kernel_init = lambda *args, **kwargs: np.random.randn(*shape) * np.sqrt(1 / np.sum(sparsity_pattern, axis=int(multiply_left))) * sparsity_pattern

    return kernel_init

NAME_INIT_SPARSE_FACTO = "sparse_facto_var_1"

def translate_keras_to_tf_model(model):
    """
    Convert keras model into tf.keras model. It works only with networks that have only one input

    :param model:
    :return:
    """
    dct_keras_tf_layer = {
        Conv2D.__name__: tf.keras.layers.Conv2D,
        Dense.__name__: tf.keras.layers.Dense,
        MaxPooling2D.__name__: tf.keras.layers.MaxPooling2D,
        BatchNormalization.__name__: tf.keras.layers.BatchNormalization,
        Add.__name__: tf.keras.layers.Add,
        Flatten.__name__: tf.keras.layers.Flatten,
        Activation.__name__: tf.keras.layers.Activation,
        Dropout.__name__: tf.keras.layers.Dropout
    }

    if not isinstance(model.layers[0], InputLayer):
        model = Model(input=model.input, output=model.output)

    network_dict = {'input_layers_of': defaultdict(lambda: []), 'new_output_tensor_of': defaultdict(lambda: [])}

    tf_equivalent_input_layer = tf.keras.layers.InputLayer(**model.layers[0].get_config())

    # Set the output tensor of the input layer
    network_dict['new_output_tensor_of'].update(
        {model.layers[0].name: tf_equivalent_input_layer.input})

    for i, layer in enumerate(model.layers):
        # each layer is set as `input` layer of all its outbound layers
        for node in layer._outbound_nodes:
            outbound_layer_name = node.outbound_layer.name
            network_dict['input_layers_of'][outbound_layer_name].append(layer.name)


    for i, layer in enumerate(model.layers[1:]):

        # get all layers input
        layer_inputs = [network_dict['new_output_tensor_of'][curr_layer_input] for curr_layer_input in network_dict['input_layers_of'][layer.name]]
        if len(layer_inputs) == 1:
            layer_inputs = layer_inputs[0]

        replacing_layer = dct_keras_tf_layer[layer.__class__.__name__](**layer.get_config())
        replacing_weights = layer.get_weights()

        x = replacing_layer(layer_inputs)
        replacing_layer.set_weights(replacing_weights)

        network_dict['new_output_tensor_of'].update({layer.name: x})

    model = tf.keras.models.Model(inputs=tf_equivalent_input_layer.input, outputs=x)

    return model

class DummyWith:
    def __enter__(self):
        return None

    def __exit__(self):
        return None
