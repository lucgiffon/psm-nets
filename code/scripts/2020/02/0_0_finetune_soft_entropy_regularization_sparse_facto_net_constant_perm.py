"""
This script finds a palminized model with given arguments then finetune it.

Usage:
    script.py [-h] [-v|-vv] --walltime int [--seed int] --input-dir path [--permutation-threshold float] [--sparsity-factor=int] [--nb-factor=intorstr] [--tb] [--add-entropies] [--param-reg-softmax-entropy=float] (--mnist|--svhn|--cifar10|--cifar100|--test-data) (--pbp-dense-layers --nb-units-dense-layer=str |--dense-layers --nb-units-dense-layer=str|--mnist-lenet|--test-model|--cifar10-vgg19|--cifar100-vgg19|--svhn-vgg19)

Options:
  -h --help                             Show this screen.
  -vv                                   Set verbosity to debug.
  -v                                    Set verbosity to info.
  --input-dir path                      Path to input directory where to find previously generated results.
  --walltime int                        The number of hour before training is stopped.
  --tb                                  Tell if tensorboard should be printed.
  --seed int                            Seed for random

Dataset:
  --mnist                               Use Mnist dataset.
  --svhn                                Use svhn dataset.
  --cifar10                             Use cifar10 dataset.
  --cifar100                            Use cifar100 dataset.
  --test-data                           Use test datasset (that is actually mnist).

Model:
  --mnist-lenet                         Use model lenet pretrained for mnist.
  --test-model                          Use test, small, model.
  --cifar10-vgg19                       Use model vgg19 pretrained on cifar10.
  --cifar100-vgg19                      Use model vgg19 pretrained on cifar100.
  --svhn-vgg19                          Use model vgg19 pretrained on svhn.
  --nb-units-dense-layer=str            Tells the number of hidden units in dense layers.
  --pbp-dense-layers                    tells to use pbp dense model.
  --dense-layers                        Tells to use simple dense model.

Sparsity options:
  --sparsity-factor=int                 Integer coefficient from which is computed the number of value in each factor.
  --nb-factor=intorstr                  Integer telling how many factors should be used or list of int telling for each layer the number of factor ("int,int,int").
  --param-reg-softmax-entropy=float     Float for the parameter of the softmax entropy
  --add-entropies                       Use addition instead of multiplication for regularization of permutations
  --permutation-threshold float         Tells what threshold to use for permutation fixation. [default: 0.5]
"""
import logging
import os
from collections import defaultdict
from copy import deepcopy

import pandas as pd
import sys
import scipy.special

import keras
import signal
import docopt

from keras.layers import Dense, InputLayer
from keras.models import Model

from palmnet.core.palminizable import Palminizable
from palmnet.data import Mnist, Test, Svhn, Cifar100, Cifar10
from palmnet.layers.pbp_layer import PBPDenseDensify, PBPDenseFixedPerm, PBPDenseFixedPermLookup, SparseFactorisationDensePBPFixed
from palmnet.layers.sparse_facto_dense_masked import SparseFactorisationDense
from palmnet.models import sparse_random_vgg19_model, sparse_random_lenet_model, pbp_lenet_model, create_pbp_model, create_dense_model
from palmnet.utils import timeout_signal_handler, get_sparsity_pattern, np_create_permutation_from_weight_matrix
from palmnet.experiments.utils import ResultPrinter, ParameterManagerRandomSparseFacto, ParameterManagerEntropyRegularization, ParameterManagerEntropyRegularizationFinetune
from skluc.utils import logger, log_memory_usage
import time
import numpy as np

lst_results_header = [
    "base_score",
    "finetuned_score",
    "nb_param",
    "nb_flop"
]


def replace_pbp_layers_with_pbp_fixed(model):
    """
    From a base model with PBP layers and trainable permutations: build the corresponding model with fixed permutations.

    :param model:
    :return:
    """
    # new_model = deepcopy(model)
    new_model = keras.models.clone_model(model)
    new_model.set_weights(model.get_weights())
    # new_model = model
    # lst_tpl_str_bool_new_model_layers = []  # this lsit will contain information on modified
    dct_new_layer_attr = defaultdict(lambda: {}) # this dict will contain the new layer object and layer weights, indexed by layer name

    arr_irow_todel = np.array([])  # depending on the previous layer, some rows of the next layer may be droped
    # in case a permutation isn't full (sum row/col = 0)
    for i, layer in enumerate(new_model.layers):
        logger.debug('Prepare layer {}'.format(layer.name))

        layer_name = layer.name
        if isinstance(layer, PBPDenseDensify):
            factors = []  # the PBP factors will be contained here
            block_diags = []
            permutations = []
            bias = None
            for weight_array, weight in zip(layer.get_weights(), layer.weights):
                # this loop constructs the list of factors
                weight_name = weight.name
                if "bias" in weight_name: # just store the bias if it exists
                    bias = weight_array
                    print("found bias")
                    continue

                if "block_diag" in weight_name:
                    factor = weight_array
                    # remove the lines specified by the deletion of columns in last permutation
                    factor = np.delete(factor, arr_irow_todel, axis=0)
                    arr_irow_todel = np.array([])
                    block_diags.append(factor)


                elif "permutation" in weight_name:  # special treatment for the permutation factor:
                    permutation_mat = np_create_permutation_from_weight_matrix(weight_array, paraman["--permutation-threshold"])
                    sum_col = np.sum(permutation_mat, axis=0)
                    sum_rows = np.sum(permutation_mat, axis=1)
                    idx_col_empty = np.where(sum_col == 0)[0]
                    # add idx to delete in next factor (useless feature because always zero -> not referenced in lookup)
                    arr_irow_todel = idx_col_empty

                    # idx_rows_non_empty = np.where(sum_rows == 1)[0]
                    # lookup_permutation = idx_rows_non_empty  # lookup only features that are not set to zero by the permutation
                    permutation_mat_pruned = np.delete(permutation_mat, idx_col_empty, axis=1)
                    permutations.append(permutation_mat_pruned)

                    # permutation_scaling = np.ones(permutation_mat_pruned.shape[-1])
                    factor = permutation_mat_pruned

                else:
                    raise ValueError("unknwon weight {}".format(weight_name))

                factors.append(factor)

            if bias is not None:
                bias = np.delete(bias, arr_irow_todel)

            # assert len(permutations) == (len(factors)+1), "There should be {}+1={} permutations. Found  {}".format(len(factors), len(factors) +1, len(permutations))
            sparsity_patterns = [get_sparsity_pattern(w) for w in factors]
            factor_weights = block_diags

            hidden_layer_dim = sparsity_patterns[-1].shape[-1]
            replacing_layer = SparseFactorisationDensePBPFixed(units=hidden_layer_dim, sparsity_patterns=sparsity_patterns, use_bias=layer.use_bias, activation=layer.activation, kernel_regularizer=layer.kernel_regularizer, use_scaling=False)
            # replacing_layer = SparseFactorisationDense(units=hidden_layer_dim, sparsity_patterns=sparsity_patterns, use_bias=layer.use_bias, activation=layer.activation, kernel_regularizer=layer.kernel_regularizer, use_scaling=False)
            # replacing_layer = PBPDenseFixedPermLookup(use_permutation_scaling=True, before_pruning_units=factors[-2].shape[-1], permutations=permutations, sparsity_factor=layer.sparsity_factor, use_bias=layer.use_bias, activation=layer.activation, kernel_regularizer=layer.kernel_regularizer)
            # replacing_layer = PBPDenseFixedPerm(use_permutation_scaling=True, before_pruning_units=factors[-2].shape[-1], permutations=permutations, sparsity_factor=layer.sparsity_factor, use_bias=layer.use_bias, activation=layer.activation, kernel_regularizer=layer.kernel_regularizer)
            replacing_weights = factor_weights + ([bias] if bias is not None else [])

        elif isinstance(layer, Dense):  # last classifcation layer (supposedly)
            replacing_layer = Dense(units=layer.units, use_bias=layer.use_bias, activation=layer.activation, kernel_regularizer=layer.kernel_regularizer)
            replacing_weights = None

        else:
            raise ValueError("Unknwon layer kind {}".format(layer.__class__.__name__))
        dct_new_layer_attr[layer_name]["layer_weights"] = replacing_weights
        dct_new_layer_attr[layer_name]["layer_obj"] = replacing_layer


    # dictionnary of dependencies between layers
    network_dict = {'input_layers_of': defaultdict(lambda: []), 'new_output_tensor_of': defaultdict(lambda: [])}

    # model should start with inputlayer
    if not isinstance(new_model.layers[0], InputLayer):
        new_model = Model(input=new_model.input, output=new_model.output)

    # Set the input layers of each layer
    for layer in new_model.layers:
        # each layer is set as `input` layer of all its outbound layers
        for node in layer._outbound_nodes:
            outbound_layer_name = node.outbound_layer.name
            network_dict['input_layers_of'].update({outbound_layer_name: [layer.name]})

    # Set the output tensor of the input layer
    network_dict['new_output_tensor_of'].update(
        {new_model.layers[0].name: new_model.input})

    for layer in new_model.layers[1:]:
        layer_name = layer.name

        layer_input = [network_dict['new_output_tensor_of'][layer_aux]
                       for layer_aux in network_dict['input_layers_of'][layer.name]]
        if len(layer_input) == 1:
            layer_input = layer_input[0]

        proxy_new_layer_attr = dct_new_layer_attr[layer_name]


        x = layer_input

        new_layer = proxy_new_layer_attr["layer_obj"]
        new_layer.name = '{}_{}'.format(layer.name,
                                        new_layer.name)
        # try:
        x = new_layer(x)
        # except:
        #     print(1)
        if proxy_new_layer_attr["layer_weights"] is not None: new_layer.set_weights(proxy_new_layer_attr["layer_weights"])
        logger.info('Layer {} modified into {}'.format(layer.name, new_layer.name))

        network_dict['new_output_tensor_of'].update({layer.name: x})

        new_model = Model(inputs=new_model.inputs, outputs=x)

    return new_model


def main():

    # (x_train, y_train), (x_test, y_test) = paraman.get_dataset()
    data_obj = paraman.get_dataset()
    (x_train, y_train), (x_test, y_test) = data_obj.load_data()


    if paraman["--mnist"]:
        if paraman["--dense-layers"] or paraman["--pbp-dense-layers"]:
            param_train_dataset = Mnist.get_model_param_training("mnist_500")
        else:
            param_train_dataset = Mnist.get_model_param_training()
    elif paraman["--svhn"]:
        param_train_dataset = Svhn.get_model_param_training()
    elif paraman["--cifar10"]:
        param_train_dataset = Cifar10.get_model_param_training()
    elif paraman["--cifar100"]:
        param_train_dataset = Cifar100.get_model_param_training()
    elif paraman["--test-data"]:
        param_train_dataset = Test.get_model_param_training()
    else:
        raise ValueError("Unknown dataset")


    if paraman["--dense-layers"] or paraman["--pbp-dense-layers"]:
        x_train = x_train.reshape(x_train.shape[0], np.prod(data_obj.shape))
        x_test = x_test.reshape(x_test.shape[0], np.prod(data_obj.shape))

    if paraman["--pbp-dense-layers"]:
        optimizer = keras.optimizers.Adam(lr=0.01)
    else:
        optimizer = param_train_dataset.optimizer

    if os.path.exists(paraman["output_file_notfinishedprinter"]) and os.path.exists(paraman["output_file_modelprinter"]):
        df = pd.read_csv(paraman["output_file_resprinter"])
        try:
            init_nb_epoch = len(pd.read_csv(paraman["output_file_csvcbprinter"]))
        except Exception as e:
            logger.error("Caught exception while reading csv history: {}".format(str(e)))
            init_nb_epoch = 0
        base_score = float(df["base_score"])
        before_finetuning_score = float(df["before_finetuning_score"])
        new_model = keras.models.load_model(paraman["output_file_modelprinter"],custom_objects={"SparseFactorisationDense": SparseFactorisationDense})
        # nb_flop_model = int(df["nb_flop"])
        traintime = int(df["traintime"])

    else:
        init_nb_epoch = 0
        traintime = 0

        base_model = keras.models.load_model(paraman["input_model_path"], custom_objects={"PBPDenseDensify": PBPDenseDensify})
        new_model = replace_pbp_layers_with_pbp_fixed(base_model)
        base_model.compile(loss=param_train_dataset.loss,
                                 optimizer=optimizer,
                                 metrics=['categorical_accuracy'])
        new_model.compile(loss=param_train_dataset.loss,
                                 optimizer=optimizer,
                                 metrics=['categorical_accuracy'])
        base_score = base_model.evaluate(x_test, y_test, verbose=1)[1]
        before_finetuning_score = new_model.evaluate(x_test, y_test, verbose=1)[1]

        print(base_score)

        # results must be already printed once in case process is killed afterward
    dct_results = {
        "finetuned_score": None,
        "base_score": base_score,
        "before_finetuning_score": before_finetuning_score,
        "traintime": traintime
    }
    resprinter.add(dct_results)
    resprinter.print()

    new_model.summary()

    call_backs = []

    model_checkpoint_callback = keras.callbacks.ModelCheckpoint(str(paraman["output_file_modelprinter"]), monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)
    call_backs.append(model_checkpoint_callback)
    if paraman["--tb"]:
        tbCallBack = keras.callbacks.TensorBoard(log_dir=str(paraman["output_file_tensorboardprinter"]), histogram_freq=20, write_graph=False, write_images=False, batch_size=param_train_dataset.batch_size, write_grads=True, update_freq="epoch")
        call_backs.append(tbCallBack)
    csvcallback = keras.callbacks.callbacks.CSVLogger(str(paraman["output_file_csvcbprinter"]), separator=',', append=True)
    call_backs.append(csvcallback)


    signal.signal(signal.SIGALRM, timeout_signal_handler)
    signal.alarm(int(paraman["--walltime"] * 3600))  # start alarm
    finetuned_score = None
    try:
        open(paraman["output_file_notfinishedprinter"], 'w').close()
        start = time.time()
        datagen_obj = param_train_dataset.image_data_generator.flow(x_train, y_train, batch_size=param_train_dataset.batch_size)
        history = new_model.fit_generator(
           datagen_obj,
           # epochs=5,
           epochs=param_train_dataset.epochs - init_nb_epoch,
           # epochs=2 - init_nb_epoch,
           verbose=2,
           validation_data=(x_test, y_test),
           callbacks=param_train_dataset.callbacks + call_backs)
        signal.alarm(0)  # stop alarm for next evaluation

        finetuned_score = new_model.evaluate(x_test, y_test, verbose=1)[1]
        print(finetuned_score)

        if os.path.exists(paraman["output_file_notfinishedprinter"]):
            os.remove(paraman["output_file_notfinishedprinter"])
    # except TimeoutError as te:
    except Exception as e:
        logging.error("Caught exception: {}".format(e))
        finetuned_score = None
    finally:
        stop = time.time()
        traintime += stop-start
        dct_results = {
            "finetuned_score": finetuned_score,
            "base_score": base_score,
            "traintime": traintime,
            "before_finetuning_score": before_finetuning_score
            # "nb_flop": nb_flop_model,
            # "nb_param": nb_param_model,
        }
        new_model.save(str(paraman["output_file_modelprinter"]))
        resprinter.add(dct_results)


if __name__ == "__main__":
    logger.info("Command line: " + " ".join(sys.argv))
    log_memory_usage("Memory at startup")
    arguments = docopt.docopt(__doc__)
    paraman = ParameterManagerEntropyRegularizationFinetune(arguments)
    initialized_results = dict((v, None) for v in lst_results_header)
    resprinter = ResultPrinter(output_file=paraman["output_file_resprinter"])
    resprinter.add(initialized_results)
    resprinter.add(paraman)
    if paraman["-v"] >= 2:
        logger.setLevel(level=logging.DEBUG)
    elif paraman["-v"] >= 1:
        logger.setLevel(level=logging.INFO)
    else:
        logger.setLevel(level=logging.WARNING)

    logger.warning("Verbosity set to warning")
    logger.info("Verbosity set to info")
    logger.debug("Verbosity set to debug")

    if not os.path.exists(paraman["output_file_notfinishedprinter"]) and \
        os.path.exists(paraman["output_file_resprinter"]) and \
        os.path.exists(paraman["output_file_modelprinter"]):
        sys.exit("Expe {} already executed. Exit".format(paraman["hash"]))

    has_failed = False
    try:
        main()
    except Exception as e:
        has_failed = True
        raise e

    finally:
        failure_dict = {
            "failure": has_failed
        }

        resprinter.add(failure_dict)
        resprinter.print()