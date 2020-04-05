"""
This script finds a palminized model with given arguments then finetune it.

Usage:
    script.py --input-dir path [-h] [-v|-vv] --walltime int [--only-mask] [--tb] (--mnist|--svhn|--cifar10|--cifar100|--test-data) [--mnist-500|--mnist-lenet|--test-model|--cifar10-vgg19|--cifar100-vgg19|--svhn-vgg19] --sparsity-factor=int [--nb-iteration-palm=int] [--delta-threshold=float] [--hierarchical] [--nb-factor=int]

Options:
  -h --help                             Show this screen.
  -vv                                   Set verbosity to debug.
  -v                                    Set verbosity to info.
  --input-dir path                      Path to input directory where to find previously generated results.
  --walltime int                        The number of hour before training is stopped.
  --tb                                  Tell if tensorboard should be printed.

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
  --mnist-500                           Use model fc 500 hidden units pretrained on mnist.

Palm-Specifc options:
  --sparsity-factor=int                 Integer coefficient from which is computed the number of value in each factor.
  --nb-iteration-palm=int               Number of iterations in the inner palm4msa calls. [default: 300]
  --delta-threshold=float               Threshold value before stopping palm iterations. [default: 1e-6]
  --hierarchical                        Tells if palm should use the hierarchical euristic or not. Muhc longer but better approximation results.
  --nb-factor=int                       Tells the number of sparse factor for palm
  --only-mask                           Use only sparsity mask given by palm but re-initialize weights.
"""
import logging
import os
import pickle
import pandas as pd
import sys
from collections import defaultdict

import time
from copy import deepcopy
import keras
from keras.engine import Model, InputLayer
import signal
import docopt
from scipy.sparse import coo_matrix

from palmnet.core.palminizer import Palminizer
from palmnet.core.palminizable import Palminizable
from palmnet.data import Mnist, Test, Svhn, Cifar100, Cifar10
# from palmnet.layers.sparse_tensor import SparseFactorisationDense#, SparseFactorisationConv2DDensify
from palmnet.layers.sparse_masked import SparseFactorisationDense, SparseFactorisationConv2DDensify
from palmnet.utils import get_sparsity_pattern, insert_layer_nonseq, timeout_signal_handler
from palmnet.experiments.utils import ParameterManagerPalminize, ParameterManagerPalminizeFinetune, ResultPrinter
from skluc.utils import logger, log_memory_usage
from keras.layers import Dense, Conv2D
import numpy as np

lst_results_header = [
    "test_accuracy_finetuned_model"
]

def replace_layers_with_sparse_facto(model, dct_name_facto):
    new_model = deepcopy(model)
    lst_tpl_str_bool_new_model_layers = []
    dct_new_layer_attr = defaultdict(lambda: {})
    for i, layer in enumerate(new_model.layers):
        layer_name = layer.name
        sparse_factorization = dct_name_facto[layer_name]
        logger.debug('Prepare layer {}'.format(layer.name))
        if sparse_factorization != (None, None):
            # scaling = 1.
            if paraman["--only-mask"]:
                scaling = []
            else:
                scaling = [np.array(sparse_factorization[0])[None]]
            # factors_sparse = [coo_matrix(fac.toarray()) for fac in sparse_factorization[1].get_list_of_factors()]
            factors = [fac.toarray() for fac in sparse_factorization[1].get_list_of_factors()]
            # sparsity_patterns = [get_sparsity_pattern(w.toarray()) for w in factors]
            sparsity_patterns = [get_sparsity_pattern(w) for w in factors]
            # factor_data_sparse = [f.data for f in factors_sparse]
            factor_data = factors

            # create new layer
            if isinstance(layer, Dense):
                hidden_layer_dim = layer.units
                activation = layer.activation
                regularizer = layer.kernel_regularizer
                replacing_layer = SparseFactorisationDense(use_scaling=not paraman["--only-mask"], units=hidden_layer_dim, sparsity_patterns=sparsity_patterns, use_bias=layer.use_bias, activation=activation, kernel_regularizer=regularizer)
                replacing_weights = scaling + factor_data + [layer.get_weights()[-1]] if layer.use_bias else []
                # new_model = insert_layer_nonseq(new_model, layer_name, lambda: replacing_layer, position="replace")
                # replacing_layer.set_weights(replacing_weights)

            elif isinstance(layer, Conv2D):
                nb_filters = layer.filters
                kernel_size = layer.kernel_size
                activation = layer.activation
                padding = layer.padding
                regularizer = layer.kernel_regularizer
                replacing_layer = SparseFactorisationConv2DDensify(use_scaling=not paraman["--only-mask"], filters=nb_filters, kernel_size=kernel_size, sparsity_patterns=sparsity_patterns, use_bias=layer.use_bias, activation=activation, padding=padding, kernel_regularizer=regularizer)
                replacing_weights = scaling + factor_data + [layer.get_weights()[-1]] if layer.use_bias else []
                # new_model = insert_layer_nonseq(new_model, layer_name, lambda: replacing_layer, position="replace")
                # replacing_layer.set_weights(replacing_weights)

            else:
                raise ValueError("unknown layer class")

            dct_new_layer_attr[layer_name]["layer_weights"] = replacing_weights
            dct_new_layer_attr[layer_name]["sparsity_pattern"] = sparsity_patterns
            dct_new_layer_attr[layer_name]["layer_obj"] = replacing_layer
            dct_new_layer_attr[layer_name]["modified"] = True

            lst_tpl_str_bool_new_model_layers.append((layer_name, True))
        else:
            dct_new_layer_attr[layer_name]["modified"] = False
            lst_tpl_str_bool_new_model_layers.append((layer_name, False))
            dct_new_layer_attr[layer_name]["layer_obj"] = layer

    network_dict = {'input_layers_of': defaultdict(lambda: []), 'new_output_tensor_of': defaultdict(lambda: [])}

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

        if proxy_new_layer_attr["modified"]:
            x = layer_input

            new_layer = proxy_new_layer_attr["layer_obj"] # type: keras.layers.Layer
            new_layer.name = '{}_{}'.format(layer.name,
                                            new_layer.name)
            x = new_layer(x)
            if not paraman["--only-mask"]:
                new_layer.set_weights(proxy_new_layer_attr["layer_weights"])
            else:
                masked_weights = []
                i = 0
                for w in new_layer.get_weights():
                    if len(w.shape) > 1:
                        new_weight = w * proxy_new_layer_attr["sparsity_pattern"][i]
                        i += 1
                    else:
                        new_weight = w
                    masked_weights.append(new_weight)
                new_layer.set_weights(masked_weights)

            logger.info('Layer {} modified into {}'.format(layer.name, new_layer.name))
        else:
            x = layer(layer_input)
            logger.debug('Layer {} unmodified'.format(layer.name))

        network_dict['new_output_tensor_of'].update({layer.name: x})

        new_model = Model(inputs=new_model.inputs, outputs=x)

    return new_model

def main():

    if paraman["--mnist-lenet"]:
        param_train_dataset = Mnist.get_model_param_training()
    elif paraman["--mnist-500"]:
        param_train_dataset = Mnist.get_model_param_training("mnist_500")
    elif paraman["--cifar10-vgg19"]:
        param_train_dataset = Cifar10.get_model_param_training()
    elif paraman["--cifar100-vgg19"]:
        param_train_dataset = Cifar100.get_model_param_training()
    elif paraman["--svhn-vgg19"]:
        param_train_dataset = Svhn.get_model_param_training()
    elif paraman["--test-model"]:
        param_train_dataset = Test.get_model_param_training()
    else:
        raise NotImplementedError("No dataset specified.")

    (x_train, y_train), (x_test, y_test) = paraman.get_dataset().load_data()

    if paraman["--mnist-500"]:
        x_test = np.reshape(x_test, (-1, 784))
        x_train = np.reshape(x_train, (-1, 784))

    if os.path.exists(paraman["output_file_notfinishedprinter"]):
        df = pd.read_csv(paraman["output_file_resprinter"])
        init_nb_epoch = len(pd.read_csv(paraman["output_file_csvcbprinter"]))
        base_score = float(df["base_score"])
        before_finetuned_score = float(df["before_finetuned_score"])
        palminized_score = float(df["palminized_score"])
        fine_tuned_model = keras.models.load_model(paraman["output_file_modelprinter"],custom_objects={'SparseFactorisationConv2DDensify':SparseFactorisationConv2DDensify,
                                                                            "SparseFactorisationDense": SparseFactorisationDense})

    else:
        init_nb_epoch = 0

        mypalminizedmodel = pickle.load(open(paraman["input_model_path"], "rb"))  # type: Palminizable

        base_model = mypalminizedmodel.base_model
        dct_name_facto = mypalminizedmodel.sparsely_factorized_layers
        base_score = base_model.evaluate(x_test, y_test, verbose=0)[1]
        print(base_score)
        palminized_model = mypalminizedmodel.compressed_model
        palminized_score = palminized_model.evaluate(x_test, y_test, verbose=1)[1]
        print(palminized_score)
        fine_tuned_model = replace_layers_with_sparse_facto(palminized_model, dct_name_facto)
        # fine_tuned_model = palminized_model

        fine_tuned_model.compile(loss=param_train_dataset.loss,
                                 optimizer=param_train_dataset.optimizer,
                                 metrics=['categorical_accuracy'])

        before_finetuned_score = fine_tuned_model.evaluate(x_test, y_test, verbose=1)[1]
        print(before_finetuned_score)

    # results must be already printed once in case process is killed afterward
    dct_results = {
        "finetuned_score": None,
        "before_finetuned_score": before_finetuned_score,
        "base_score": base_score,
        "palminized_score": palminized_score,
    }
    resprinter.add(dct_results)
    resprinter.print()

    # if paraman["--hierarchical"]:
    if not paraman["--only-mask"]:
        assert before_finetuned_score == palminized_score, \
        "the reconstructed model with sparse facto should equal in perf to the reconstructed model with dense product. {} != {}".format(before_finetuned_score, palminized_score)
    # else: # small fix for a bug where when I wasn't using hierarchical palm returned a matrix that wasn't multiplied by lambda
    #     # this should pass until results are generated without bug..
    #     assert before_finetuned_score != palminized_score, \
    #         "the reconstructed model with sparse facto should equal in perf to the reconstructed model with dense product. {} != {}".format(before_finetuned_score, palminized_score)
    fine_tuned_model.summary()

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
        history = fine_tuned_model.fit(param_train_dataset.image_data_generator.flow(x_train, y_train, batch_size=param_train_dataset.batch_size),
                                       epochs=param_train_dataset.epochs - init_nb_epoch,
                                       # epochs=2 - init_nb_epoch,
                                       verbose=2,
                                       validation_data=(x_test, y_test),
                                       callbacks=param_train_dataset.callbacks + call_backs)
        signal.alarm(0)  # stop alarm for next evaluation
        finetuned_score = fine_tuned_model.evaluate(x_test, y_test, verbose=1)[1]
        print(finetuned_score)

        if os.path.exists(paraman["output_file_notfinishedprinter"]):
            os.remove(paraman["output_file_notfinishedprinter"])
    # except TimeoutError as te:
    except Exception as e:
        logging.error("Caught exception: {}".format(e))
    finally:
        dct_results = {
            "finetuned_score": finetuned_score,
            "before_finetuned_score": before_finetuned_score,
            "base_score": base_score,
            "palminized_score": palminized_score,
        }
        fine_tuned_model.save(str(paraman["output_file_modelprinter"]))
        resprinter.add(dct_results)


if __name__ == "__main__":
    logger.info("Command line: " + " ".join(sys.argv))
    log_memory_usage("Memory at startup")
    arguments = docopt.docopt(__doc__)
    paraman = ParameterManagerPalminizeFinetune(arguments)
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