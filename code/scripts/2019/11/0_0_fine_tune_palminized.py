"""
This script finds a palminized model with given arguments then finetune it.

Usage:
    script.py --input-dir path [-h] [-v|-vv] (--mnist|--svhn|--cifar10|--cifar100|--test-data) [--mnist-lenet|--test-model|--cifar10-vgg19|--cifar100-vgg19|--svhn-vgg19] --sparsity-factor=int [--nb-iteration-palm=int] [--delta-threshold=float] [--hierarchical]

Options:
  -h --help                             Show this screen.
  -vv                                   Set verbosity to debug.
  -v                                    Set verbosity to info.
  --input-dir path                      Path to input directory where to find previously generated results.

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


Palm-Specifc options:
  --sparsity-factor=int                 Integer coefficient from which is computed the number of value in each factor.
  --nb-iteration-palm=int               Number of iterations in the inner palm4msa calls. [default: 300]
  --delta-threshold=float               Threshold value before stopping palm iterations. [default: 1e-6]
  --hierarchical                        Tells if palm should use the hierarchical euristic or not. Muhc longer but better approximation results.
"""
import logging
import pickle
import sys
import time
from copy import deepcopy

import docopt

from palmnet.core.palminize import Palminizer, Palminizable
from palmnet.data import Mnist, Test, Svhn, Cifar100, Cifar10
from palmnet.layers import SparseFactorisationDense, SparseFactorisationConv2D
from palmnet.utils import ParameterManager, ResultPrinter, ParameterManagerFinetune, get_sparsity_pattern, insert_layer_nonseq
from skluc.utils import logger, log_memory_usage
from keras.layers import Dense, Conv2D
import numpy as np

lst_results_header = [
    "test_accuracy_finetuned_model"
]

def replace_layers_with_sparse_facto(model, dct_name_facto):
    new_model = deepcopy(model)
    modified_layer_names = []

    for i, layer in enumerate(new_model.layers):
        layer_name = layer.name
        sparse_factorization = dct_name_facto[layer_name]

        if sparse_factorization != (None, None):
            scaling = sparse_factorization[0]
            factors = [fac.toarray() for fac in sparse_factorization[1].get_list_of_factors()]
            sparsity_patterns = [get_sparsity_pattern(w) for w in factors]

            # create new layer
            if isinstance(layer, Dense):
                hidden_layer_dim = layer.units
                activation = layer.activation
                replacing_layer = SparseFactorisationDense(units=hidden_layer_dim, sparsity_patterns=sparsity_patterns, use_bias=layer.use_bias, activation=activation)
                replacing_weights = [np.array(scaling)[None]] + factors + [layer.get_weights()[-1]] if layer.use_bias else []
                new_model = insert_layer_nonseq(new_model, layer_name, lambda: replacing_layer, position="replace")
                replacing_layer.set_weights(replacing_weights)

            elif isinstance(layer, Conv2D):
                nb_filters = layer.filters
                kernel_size = layer.kernel_size
                activation = layer.activation
                padding = layer.padding
                strides = layer.strides
                replacing_layer = SparseFactorisationConv2D(filters=nb_filters, kernel_size=kernel_size, sparsity_patterns=sparsity_patterns, use_bias=layer.use_bias, activation=activation, padding=padding, strides=strides)
                replacing_weights = [np.array(scaling)[None]] + factors + [layer.get_weights()[-1]] if layer.use_bias else []
                new_model = insert_layer_nonseq(new_model, layer_name, lambda: replacing_layer, position="replace")
                replacing_layer.set_weights(replacing_weights)

            else:
                raise ValueError("unknown layer class")
            modified_layer_names.append((layer_name, replacing_layer.name))

    return new_model, modified_layer_names

def main():

    if paraman["--mnist-lenet"]:
        param_train_dataset = Mnist.get_model_param_training()
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


    mypalminizedmodel = pickle.load(open(paraman["input_model_path"], "rb"))  # type: Palminizable

    (x_train, y_train), (x_test, y_test) = paraman.get_dataset()

    base_model = mypalminizedmodel.base_model
    dct_name_facto = mypalminizedmodel.sparsely_factorized_layers
    base_score = base_model.evaluate(x_test, y_test, verbose=0)

    palminized_model = mypalminizedmodel.compressed_model
    palminized_score = palminized_model.evaluate(x_test, y_test, verbose=0)

    fine_tuned_model, modified_layer_names = replace_layers_with_sparse_facto(palminized_model, dct_name_facto)

    fine_tuned_model.compile(loss=param_train_dataset.loss,
                             optimizer=param_train_dataset.optimizer,
                             metrics=['categorical_accuracy'])

    before_finetuned_score = fine_tuned_model.evaluate(x_test, y_test, verbose=0)

    assert before_finetuned_score[1] == palminized_score[1], "the reconstructed model with sparse facto should equal in perf to the reconstructed model with dense product. {} != {}".format(before_finetuned_score, palminized_score)

    fine_tuned_model.summary()
    exit()
    fine_tuned_model.fit(param_train_dataset.image_data_generator.flow(x_train, y_train, batch_size=param_train_dataset.batch_size),
                         epochs=param_train_dataset.epochs,
                         verbose=1,
                         validation_data=(x_test, y_test),
                         callbacks=param_train_dataset.callbacks)

    finetuned_score = fine_tuned_model.evaluate(x_test, y_test, verbose=0)


    dct_results = {
        "finetuned_score": finetuned_score[-1],
        "before_finetuned_score": before_finetuned_score[-1],
        "base_score": base_score[-1],
        "palminized_score": palminized_score[-1],
    }

    fine_tuned_model.save(str(paraman["output_file_modelprinter"]))
    resprinter.add(dct_results)


if __name__ == "__main__":
    logger.info("Command line: " + " ".join(sys.argv))
    log_memory_usage("Memory at startup")
    arguments = docopt.docopt(__doc__)
    paraman = ParameterManagerFinetune(arguments)
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