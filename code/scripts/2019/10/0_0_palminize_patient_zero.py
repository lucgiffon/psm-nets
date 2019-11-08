"""
This script is the experiment script for palminizing models.

Usage:
    script.py [-h] [-v|-vv] (--mnist|--svhn|--cifar10|--cifar100|--test-data) [--mnist-lenet|--test-model|--cifar10-vgg19|--cifar100-vgg19|--svhn-vgg19] --sparsity-factor=int [--nb-iteration-palm=int] [--delta-threshold=float] [--hierarchical]

Options:
  -h --help                             Show this screen.
  -vv                                   Set verbosity to debug.
  -v                                    Set verbosity to info.

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

import docopt

from palmnet.core.palminize import Palminizer, Palminizable
from palmnet.utils import ParameterManager, ResultPrinter
from skluc.utils import logger, log_memory_usage

lst_results_header = [
    "palminization_time",
    "nb_param_model_layers_conv_dense",
    "nb_flops_model_layers_conv_dense",
    "test_accuracy_base_model",
    "test_accuracy_compressed_model"
]

def main():
    base_model = paraman.get_model()
    (x_train, y_train), (x_test, y_test) = paraman.get_dataset()

    palminizer = Palminizer(sparsity_fac=paraman["--sparsity-factor"],
                            nb_iter=paraman["--nb-iteration-palm"],
                            delta_threshold_palm=paraman["--delta-threshold"],
                            hierarchical=paraman["--hierarchical"],
                            fast_unstable_proj=True)

    palminizable = Palminizable(base_model, palminizer)
    start_palminize = time.time()
    palminizable.palminize()
    stop_palminize = time.time()

    nb_param_base_model, nb_param_compressed_model, nb_flop_base_model, nb_flop_compressed_model = palminizable.count_model_param_and_flops()


    score_base, acc_base, score_compressed, acc_compressed = palminizable.evaluate(x_test, y_test)

    dct_results = {
        "palminization_time": stop_palminize - start_palminize,
        "nb_param_base_layers_conv_dense": nb_param_base_model,
        "nb_flops_base_layers_conv_dense": nb_flop_base_model,
        "nb_param_compressed_layers_conv_dense": nb_param_compressed_model,
        "nb_flops_compressed_layers_conv_dense": nb_flop_compressed_model,
        "test_accuracy_base_model": acc_base,
        "test_accuracy_compressed_model": acc_compressed,
        "test_loss_base_model": score_base,
        "test_loss_compressed_model": score_compressed
    }

    pickle.dump(palminizable, open(str(paraman["output_file_modelprinter"]), 'wb'))
    resprinter.add(dct_results)


if __name__ == "__main__":
    logger.info("Command line: " + " ".join(sys.argv))
    log_memory_usage("Memory at startup")
    arguments = docopt.docopt(__doc__)
    paraman = ParameterManager(arguments)
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