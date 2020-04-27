"""
This script is for running compression of convolutional networks using tucker decomposition.

Usage:
    script.py [-h] [-v|-vv] [--nb-factor=int] --sparsity-factor=int [--nb-iteration-palm=int] [--tol=float] [--hierarchical] (--mnist|--svhn|--cifar10|--cifar100|--test-data) [--cifar100-resnet50|--cifar100-resnet50-new|--cifar100-resnet20|--mnist-500|--mnist-lenet|--test-model|--cifar10-vgg19|--cifar100-vgg19|--svhn-vgg19]

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
  --mnist-500                           Use model fc 500 hidden units pretrained on mnist.
  --cifar100-resnet50                   Use model resnet50 pretrained on cifar100.
  --cifar100-resnet50-new               Use model resnet50 pretrained on cifar100 with new architecture.
  --cifar100-resnet20                   Use model resnet20 pretrained on cifar100.

Palm-Specifc options:
  --sparsity-factor=int                 Integer coefficient from which is computed the number of value in each factor.
  --nb-iteration-palm=int               Number of iterations in the inner palm4msa calls. [default: 300]
  --tol=float                           Threshold value before stopping palm iterations. [default: 1e-6]
  --hierarchical                        Tells if palm should use the hierarchical euristic or not. Muhc longer but better approximation results.
  --nb-factor=int                       Tells the number of factors for palm
"""
import logging
import pickle
import sys
import time
import numpy as np
import keras
import docopt
import os
import pandas as pd

from palmnet.core.faustizer import Faustizer
from palmnet.core.layer_replacer_faust import LayerReplacerFaust
from palmnet.experiments.utils import ResultPrinter, ParameterManager
from skluc.utils import logger, log_memory_usage
from pathlib import Path
import zlib

class ParameterManagerCompressionFaust(ParameterManager):
    def __init__(self, dct_params, **kwargs):
        super().__init__(self, **dct_params, **kwargs)
        self["--sparsity-factor"] = int(self["--sparsity-factor"]) if self["--sparsity-factor"] is not None else None
        self["--nb-iteration-palm"] = int(self["--nb-iteration-palm"]) if self["--nb-iteration-palm"] is not None else None
        self["--tol"] = float(self["--tol"]) if self["--tol"] is not None else None
        self["--nb-factor"] = int(self["--nb-factor"]) if self["--nb-factor"] is not None else None

        self.__init_hash_expe()
        self.__init_output_file()

        self["output_file_modelprinter"].mkdir(parents=True, exist_ok=True)

    def __init_output_file(self):
        self["output_file_resprinter"] = Path(self["hash"] + "_results.csv")
        self["output_file_modelprinter"] = Path(self["hash"] + "_faust_objs")
        self["output_file_notfinishedprinter"] = Path(self["hash"] + ".notfinished")

    def __init_hash_expe(self):
        lst_elem_to_remove_for_hash = [
            'identifier',
            '-v',
            '--help',
            "output_file_resprinter",
            "output_file_modelprinter",
            "output_file_notfinishedprinter",
            "output_file_csvcbprinter"
        ]

        keys_expe = sorted(self.keys())
        any(keys_expe.remove(item) for item in lst_elem_to_remove_for_hash if item in keys_expe)
        val_expe = [self[k] for k in keys_expe]
        str_expe = [str(val) for pair in zip(keys_expe, val_expe) for val in pair]
        self["hash"] = hex(zlib.crc32(str.encode("".join(str_expe))))


lst_results_header = [
    "decomposition_time",
    "test_accuracy_base_model",
    "test_accuracy_compressed_model",
    "test_loss_base_model",
    "test_loss_compressed_model",
]

def get_base_model():
    base_model = paraman.get_model()
    return base_model

def compress_model(base_model):

    faustizer = Faustizer(sparsity_fac=paraman["--sparsity-factor"],
                          nb_factor=paraman["--nb-factor"],
                          nb_iter=paraman["--nb-iteration-palm"],
                          tol=paraman["--tol"],
                          hierarchical=paraman["--hierarchical"])

    layer_replacer = LayerReplacerFaust(only_mask=False, sparse_factorizer=faustizer, path_checkpoint_file=paraman["output_file_modelprinter"])

    if os.path.exists(paraman["output_file_notfinishedprinter"]):
        layer_replacer.load_dct_name_compression()
    else:
        open(paraman["output_file_notfinishedprinter"], 'w').close()

    start_replace = time.time()
    layer_replacer.fit(base_model)
    stop_replace = time.time()

    dct_results = {
        "decomposition_time": stop_replace - start_replace
    }
    resprinter.add(dct_results)


def main():
    # Base Model #
    base_model = get_base_model()
    # write preliminary results before compression (ease debugging)
    resprinter.print()
    # Do compression #
    compress_model(base_model)

    if os.path.exists(paraman["output_file_notfinishedprinter"]):
        os.remove(paraman["output_file_notfinishedprinter"])


if __name__ == "__main__":
    logger.info("Command line: " + " ".join(sys.argv))
    log_memory_usage("Memory at startup")
    arguments = docopt.docopt(__doc__)
    paraman = ParameterManagerCompressionFaust(arguments)
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