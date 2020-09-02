"""
This script is for running compression of convolutional networks using tucker decomposition.

Usage:
    script.py tensortrain [-h] [-v|-vv] [--rank0-1-conv] [--only-dense] [--use-pretrained] [--rank-value int] [--order int] [--keep-first-layer] [--keep-last-layer] (--mnist|--svhn|--cifar10|--cifar100|--test-data) [--cifar100-resnet50-new|--cifar100-resnet20-new|--cifar100-resnet50|--cifar100-resnet20|--mnist-500|--mnist-lenet|--test-model|--cifar10-vgg19|--cifar100-vgg19|--svhn-vgg19]
    script.py random [-h] [-v|-vv] --sparsity-factor int [--nb-factor int] [--only-dense] [--keep-last-layer] [--keep-first-layer] (--mnist|--svhn|--cifar10|--cifar100|--test-data) [--cifar100-resnet50-new|--cifar100-resnet20-new|--cifar100-resnet50|--cifar100-resnet20|--mnist-500|--mnist-lenet|--test-model|--cifar10-vgg19|--cifar100-vgg19|--svhn-vgg19]

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
  --cifar100-resnet50-new               Use model resnet50 pretrained on cifar100.
  --cifar100-resnet20                   Use model resnet20 pretrained on cifar100.
  --cifar100-resnet20-new               Use model resnet50 pretrained on cifar100.

Compression specific options:
    --keep-first-layer                  Tell the replacer to keep the first layer of the network
    --keep-last-layer                   Tell the replacer to keep the last layer (classification) of the network
    --only-dense                        Tell the replacer to replace only dense layers.

Tensortrain specific option:
    --rank-value int                    The values for r0, r1 r... rk (exemple: 2, 4, 6)
    --order int                         The value for k (number of cores)
    --use-pretrained                    Tell the layer replacer to use the decomposition of the initial weights.
    --rank0-1-conv                      Tell to use a r0 value equal to 1 in convolutional layers. For fair comparison in term of input size.

Random Sparse Facto options:
    --sparsity-factor int                 Integer coefficient from which is computed the number of value in each factor.
    --nb-factor int                       Tells the number of sparse factor for palm

"""
import logging

from keras.layers import Conv2D, Dense

mpl_logger = logging.getLogger("matplotlib")
mpl_logger.setLevel(logging.WARNING)
from tensorflow import set_random_seed
import yaml

import sys
import numpy as np
import keras as base_keras
import docopt
import os
from pathlib import Path
import zlib
import tensorflow as tf

from palmnet.core.layer_replacer_TT import LayerReplacerTT
from palmnet.core.layer_replacer_random_sparse_facto import LayerReplacerRandomSparseFacto
from palmnet.core.randomizer import Randomizer
from palmnet.data import Mnist, Cifar10, Cifar100, Svhn, Test
from palmnet.experiments.utils import ResultPrinter, ParameterManager
from palmnet.utils import translate_keras_to_tf_model, load_function_lr
from skluc.utils import logger, log_memory_usage

# from tensorflow_model_optimization.python.core.api.sparsity import keras


lst_results_header = [
]


class ParameterManagerTensotrainAndTuckerDecomposition(ParameterManager):
    def __init__(self, dct_params, **kwargs):
        super().__init__(self, **dct_params, **kwargs)
        self.__init_seed()

        # tensortrain parameters
        self["--rank-value"] = int(self["--rank-value"]) if self["--rank-value"] is not None else None
        self["--order"] = int(self["--order"]) if self["--order"] is not None else None


        self["--sparsity-factor"] = int(self["--sparsity-factor"]) if self["--sparsity-factor"] is not None else None
        self["--nb-factor"] = int(self["--nb-factor"]) if self["--nb-factor"] is not None else None

        self.__init_hash_expe()
        self.__init_output_file()

    def __init_output_file(self):
        self["output_file_resprinter"] = Path(self["hash"] + "_results.csv")
        self["output_file_modelprinter"] = Path(self["hash"] + "_model.h5")
        self["output_file_notfinishedprinter"] = Path(self["hash"] + ".notfinished")
        self["output_file_csvcbprinter"] = Path(self["hash"] + "_history.csv")
        self["output_file_layerbylayer"] = Path(self["hash"] + "_layerbylayer.csv")

    def __init_seed(self):
        if not "--seed" in self.keys():
            self["seed"] = np.random.randint(0, 2 ** 32 - 2)

        elif self["--seed"] is not None:
            self["seed"] = int(self["--seed"])
            np.random.seed(self["seed"])
        else:
            self["seed"] = np.random.randint(0, 2 ** 32 - 2)

    def __init_hash_expe(self):
        lst_elem_to_remove_for_hash = [
            'identifier',
            '-v',
            '--help',
            "output_file_resprinter",
            "output_file_modelprinter",
            "output_file_notfinishedprinter",
            "output_file_csvcbprinter",
            "output_file_layerbylayer",
        ]

        keys_expe = sorted(self.keys())
        any(keys_expe.remove(item) for item in lst_elem_to_remove_for_hash if item in keys_expe)
        val_expe = [self[k] for k in keys_expe]
        str_expe = [str(val) for pair in zip(keys_expe, val_expe) for val in pair]
        self["hash"] = hex(zlib.crc32(str.encode("".join(str_expe))))


def get_and_evaluate_base_model(model_compilation_params):
    base_model = paraman.get_model()
    base_model.compile(**model_compilation_params)
    return base_model

def compress_and_evaluate_model(base_model, model_compilation_params):
    if paraman["tensortrain"]:
        layer_replacer = LayerReplacerTT(keep_last_layer=paraman["--keep-last-layer"], keep_first_layer=paraman["--keep-first-layer"], only_dense=paraman["--only-dense"],
                                         rank_value=paraman["--rank-value"], order=paraman["--order"], use_pretrained=paraman["--use-pretrained"], tt_rank0_conv_1=paraman["--rank0-1-conv"])
    elif paraman["random"]:
        randomizer = Randomizer(sparsity_fac=paraman["--sparsity-factor"],
                                nb_factor=paraman["--nb-factor"])

        layer_replacer = LayerReplacerRandomSparseFacto(only_mask=True, sparse_factorizer=randomizer,
                                                        keep_last_layer=paraman["--keep-last-layer"],
                                                        keep_first_layer=paraman["--keep-first-layer"],
                                                        only_dense=paraman["--only-dense"]
                                                        )
    else:
        raise ValueError("Unknown compression method.")

    new_model = layer_replacer.fit_transform(base_model)
    new_model.compile(**model_compilation_params)

    return new_model

def get_or_load_new_model(model_compilation_params):

    base_model = get_and_evaluate_base_model(model_compilation_params)

    # New model compression #
    new_model = compress_and_evaluate_model(base_model, model_compilation_params)
    return new_model, base_model


def main():

    # Params optimizer #
    model_compilation_param = {"loss": "categorical_crossentropy", "optimizer": "adam"}
    # Do compression or load #
    new_model, base_model = get_or_load_new_model(model_compilation_param)
    # Write results before finetuning #

    max_size = -1
    max_layer = None
    for layer in new_model.layers:
        if hasattr(layer, "image_max_size"):
            layer_size = layer.image_max_size
            max_size = max(max_size, layer_size)
            if max_size == layer_size:
                max_layer = layer.name
        elif isinstance(layer, Conv2D) or isinstance(layer, Dense):
            size_input = np.prod(layer.input_shape[1:])
            layer_size = size_input
            max_size = max(max_size, layer_size)
            if max_size == layer_size:
                max_layer = layer.name

    dct_max_size = {
        "max_item_size": max_size,
        "max_layer": max_layer
    }
    resprinter.add(dct_max_size)
    resprinter.print()


if __name__ == "__main__":
    logger.info("Command line: " + " ".join(sys.argv))
    log_memory_usage("Memory at startup")
    arguments = docopt.docopt(__doc__)
    paraman = ParameterManagerTensotrainAndTuckerDecomposition(arguments)
    set_random_seed(paraman["seed"])
    initialized_results = dict((v, None) for v in lst_results_header)
    keras = base_keras

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