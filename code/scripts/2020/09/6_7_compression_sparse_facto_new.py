"""
Usage:
    script.py [-h] [-v|-vv] (faust|palm) [--nb-factor=int] [--seed int] [--activations] [--max-cum-batch-size int] [--batch-size int] [--nb-epochs int] --sparsity-factor=int [--nb-iteration-palm=int] [--train-val-split float] [--tol=float] [--hierarchical] (--mnist|--svhn|--cifar10|--cifar100|--test-data) [--cifar100-resnet50|--cifar100-resnet50-new|--cifar100-resnet20-new|--cifar100-resnet20|--mnist-500|--mnist-lenet|--test-model|--cifar10-vgg19|--cifar100-vgg19|--svhn-vgg19]

Options:
  -h --help                             Show this screen.
  -vv                                   Set verbosity to debug.
  -v                                    Set verbosity to info.
  --seed int                            Seed

Dataset:
  --mnist                               Use Mnist dataset.
  --svhn                                Use svhn dataset.
  --cifar10                             Use cifar10 dataset.
  --cifar100                            Use cifar100 dataset.
  --test-data                           Use test datasset (that is actually mnist).
  --train-val-split float               Validation percentage from test.

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
  --cifar100-resnet20-new               Use model resnet20 pretrained on cifar100 with new architecture.

Palm-Specifc options:
  faust                                 use faust
  palm                                  use pyqalm
  --sparsity-factor=int                 Integer coefficient from which is computed the number of value in each factor.
  --nb-iteration-palm=int               Number of iterations in the inner palm4msa calls. [default: 300]
  --tol=float                           Threshold value before stopping palm iterations. [default: 1e-6]
  --hierarchical                        Tells if palm should use the hierarchical euristic or not. Muhc longer but better approximation results.
  --nb-factor=int                       Tells the number of factors for palm
  --activations                         Use the linear activations for the factorization

Palm-act specific options:
  --max-cum-batch-size int              The cumulated size of batches in memory
  --batch-size int                      The size of a single batch.
  --nb-epochs int                       The number of epochs
"""
import logging
from tensorflow.python.framework.errors_impl import InvalidArgumentError

from palmnet.core.activation_palminizer import ActivationPalminizer
from palmnet.core.layer_replacer_sparse_facto_activations import LayerReplacerSparseFactoActivations
from palmnet.layers.sparse_facto_conv2D_masked import SparseFactorisationConv2D
from palmnet.layers.sparse_facto_dense_masked import SparseFactorisationDense
from palmnet.utils import get_nb_learnable_weights_from_model, get_nb_learnable_weights

mpl_logger = logging.getLogger("matplotlib")
mpl_logger.setLevel(logging.WARNING)

from keras.models import Model
import tensorflow as tf

from palmnet.core import palminizable
from collections import defaultdict
# from palmnet.layers.sparse_tensor import SparseFactorisationDense#, SparseFactorisationConv2DDensify
import logging
import time
from keras.layers import Conv2D, Dense
import docopt
import keras.backend as K
import os
import numpy as np
from palmnet.core.faustizer import Faustizer
from palmnet.core.layer_replacer_faust import LayerReplacerFaust
from palmnet.core.layer_replacer_palm import LayerReplacerPalm
from palmnet.core.palminizer import Palminizer
from palmnet.experiments.utils import ResultPrinter, ParameterManager
from skluc.utils import logger, log_memory_usage
from pathlib import Path
import zlib
from sklearn.model_selection import train_test_split
import pandas as pd
palminizable.Palminizer = Palminizer
import sys
sys.modules["palmnet.core.palminize"] = palminizable

class ParameterManagerCompressionFaust(ParameterManager):
    def __init__(self, dct_params, **kwargs):
        super().__init__(self, **dct_params, **kwargs)
        self.__init_seed()
        self["--sparsity-factor"] = int(self["--sparsity-factor"]) if self["--sparsity-factor"] is not None else None
        self["--nb-iteration-palm"] = int(self["--nb-iteration-palm"]) if self["--nb-iteration-palm"] is not None else None
        self["--tol"] = float(self["--tol"]) if self["--tol"] is not None else None
        self["--nb-factor"] = int(self["--nb-factor"]) if self["--nb-factor"] is not None else None

        if self["--activations"]:
            self["--nb-epochs"] = int(self["--nb-epochs"])
            self["--max-cum-batch-size"] = int(self["--max-cum-batch-size"])
            self["--batch-size"] = int(self["--batch-size"])
            self["nb-queue"] = self["--max-cum-batch-size"] // self["--batch-size"]

        if "--train-val-split" in self.keys() and self["--train-val-split"] is not None:
            self["--train-val-split"] = float(self["--train-val-split"]) if self["--train-val-split"] is not None else None
            assert 0 <= self["--train-val-split"] <= 1, f"Train-val split should be comprise between 0 and 1. {self['--train-val-split']}"

        self.__init_hash_expe()
        self.__init_output_file()

        if self["faust"]:
            self["output_file_modelprinter"].mkdir(parents=True, exist_ok=True)

    def __init_seed(self):
        if not "--seed" in self.keys():
            self["seed"] = np.random.randint(0, 2 ** 32 - 2)

        if self["--seed"] is not None:
            self["seed"] = int(self["--seed"])
            np.random.seed(self["seed"])
        else:
            self["seed"] = np.random.randint(0, 2 ** 32 - 2)

    def __init_output_file(self):
        self["output_file_resprinter"] = Path(self["hash"] + "_results.csv")
        self["output_file_modelprinter"] = Path(self["hash"] + "_faust_objs")
        self["output_file_notfinishedprinter"] = Path(self["hash"] + ".notfinished")
        self["output_file_layerbylayer"] = Path(self["hash"] + "_layerbylayer.csv")
        self["ouput_file_objectives"] = Path(self["hash"] + "_objectives.csv")

    def __init_hash_expe(self):
        lst_elem_to_remove_for_hash = [
            'identifier',
            '-v',
            '--help',
            "output_file_resprinter",
            "output_file_modelprinter",
            "output_file_notfinishedprinter",
            "output_file_csvcbprinter",
            "seed"
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


def compress_model(base_model, x_train, x_val):

    if paraman["faust"]:
        faustizer = Faustizer(sparsity_fac=paraman["--sparsity-factor"],
                              nb_factor=paraman["--nb-factor"],
                              nb_iter=paraman["--nb-iteration-palm"],
                              tol=paraman["--tol"],
                              hierarchical=paraman["--hierarchical"])

        layer_replacer = LayerReplacerFaust(only_mask=False, sparse_factorizer=faustizer, path_checkpoint_file=paraman["output_file_modelprinter"])
    elif paraman["palm"]:
        if paraman["--activations"]:
            palminizer = ActivationPalminizer(train_data=x_train,
                                              nb_epochs=paraman["--nb-epochs"],
                                              batch_size=paraman["--batch-size"],
                                              queue_maxisize=paraman["nb-queue"],
                                              seed=paraman["seed"],
                                              sparsity_fac=paraman["--sparsity-factor"],
                                              nb_factor=paraman["--nb-factor"],
                                              nb_iter=paraman["--nb-iteration-palm"],
                                              delta_threshold_palm=paraman["--tol"],
                                              hierarchical=paraman["--hierarchical"],
                                              val_data=x_val
                                              )
            class_layer_replacer_palm = LayerReplacerSparseFactoActivations
        else:
            palminizer = Palminizer(sparsity_fac=paraman["--sparsity-factor"],
                                  nb_factor=paraman["--nb-factor"],
                                  nb_iter=paraman["--nb-iteration-palm"],
                                  delta_threshold_palm=paraman["--tol"],
                                  hierarchical=paraman["--hierarchical"])

            class_layer_replacer_palm = LayerReplacerPalm

        layer_replacer = class_layer_replacer_palm(only_mask=False, sparse_factorizer=palminizer, path_checkpoint_file=paraman["output_file_modelprinter"])
    else:
        raise NotImplementedError

    if os.path.exists(paraman["output_file_notfinishedprinter"]):
        layer_replacer.load_dct_name_compression()
    else:
        open(paraman["output_file_notfinishedprinter"], 'w').close()

    start_replace = time.time()
    new_model = layer_replacer.fit_transform(base_model)
    stop_replace = time.time()

    dct_results = {
        "decomposition_time": stop_replace - start_replace
    }
    resprinter.add(dct_results)

    return new_model, layer_replacer


def get_dataset():
    (x_train, y_train), (x_test, y_test) = paraman.get_dataset().load_data()
    if paraman["--mnist-500"]:
        x_test = np.reshape(x_test, (-1, 784))
        x_train = np.reshape(x_train, (-1, 784))

    if paraman["--train-val-split"] is not None:
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=paraman["--train-val-split"], random_state=paraman["seed"])
    else:
        x_val, y_val = x_test, y_test

    return (x_train, y_train), (x_val, y_val), (x_test, y_test)


def count_models_parameters(base_model, new_model, dct_name_compression, x_val):
    base_model_nb_param = get_nb_learnable_weights_from_model(base_model)
    new_model_nb_param = get_nb_learnable_weights_from_model(new_model)

    dct_results_param = {
        "base_model_nb_param": base_model_nb_param,
        "new_model_nb_param": new_model_nb_param
    }

    resprinter.add(dct_results_param)

    if len(base_model.layers) < len(new_model.layers):
        base_model = Model(inputs=base_model.inputs, outputs=base_model.outputs)

    dct_results_matrices = defaultdict(lambda: [])

    for idx_layer, compressed_layer in enumerate(new_model.layers):
        if any(isinstance(compressed_layer, _class) for _class in (Dense, Conv2D, SparseFactorisationDense, SparseFactorisationConv2D)):
            log_memory_usage("Start secondary loop")
            base_layer = base_model.layers[idx_layer]

            intermediate_layer_base_model = Model(inputs=base_model.input,
                                             outputs=base_layer.output)
            intermediate_output_base_model = intermediate_layer_base_model.predict(x_val)
            intermediate_layer_compressed_model = Model(inputs=new_model.input,
                                             outputs=compressed_layer.output)
            intermediate_output_compressed_model = intermediate_layer_compressed_model.predict(x_val)

            diff_total_processing = np.linalg.norm(intermediate_output_base_model - intermediate_output_compressed_model) / np.linalg.norm(intermediate_output_base_model)
            dct_results_matrices["diff-total-processing"].append(diff_total_processing)

            try:
                just_before_intermediate_layer_compressed_model = Model(inputs=new_model.input,
                                                            outputs=compressed_layer.input)
                output_just_before_intermediate_layer_compressed_model = just_before_intermediate_layer_compressed_model.predict(x_val)
            except InvalidArgumentError:
                output_just_before_intermediate_layer_compressed_model = x_val

            only_intermediate_layer_compressed_model = K.function([compressed_layer.input], [compressed_layer.output])
            output_only_intermediate_layer_compressed_model = only_intermediate_layer_compressed_model([output_just_before_intermediate_layer_compressed_model])[0]

            only_intermediate_layer_base_model = K.function([base_layer.input], [base_layer.output])
            output_only_intermediate_layer_base_model = only_intermediate_layer_base_model([output_just_before_intermediate_layer_compressed_model])[0]

            diff_only_layer_processing = np.linalg.norm(output_only_intermediate_layer_compressed_model - output_only_intermediate_layer_base_model) / np.linalg.norm(output_only_intermediate_layer_base_model)
            dct_results_matrices["diff-only-layer-processing"].append(diff_only_layer_processing)

            # get informations to identify the layer (and do cross references)
            dct_results_matrices["idx-expe"].append(paraman["identifier"])
            dct_results_matrices["hash"].append(paraman["hash"])
            dct_results_matrices["layer-name-compressed"].append(compressed_layer.name)
            dct_results_matrices["layer-name-base"].append(base_layer.name)
            dct_results_matrices["idx-layer"].append(idx_layer)
            dct_results_matrices["layer-class-name"].append(base_layer.__class__.__name__)
            dct_results_matrices["layer-compressed-class-name"].append(compressed_layer.__class__.__name__)

            if isinstance(compressed_layer, SparseFactorisationDense) or isinstance(compressed_layer, SparseFactorisationConv2D):
                actual_nb_factor = int(compressed_layer.nb_factor)
            else:
                actual_nb_factor = None
            dct_results_matrices["actual-nb-factor"].append(actual_nb_factor)
            # complexity analysis #
            # get nb val base layer and comrpessed layer
            nb_weights_base_layer = get_nb_learnable_weights(base_layer)
            dct_results_matrices["nb-non-zero-base"].append(nb_weights_base_layer)
            nb_weights_compressed_layer = get_nb_learnable_weights(compressed_layer)
            dct_results_matrices["nb-non-zero-compressed"].append(nb_weights_compressed_layer)
            dct_results_matrices["nb-non-zero-compression-rate"].append(nb_weights_base_layer / nb_weights_compressed_layer)

            sparse_factorization = dct_name_compression.get(base_layer.name, None)
            if type(sparse_factorization) == tuple:
                scaling = sparse_factorization[0]
                factors = Palminizer.get_factors_from_op_sparsefacto(sparse_factorization[1])
            else:
                scaling = sparse_factorization["lambda"]
                if paraman["faust"]:
                    factors = Faustizer.get_factors_from_op_sparsefacto(sparse_factorization["sparse_factors"])
                else:
                    factors = Palminizer.get_factors_from_op_sparsefacto(sparse_factorization["sparse_factors"])

            # rebuild full matrix to allow comparisons
            reconstructed_matrix = np.linalg.multi_dot(factors) * scaling
            base_matrix = np.reshape(base_layer.get_weights()[0], reconstructed_matrix.shape)

            # normalized approximation errors
            diff = np.linalg.norm(base_matrix - reconstructed_matrix) / np.linalg.norm(base_matrix)
            dct_results_matrices["diff-approx"].append(diff)

    df_results_layers = pd.DataFrame.from_dict(dct_results_matrices)
    df_results_layers.to_csv(paraman["output_file_layerbylayer"])

def save_objectives(layer_replacer):
    if paraman["--activations"]:
        dct_lst_objectives = layer_replacer.sparse_factorizer.dct_lst_objectives
        lst_rows = []
        for layer, lst_obj in dct_lst_objectives.items():
            for obj_entry in lst_obj:
                tpl_row = (layer, *obj_entry)
                lst_rows.append(tpl_row)
        array_obj = np.array(lst_rows)
        df = pd.DataFrame(array_obj)
        df.to_csv(paraman["ouput_file_objectives"])


def main():
    # Base Model #
    base_model = get_base_model()
    # Dataset
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = get_dataset()
    # write preliminary results before compression (ease debugging)
    resprinter.print()
    # Do compression #
    new_model, layer_replacer = compress_model(base_model, x_train, x_val[np.random.permutation(x_val.shape[0])[:paraman["--batch-size"]]])

    save_objectives(layer_replacer)

    count_models_parameters(base_model, new_model, layer_replacer.dct_name_compression, x_val)

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