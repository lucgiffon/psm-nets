"""
This script finds a palminized model with given arguments then finetune it.

Usage:
    script.py (faust|palm) [--tucker] --input-dir path [--nb-factor-old=int] [--sparsity-factor-old=int]  [--nb-iteration-palm-old=int] [--delta-threshold-old=float] [--hierarchical-old] [--from-compressed-model path] [--lr-configuration-file path] [-h] [-v|-vv] [--seed int] [--only-dense] [--logrange-clr] [--train-val-split float] [--batchnorm] [--keep-last-layer] [--keep-first-layer] [--lr float] [--use-clr policy] [--min-lr float --max-lr float] [--epoch-step-size int] [--nb-epoch int] [--only-mask] [--tb] (--mnist|--svhn|--cifar10|--cifar100|--test-data) [--cifar100-resnet50|--cifar100-resnet20|--cifar100-resnet50-new|--cifar100-resnet20-new|--mnist-500|--mnist-lenet|--test-model|--cifar10-vgg19|--cifar100-vgg19|--svhn-vgg19] --sparsity-factor=int [--nb-iteration-palm=int] [--delta-threshold=float] [--hierarchical] [--nb-factor=int] [--max-cum-batch-size int] [--batch-size int] [--nb-epochs int] [--full-model-approx] [--only-one-batch] [--activations]


Options:
  -h --help                             Show this screen.
  -vv                                   Set verbosity to debug.
  -v                                    Set verbosity to info.
  --seed int                            The seed for the experiments
  --input-dir path                      Path to input directory where to find previously generated results.
  --tb                                  Tell if tensorboard should be printed.
  --lr float                            Flat lr to be used (Overidable)
  --min-lr float                        Tells the min reasonable lr (Overide everything else).
  --max-lr float                        Tells the max reasonable lr (Overide everything else).
  --nb-epoch int                        Number of epochs of training (Overide everything else).
  --epoch-step-size int                 Number of epochs for an half cycle of CLR.
  --use-clr policy                      Tell to use clr. Policy can be "triangular" or "triangular2" (see Cyclical learning rate)
  --logrange-clr                        Use logrange in cycle of learning rate instead of linear.
  --keep-last-layer                     Do not compress classification layer.
  --keep-first-layer                    Do not compress the first layer.
  --train-val-split float               Tells the proportion of validation data. If not specified, validation data is test data.
  --only-dense                          Compress only dense layers.
  --lr-configuration-file path          Tell the path to the lr configuration file
  --from-compressed-model path          Path to input directory where to find previously generated results.

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
  --cifar100-resnet20                   Use model resnet20 pretrained on cifar100.
  --cifar100-resnet50-new               Use model resnet50 pretrained on cifar100.
  --cifar100-resnet20-new               Use model resnet20 pretrained on cifar100.

Palm-Specifc options:
  faust                                 use faust
  palm                                  use pyqalm
  --sparsity-factor=int                 Integer coefficient from which is computed the number of value in each factor.
  --nb-iteration-palm=int               Number of iterations in the inner palm4msa calls. [default: 300]
  --delta-threshold=float               Threshold value before stopping palm iterations. [default: 1e-6]
  --hierarchical                        Tells if palm should use the hierarchical euristic or not. Muhc longer but better approximation results.
  --nb-factor=int                       Tells the number of sparse factor for palm
  --only-mask                           Use only sparsity mask given by palm but re-initialize weights.
  --batchnorm                           Intertwine batchnorm between factors of dense layers.
  --tucker                              Use tucker decomposition before palm.

Palm-specific options for former compression:
  --sparsity-factor-old=int                 Integer coefficient from which is computed the nu
  --nb-iteration-palm-old=int               Number of iterations in the inner palm4msa calls. [default: 300]
  --delta-threshold-old=float               Threshold value before stopping palm iterations. [default: 1e-6]
  --hierarchical-old                        Tells if palm should use the hierarchical euristi
  --nb-factor-old=int                       Tells the number of factors for palm

Palm-act specific options:
  --max-cum-batch-size int              The cumulated size of batches in memory
  --batch-size int                      The size of a single batch.
  --nb-epochs int                       The number of epochs in the palm act algorithm
  --full-model-approx                   Tells to use the target model layers for preprocessing
  --only-one-batch                      Tell to use only one batch of activations
  --activations                         Use the linear activations for the factorization
"""
import logging
mpl_logger = logging.getLogger("matplotlib")
mpl_logger.setLevel(logging.WARNING)
import os
import pathlib
import zlib
from pathlib import Path
from collections import defaultdict

from keras.callbacks import CSVLogger
from keras.layers import Conv2D, Dense
from keras.models import Model
import docopt
import keras
from keras.models import load_model
import keras.backend as K
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from palmnet.core import palminizable
from palmnet.core.faustizer import Faustizer
from palmnet.core.layer_replacer_faust import LayerReplacerFaust
from palmnet.core.layer_replacer_palm import LayerReplacerPalm
from palmnet.core.layer_replacer_sparse_facto_tucker_faust import LayerReplacerSparseFactoTuckerFaust
from palmnet.core.palminizer import Palminizer
from palmnet.data import Mnist, Test, Svhn, Cifar100, Cifar10, categorical_cross_entropy_from_logits
from palmnet.experiments.utils import ResultPrinter, ParameterManagerPalminize, get_line_of_interest
# from palmnet.layers.sparse_tensor import SparseFactorisationDense#, SparseFactorisationConv2DDensify
from palmnet.layers.sparse_facto_conv2D_masked import SparseFactorisationConv2D
from palmnet.layers.sparse_facto_dense_masked import SparseFactorisationDense
from palmnet.layers.tucker_layer import TuckerLayerConv
from palmnet.layers.tucker_layer_sparse_facto import TuckerSparseFactoLayerConv
from palmnet.utils import CSVLoggerByBatch, get_nb_learnable_weights, get_nb_learnable_weights_from_model, get_sparsity_pattern, SafeModelCheckpoint, load_function_lr
from palmnet.utils import CyclicLR
from palmnet.visualization.utils import get_df
from skluc.utils import logger, log_memory_usage
from tensorflow import set_random_seed
import yaml

palminizable.Palminizer = Palminizer
import sys
sys.modules["palmnet.core.palminize"] = palminizable

# noinspection PyUnreachableCode
lst_results_header = [
    "test_accuracy_base_model",
    "test_accuracy_compressed_model",
    "test_accuracy_finetuned_model",
    "test_loss_base_model",
    "test_loss_compressed_model",
    "test_loss_finetuned_model",
    "val_accuracy_base_model",
    "val_accuracy_compressed_model",
    "val_accuracy_finetuned_model",
    "val_loss_base_model",
    "val_loss_compressed_model",
    "val_loss_finetuned_model",
    "train_accuracy_base_model",
    "train_accuracy_compressed_model",
    "train_accuracy_finetuned_model",
    "train_loss_base_model",
    "train_loss_compressed_model",
    "train_loss_finetuned_model",
    "actual-lr",
    "actual-batch-size",
    "actual-nb-epochs",
    "actual-min-lr",
    "actual-max-lr",
    "base_model_nb_param",
    "new_model_nb_param"
]


class ParameterManagerPalminizeFinetune(ParameterManagerPalminize):
    def __init__(self, dct_params, **kwargs):
        super().__init__(dct_params, **kwargs)
        self.__init_seed()

        self["--input-dir"] = pathlib.Path(self["--input-dir"])
        self["--min-lr"] = float(self["--min-lr"]) if self["--min-lr"] is not None else None
        self["--max-lr"] = float(self["--max-lr"]) if self["--max-lr"] is not None else None
        self["--lr"] = float(self["--lr"]) if self["--lr"] is not None else None
        self["--nb-epoch"] = int(self["--nb-epoch"]) if self["--nb-epoch"] is not None else None
        self["--epoch-step-size"] = int(self["--epoch-step-size"]) if self["--epoch-step-size"] is not None else None

        self["--sparsity-factor-old"] = int(self["--sparsity-factor-old"]) if self["--sparsity-factor-old"] is not None else None
        self["--nb-iteration-palm-old"] = int(self["--nb-iteration-palm-old"]) if self["--nb-iteration-palm-old"] is not None else None
        self["--delta-threshold-old"] = float(self["--delta-threshold-old"]) if self["--delta-threshold-old"] is not None else None
        self["--nb-factor-old"] = int(self["--nb-factor-old"]) if self["--nb-factor-old"] is not None else None


        if self["--use-clr"] not in ["triangular", "triangular2"] and self["--use-clr"] is not None:
            if type(self["--use-clr"]) is bool:
                pass
            else:
                raise ValueError(f"CLR policy should be triangular or triangular2. {self['--use-clr']}")

        if self["--activations"]:
            self["--nb-epochs"] = int(self["--nb-epochs"])
            self["--max-cum-batch-size"] = int(self["--max-cum-batch-size"])
            self["--batch-size"] = int(self["--batch-size"])
            self["nb-queue"] = self["--max-cum-batch-size"] // self["--batch-size"]

        self["--max-cum-batch-size"] = int(self["--max-cum-batch-size"]) if self["--max-cum-batch-size"] is not None else None

        if "--train-val-split" in self.keys() and self["--train-val-split"] is not None:
            self["--train-val-split"] = float(self["--train-val-split"]) if self["--train-val-split"] is not None else None
            assert 0 <= self["--train-val-split"] <= 1, f"Train-val split should be comprise between 0 and 1. {self['--train-val-split']}"

        self["--lr-configuration-file"] = Path(self["--lr-configuration-file"]) if self["--lr-configuration-file"] is not None else None
        self["--from-compressed-model"] = Path(self["--from-compressed-model"]) if self["--from-compressed-model"] is not None else None

        self.__init_hash_expe()
        self.__init_output_file()
        self.__init_model_path()

        if self["--from-compressed-model"] is not None:
            self.__init_model_path_compressed_model()

    def __init_seed(self):
        if not "--seed" in self.keys():
            self["seed"] = np.random.randint(0, 2 ** 32 - 2)

        if self["--seed"] is not None:
            self["seed"] = int(self["--seed"])
            self["--seed"] = int(self["--seed"])
            np.random.seed(self["seed"])
        else:
            self["seed"] = np.random.randint(0, 2 ** 32 - 2)

    def __init_hash_expe(self):
        lst_elem_to_remove_for_hash = [
            'output_file_modelprinter',
            'identifier',
            'output_file_resprinter',
            '-v',
            '--help',
            '--input-dir',
        ]
        keys_expe = sorted(self.keys())
        any(keys_expe.remove(item) for item in lst_elem_to_remove_for_hash)
        val_expe = [self[k] for k in keys_expe]
        str_expe = [str(val) for pair in zip(keys_expe, val_expe) for val in pair]
        self["hash"] = hex(zlib.crc32(str.encode("".join(str_expe))))

    def __init_output_file(self):
        self["output_file_resprinter"] = Path(self["hash"] + "_results.csv")
        self["output_file_modelprinter"] = Path(self["hash"] + "_model.h5")
        self["output_file_notfinishedprinter"] = Path(self["hash"] + ".notfinished")
        self["output_file_finishedprinter"] = Path(self["hash"] + ".finished")
        self["output_file_tensorboardprinter"] = Path(self["hash"] + ".tb")
        self["output_file_csvcbprinter"] = Path(self["hash"] + "_history.csv")
        self["output_file_layerbylayer"] = Path(self["hash"] + "_layerbylayer.csv")
        self["output_file_csvcbprinter_epoch"] = Path(self["hash"] + "_history_epoch.csv")

    def __init_model_path(self):
        df = get_df(self["--input-dir"])
        keys_of_interest = ['--cifar10',
                            '--cifar10-vgg19',
                            '--cifar100',
                            '--cifar100-vgg19',
                            '--delta-threshold',
                            '--hierarchical',
                            '--mnist',
                            '--mnist-lenet',
                            '--nb-iteration-palm',
                            '--sparsity-factor',
                            '--svhn',
                            '--svhn-vgg19',
                            '--test-data',
                            '--test-model',
                            "--nb-factor",
                            "--tucker",
                            '--cifar100-resnet50',
                            '--cifar100-resnet20',
                            '--cifar100-resnet50-new',
                            '--cifar100-resnet20-new',
                            "--seed"
                            ]

        # queries = []
        # for k in keys_of_interest:
        #     logger.debug("{}, {}, {}".format(self[k], type(self[k]), k))
        #     if self[k] is None:
        #         str_k = "'None'"
        #     else:
        #         str_k = self[k]
        #
        #     query = "(df['{}']=={})".format(k, str_k)
        #     queries.append(query)
        #
        # s_query = " & ".join(queries)
        # s_eval = "df[({})]".format(s_query)
        # line_of_interest = eval(s_eval)
        # logger.debug(line_of_interest)
        # logger.debug(s_eval)
        #
        # assert len(line_of_interest) == 1, "The parameters doesn't allow to discriminate only one pre-trained model in directory"
        line_of_interest = get_line_of_interest(df, keys_of_interest, self)
        self["input_model_path"] = self["--input-dir"] / line_of_interest["output_file_modelprinter"][0]

    def __init_model_path_compressed_model(self):
        df = get_df(self["--from-compressed-model"])
        keys_of_interest = ['--cifar10',
                            '--cifar10-vgg19',
                            '--cifar100',
                            '--cifar100-vgg19',
                            # '--delta-threshold',
                            '--hierarchical',
                            '--mnist',
                            '--mnist-lenet',
                            '--nb-iteration-palm',
                            '--sparsity-factor',
                            '--svhn',
                            '--svhn-vgg19',
                            '--test-data',
                            '--test-model',
                            "--nb-factor",
                            '--cifar100-resnet50',
                            '--cifar100-resnet20',
                            '--cifar100-resnet50-new',
                            '--cifar100-resnet20-new',
                            "--seed",
                            "--max-cum-batch-size",
                            "--batch-size",
                            "--nb-epochs",
                            "--full-model-approx",
                            "--only-one-batch",
                            "--activations",
                            ]

        dct_mapping_keys = {
            "--sparsity-factor": "--sparsity-factor-old",
            "--nb-iteration-palm": "--nb-iteration-palm-old",
            '--delta-threshold': "--delta-threshold-old",
            "--hierarchical": "--hierarchical-old",
            "--nb-factor": "--nb-factor-old"
        }

        line_of_interest = get_line_of_interest(df, keys_of_interest, self, dct_mapping_key_of_interest_to_dict=dct_mapping_keys)
        self["input_model_path_compressed_model"] = self["--from-compressed-model"] / line_of_interest["output_file_modelprinter"][0]


def get_params_optimizer():
    # designed using cyclical learning rate with evaluation of different learning rates on 10 epochs


    if paraman["--mnist-lenet"]:
        param_train_dataset = Mnist.get_model_param_training()
        str_data_param = "mnist"
        str_model_param = "lenet"
    elif paraman["--mnist-500"]:
        param_train_dataset = Mnist.get_model_param_training("mnist_500")
        str_data_param = ""
        str_model_param = ""
    elif paraman["--cifar10-vgg19"]:
        param_train_dataset = Cifar10.get_model_param_training()
        str_data_param = "cifar10"
        str_model_param = "vgg19"
    elif paraman["--cifar100-vgg19"]:
        param_train_dataset = Cifar100.get_model_param_training()
        str_data_param = "cifar100"
        str_model_param = "vgg19"
    elif paraman["--cifar100-resnet20"]:
        param_train_dataset = Cifar100.get_model_param_training("cifar100_resnet")
        str_data_param = "cifar100"
        str_model_param = "resnet20"
    elif paraman["--cifar100-resnet50"]:
        param_train_dataset = Cifar100.get_model_param_training("cifar100_resnet")
        str_data_param = "cifar100"
        str_model_param = "resnet50"
    elif paraman["--cifar100-resnet20-new"]:
        param_train_dataset = Cifar100.get_model_param_training("cifar100_resnet")
        str_data_param = "cifar100"
        str_model_param = "resnet20"
    elif paraman["--cifar100-resnet50-new"]:
        param_train_dataset = Cifar100.get_model_param_training("cifar100_resnet")
        str_data_param = "cifar100"
        str_model_param = "resnet50"
    elif paraman["--svhn-vgg19"]:
        param_train_dataset = Svhn.get_model_param_training()
        str_data_param = "svhn"
        str_model_param = "vgg19"
    elif paraman["--test-model"]:
        param_train_dataset = Test.get_model_param_training()
        str_data_param = ""
        str_model_param = ""
    else:
        raise NotImplementedError("No dataset specified.")

    if paraman["--tucker"]:
        str_method = "tucker_sparse_facto"
    elif paraman["palm"]:

        str_method = "palm"
    else:
        str_method = "faust"

    sparsity = int(paraman['--sparsity-factor'])
    try:
        nb_fac = int(paraman["--nb-factor"])
    except:
        nb_fac = None

    # hierarchical = paraman["--hierarchical"]
    # str_hierarchical = " H" if hierarchical else ""

    params_optimizer = param_train_dataset.params_optimizer

    if paraman["--lr-configuration-file"] is not None and paraman["--lr"] is None:
        with open(str(paraman["--lr-configuration-file"]), 'r') as f:
            dct_config_lr = yaml.full_load(f)

        params_optimizer["lr"] = load_function_lr(dct_res_new=dct_config_lr,
                         dataset=str_data_param,
                         model=str_model_param,
                         compression=str_method,
                         sparsity_value_palm=sparsity,
                         nb_fac_palm=nb_fac,
                         only_mask=paraman["--only-mask"])
    else:
        params_optimizer["lr"] = paraman["--lr"] if paraman["--lr"] is not None else params_optimizer["lr"]

    model_compilation_params = {
        "loss": param_train_dataset.loss,
        "optimizer": param_train_dataset.optimizer(**params_optimizer),
        "metrics": ['categorical_accuracy']
    }
    return model_compilation_params, param_train_dataset


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

def count_models_parameters(base_model, new_model, dct_name_compression):
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
        if any(isinstance(compressed_layer, _class) for _class in (Dense, Conv2D, SparseFactorisationDense, SparseFactorisationConv2D, TuckerSparseFactoLayerConv)):
            base_layer = base_model.layers[idx_layer]
            log_memory_usage("Start secondary loop")

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

            if not paraman["--tucker"]:
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


def compress_and_evaluate_model(base_model, model_compilation_params, x_train, y_train, x_val, y_val, x_test, y_test):
    if paraman["--tucker"]:
        layer_replacer = LayerReplacerSparseFactoTuckerFaust(keep_last_layer=paraman["--keep-last-layer"], sparse_factorizer=Faustizer(),
                                           path_checkpoint_file=paraman["input_model_path"], keep_first_layer=paraman["--keep-first-layer"], only_dense=paraman["--only-dense"])
    elif paraman["faust"]:
        layer_replacer = LayerReplacerFaust(keep_last_layer=paraman["--keep-last-layer"], sparse_factorizer=Faustizer(),
                                           only_mask=paraman["--only-mask"], path_checkpoint_file=paraman["input_model_path"],
                                           intertwine_batchnorm=paraman["--batchnorm"], keep_first_layer=paraman["--keep-first-layer"], only_dense=paraman["--only-dense"])
    elif paraman["palm"]:
        layer_replacer = LayerReplacerPalm(keep_last_layer=paraman["--keep-last-layer"], sparse_factorizer=Palminizer(),
                                            only_mask=paraman["--only-mask"], path_checkpoint_file=paraman["input_model_path"],
                                            intertwine_batchnorm=paraman["--batchnorm"], keep_first_layer=paraman["--keep-first-layer"], only_dense=paraman["--only-dense"])
    else:
        raise NotImplementedError("Unknown method")

    layer_replacer.load_dct_name_compression()
    new_model = layer_replacer.transform(base_model)
    # new_model = base_model
    new_model.compile(**model_compilation_params)
    test_score_compressed, test_acc_compressed = new_model.evaluate(x_test, y_test, verbose=0)
    train_score_compressed, train_acc_compressed = new_model.evaluate(x_train, y_train, verbose=0)
    val_score_compressed, val_acc_compressed = new_model.evaluate(x_val, y_val, verbose=0)
    actual_learning_rate = K.eval(new_model.optimizer.lr)

    dct_results = {
        "actual-lr": actual_learning_rate,
        "test_accuracy_compressed_model": test_acc_compressed,
        "test_loss_compressed_model": test_score_compressed,
        "train_accuracy_compressed_model": train_acc_compressed,
        "train_loss_compressed_model": train_score_compressed,
        "val_accuracy_compressed_model": val_acc_compressed,
        "val_loss_compressed_model": val_score_compressed,
    }
    resprinter.add(dct_results)

    count_models_parameters(base_model, new_model, layer_replacer.dct_name_compression)

    return new_model


def get_and_evaluate_base_model(model_compilation_params, x_train, y_train, x_val, y_val, x_test, y_test):
    if paraman["--from-compressed-model"] is None:
        base_model = paraman.get_model()
    else:
        base_model = load_model(paraman["input_model_path_compressed_model"], custom_objects={"SparseFactorisationConv2D": SparseFactorisationConv2D,
                                                                             "SparseFactorisationDense": SparseFactorisationDense,
                                                                             "categorical_cross_entropy_from_logits": categorical_cross_entropy_from_logits})



    base_model.compile(**model_compilation_params)

    test_score_base, test_acc_base = base_model.evaluate(x_test, y_test, verbose=0)
    train_score_base, train_acc_base = base_model.evaluate(x_train, y_train, verbose=0)
    val_score_base, val_acc_base = base_model.evaluate(x_val, y_val, verbose=0)

    dct_results = {
        "test_accuracy_base_model": test_acc_base,
        "test_loss_base_model": test_score_base,
        "train_accuracy_base_model": train_acc_base,
        "train_loss_base_model": train_score_base,
        "val_accuracy_base_model": val_acc_base,
        "val_loss_base_model": val_score_base,
    }
    resprinter.add(dct_results)

    return base_model


def get_or_load_new_model(model_compilation_params, x_train, y_train, x_val, y_val, x_test, y_test):
    if os.path.exists(paraman["output_file_notfinishedprinter"]) and \
            os.path.exists(paraman["output_file_modelprinter"]) and \
            os.path.exists(paraman["output_file_resprinter"]) and \
            os.path.exists(paraman["output_file_csvcbprinter"]):

        df_history = pd.read_csv(paraman["output_file_csvcbprinter"])
        init_nb_epoch = df_history["epoch"].max() - 1
        logger.info(f"Restart from iteration {init_nb_epoch}")

        df = pd.read_csv(paraman["output_file_resprinter"])

        # get all results from before
        dct_results = {}
        for header in lst_results_header:
            if not header in df.columns:
                continue
            dct_results[header] = df[header].values[0]
        resprinter.add(dct_results)

        new_model = keras.models.load_model(paraman["output_file_modelprinter"], custom_objects={'SparseFactorisationConv2D': SparseFactorisationConv2D,
                                                                                                 "SparseFactorisationDense": SparseFactorisationDense,
                                                                                                 'TuckerSparseFactoLayerConv': TuckerSparseFactoLayerConv,
                                                                                                 "TuckerLayerConv": TuckerLayerConv,
                                                                                                 'categorical_cross_entropy_from_logits': categorical_cross_entropy_from_logits
                                                                                                 })
    else:
        # Base Model #
        base_model = get_and_evaluate_base_model(model_compilation_params, x_train=x_train, y_train=y_train,
                                                 x_val=x_val, y_val=y_val, x_test=x_test, y_test=y_test)

        # New model compression #
        new_model = compress_and_evaluate_model(base_model, model_compilation_params,
                                                x_train=x_train, y_train=y_train,
                                                x_val=x_val, y_val=y_val, x_test=x_test, y_test=y_test)
        del base_model
        init_nb_epoch = 0

    return new_model, init_nb_epoch


def define_callbacks(param_train_dataset, x_train):
    call_backs = []

    model_checkpoint_callback = SafeModelCheckpoint(str(paraman["output_file_modelprinter"]), monitor='val_loss', verbose=0, save_best_only=False,
                                                                save_weights_only=False, mode='auto', period=1)
    call_backs.append(model_checkpoint_callback)

    actual_min_lr = param_train_dataset.min_lr if paraman["--min-lr"] is None else paraman["--min-lr"]
    actual_max_lr = param_train_dataset.max_lr if paraman["--max-lr"] is None else paraman["--max-lr"]

    if paraman["--use-clr"]:
        clr_cb = CyclicLR(base_lr=actual_min_lr,
                          max_lr=actual_max_lr,
                          step_size=(paraman["--epoch-step-size"] * (x_train.shape[0] // param_train_dataset.batch_size)),
                          logrange=paraman["--logrange-clr"])
        call_backs.append(clr_cb)

    csvcallback = CSVLoggerByBatch(str(paraman["output_file_csvcbprinter"]), n_batch_between_display=100, separator=',', append=True)
    call_backs.append(csvcallback)

    csvepochcallback = CSVLogger(filename=paraman["output_file_csvcbprinter_epoch"], append=True)
    call_backs.append(csvepochcallback)

    dct_results = {
        "actual-batch-size": param_train_dataset.batch_size,
        "actual-nb-epochs": param_train_dataset.epochs if paraman["--nb-epoch"] is None else paraman["--nb-epoch"],
        "actual-min-lr": actual_min_lr,
        "actual-max-lr": actual_max_lr,
    }

    resprinter.add(dct_results)

    return call_backs


def fit_new_model(new_model, param_train_dataset, init_nb_epoch, call_backs, x_train, y_train, x_test, y_test, x_val, y_val):
    open(paraman["output_file_notfinishedprinter"], 'w').close()
    # batch_size = param_train_dataset.batch_size if paraman["--batch-size"] is None else paraman["--batch-size"]

    new_model.fit(param_train_dataset.image_data_generator.flow(x_train, y_train, batch_size=param_train_dataset.batch_size),
         epochs=(param_train_dataset.epochs if paraman["--nb-epoch"] is None else paraman["--nb-epoch"]) - init_nb_epoch,
         # epochs=2 - init_nb_epoch,
         verbose=2,
         # validation_data=(x_val, y_val),
         callbacks=param_train_dataset.callbacks + call_backs)


    test_score_finetuned, test_acc_finetuned = new_model.evaluate(x_test, y_test, verbose=0)
    train_score_finetuned, train_acc_finetuned = new_model.evaluate(x_train, y_train, verbose=0)
    val_score_finetuned, val_acc_finetuned = new_model.evaluate(x_val, y_val, verbose=0)

    if os.path.exists(paraman["output_file_notfinishedprinter"]):
        os.remove(paraman["output_file_notfinishedprinter"])

    dct_results = {
        "test_accuracy_finetuned_model": test_acc_finetuned,
        "test_loss_finetuned_model": test_score_finetuned,
        "train_accuracy_finetuned_model": train_acc_finetuned,
        "train_loss_finetuned_model": train_score_finetuned,
        "val_accuracy_finetuned_model": val_acc_finetuned,
        "val_loss_finetuned_model": val_score_finetuned,
        # "actual_batch_size": batch_size
    }

    resprinter.add(dct_results)

def main():

    # Params optimizer #
    model_compilation_params, param_train_dataset = get_params_optimizer()
    # Dataset #
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = get_dataset()
    # Do compression or load #
    new_model, init_nb_epoch = get_or_load_new_model(model_compilation_params, x_train=x_train, y_train=y_train,
                                                 x_val=x_val, y_val=y_val, x_test=x_test, y_test=y_test)
    # Callbacks definition #
    call_backs = define_callbacks(param_train_dataset, x_train)
    # Write results before finetuning #
    resprinter.print()
    # New model Finetuning #
    fit_new_model(new_model, param_train_dataset, init_nb_epoch, call_backs, x_train=x_train, y_train=y_train,
                                                 x_val=x_val, y_val=y_val, x_test=x_test, y_test=y_test)


if __name__ == "__main__":
    logger.info("Command line: " + " ".join(sys.argv))
    log_memory_usage("Memory at startup")
    arguments = docopt.docopt(__doc__)
    paraman = ParameterManagerPalminizeFinetune(arguments)
    set_random_seed(paraman["seed"])
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