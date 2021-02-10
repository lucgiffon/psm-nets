"""
This script is for running compression of convolutional networks using tucker decomposition.

Usage:
    script.py tucker [-h] [-v|-vv] [--lr-configuration-file path] [--seed int] [--only-dense] [--rank-percentage-dense float] [--batch-size int] [--train-val-split float] [--rank-percentage-conv float] [--rank-percentage float] [--keep-first-layer] [--keep-last-layer] [--lr float] [--nb-epoch int] [--use-clr] [--min-lr float] [--max-lr float] [--epoch-step-size int] (--mnist|--svhn|--cifar10|--cifar100|--test-data) [--cifar100-resnet50-new|--cifar100-resnet20-new|--cifar100-resnet50|--cifar100-resnet20|--mnist-500|--mnist-lenet|--test-model|--cifar10-vgg19|--cifar100-vgg19|--svhn-vgg19]
    script.py tensortrain [-h] [-v|-vv] [--lr-configuration-file path] [--seed int] [--rank0-1-conv] [--only-dense] [--batch-size int] [--use-pretrained] [--train-val-split float] [--rank-value int] [--order int] [--keep-first-layer] [--keep-last-layer] [--lr float] [--nb-epoch int] [--use-clr] [--min-lr float] [--max-lr float] [--epoch-step-size int] (--mnist|--svhn|--cifar10|--cifar100|--test-data) [--cifar100-resnet50-new|--cifar100-resnet20-new|--cifar100-resnet50|--cifar100-resnet20|--mnist-500|--mnist-lenet|--test-model|--cifar10-vgg19|--cifar100-vgg19|--svhn-vgg19]
    script.py deepfried [-h] [-v|-vv] [--lr-configuration-file path] [--seed int] [--only-dense] [--keep-last-layer] [--batch-size int] [--train-val-split float] [--keep-first-layer] [--nb-stack int] [--lr float] [--nb-epoch int] [--use-clr] [--min-lr float] [--max-lr float] [--epoch-step-size int] (--mnist|--svhn|--cifar10|--cifar100|--test-data) [--cifar100-resnet50-new|--cifar100-resnet20-new|--cifar100-resnet50|--cifar100-resnet20|--mnist-500|--mnist-lenet|--test-model|--cifar10-vgg19|--cifar100-vgg19|--svhn-vgg19]
    script.py magnitude [-h] [-v|-vv] [--lr-configuration-file path] [--seed int] --final-sparsity float [--only-dense] [--batch-size int] [--train-val-split float] [--keep-last-layer] [--keep-first-layer] [--lr float] [--nb-epoch int] [--use-clr] [--min-lr float] [--max-lr float] [--epoch-step-size int] (--mnist|--svhn|--cifar10|--cifar100|--test-data) [--cifar100-resnet50-new|--cifar100-resnet20-new|--cifar100-resnet50|--cifar100-resnet20|--mnist-500|--mnist-lenet|--test-model|--cifar10-vgg19|--cifar100-vgg19|--svhn-vgg19]
    script.py random [-h] [-v|-vv] [--lr-configuration-file path] [--seed int] --sparsity-factor int [--nb-factor int] [--batch-size int] [--train-val-split float] [--only-dense] [--keep-last-layer] [--keep-first-layer] [--lr float] [--nb-epoch int] [--use-clr] [--min-lr float] [--max-lr float] [--epoch-step-size int] (--mnist|--svhn|--cifar10|--cifar100|--test-data) [--cifar100-resnet50-new|--cifar100-resnet20-new|--cifar100-resnet50|--cifar100-resnet20|--mnist-500|--mnist-lenet|--test-model|--cifar10-vgg19|--cifar100-vgg19|--svhn-vgg19]

Options:
  -h --help                             Show this screen.
  -vv                                   Set verbosity to debug.
  -v                                    Set verbosity to info.
  --train-val-split float               Tells the proportion of validation data. If not specified, validation data is test data.
  --seed int                            Tell the seed to use in the experiment.
  --lr-configuration-file path          Tell the path to the lr configuration file

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

Tucker specific option:
    --rank-percentage-dense float       Tell tucker to replace Dense layers by low rank decomposition with rank equal percentage of base rank
    --rank-percentage-conv float        Tell tucker to use a percentage of in_channel and out_channel to determine the rank in convolution and not VBMF.
    --rank-percentage float             Tell tucker to use the same rank percentage for dense and conv

Tensortrain specific option:
    --rank-value int                    The values for r0, r1 r... rk (exemple: 2, 4, 6)
    --order int                         The value for k (number of cores)
    --use-pretrained                    Tell the layer replacer to use the decomposition of the initial weights.
    --rank0-1-conv                      Tell to use a r0 value equal to 1 in convolutional layers. For fair comparison in term of input size.

DeepFried specific option:
    --nb-stack int                      The values for r0, r1 r... rk (exemple: 2, 4, 6)

Magnitude pruning specific option:
    --final-sparsity float              The final sparsity ratio: proportion of zero parameters.

Random Sparse Facto options:
    --sparsity-factor int                 Integer coefficient from which is computed the number of value in each factor.
    --nb-factor int                       Tells the number of sparse factor for palm

Finetuning options:
    --lr float                          Overide learning rate for optimization
    --min-lr float                      Set min lr if use-clr is used learning rate for optimization
    --max-lr float                      Set max lr if use-clr is used learning rate for optimization
    --epoch-step-size int               Number of epoch before an half cycle.
    --use-clr                           Tells to use cyclical learning rate instead of standard learning rate, lr and clr can't be set together
    --nb-epoch int                      Overide the number of epochs
    --batch-size int                    Use specified batch size
"""
import logging
mpl_logger = logging.getLogger("matplotlib")
mpl_logger.setLevel(logging.WARNING)
from tensorflow import set_random_seed
import yaml
from sklearn.model_selection import train_test_split

import sys
import time
import numpy as np
import keras as base_keras
import docopt
import os
from collections import defaultdict
import pandas as pd
from pathlib import Path
from keras.models import Model
from keras.layers import Conv2D, Dense
import zlib
import tensorflow as tf
from tensorflow_model_optimization.python.core.sparsity.keras.pruning_wrapper import PruneLowMagnitude

from palmnet.core.layer_replacer_TT import LayerReplacerTT
from palmnet.core.layer_replacer_deepfried import LayerReplacerDeepFried
from palmnet.core.layer_replacer_magnitude_pruning import LayerReplacerMagnitudePruning
from palmnet.core.layer_replacer_random_sparse_facto import LayerReplacerRandomSparseFacto
from palmnet.core.layer_replacer_tucker import LayerReplacerTucker
from palmnet.core.randomizer import Randomizer
from palmnet.data import Mnist, Cifar10, Cifar100, Svhn, Test
from palmnet.experiments.utils import ResultPrinter, ParameterManager
from palmnet.layers.fastfood_layer_conv import FastFoodLayerConv
from palmnet.layers.fastfood_layer_dense import FastFoodLayerDense
from palmnet.layers.low_rank_dense_layer import LowRankDense
from palmnet.layers.sparse_facto_conv2D_masked import SparseFactorisationConv2D
from palmnet.layers.sparse_facto_dense_masked import SparseFactorisationDense
from palmnet.layers.tt_layer_conv import TTLayerConv
from palmnet.layers.tt_layer_dense import TTLayerDense
from palmnet.layers.tucker_layer import TuckerLayerConv
from palmnet.utils import CyclicLR, CSVLoggerByBatch, get_nb_learnable_weights_from_model, get_nb_learnable_weights, SafeModelCheckpoint, translate_keras_to_tf_model, DummyWith, load_function_lr
from skluc.utils import logger, log_memory_usage
import keras.backend as K
from tensorflow_model_optimization.sparsity import keras as sparsity
# from tensorflow_model_optimization.python.core.api.sparsity import keras


lst_results_header = [
    "decomposition_time",
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
    "base_model_nb_param",
    "new_model_nb_param",
    "actual-lr",
    "actual-batch-size",
    "actual-nb-epochs",
    "actual-min-lr",
    "actual-max-lr",
]


class ParameterManagerTensotrainAndTuckerDecomposition(ParameterManager):
    def __init__(self, dct_params, **kwargs):
        super().__init__(self, **dct_params, **kwargs)
        self.__init_seed()

        self["--min-lr"] = float(self["--min-lr"]) if self["--min-lr"] is not None else None
        self["--max-lr"] = float(self["--max-lr"]) if self["--max-lr"] is not None else None
        self["--lr"] = float(self["--lr"]) if self["--lr"] is not None else None
        self["--nb-epoch"] = int(self["--nb-epoch"]) if self["--nb-epoch"] is not None else None
        self["--epoch-step-size"] = int(self["--epoch-step-size"]) if self["--epoch-step-size"] is not None else None

        # tensortrain parameters
        self["--rank-value"] = int(self["--rank-value"]) if self["--rank-value"] is not None else None
        self["--order"] = int(self["--order"]) if self["--order"] is not None else None

        # tucker parameters
        self["--rank-percentage-dense"] = float(self["--rank-percentage-dense"]) if self["--rank-percentage-dense"] is not None else None
        self["--rank-percentage-conv"] = float(self["--rank-percentage-conv"]) if self["--rank-percentage-conv"] is not None else None
        self["--rank-percentage"] = float(self["--rank-percentage"]) if self["--rank-percentage"] is not None else None

        assert (self["--rank-percentage-conv"] is None and self["--rank-percentage-dense"] is None) or self["--rank-percentage"] is None, "--rank-percentage and --rank-percentage-conv or dense can't be set together"

        if self["--rank-percentage"] is not None:
            self["actual-rank-percentage-dense"] = self["--rank-percentage"]
            self["actual-rank-percentage-conv"] = self["--rank-percentage"]
        else:
            self["actual-rank-percentage-dense"] = self["--rank-percentage-dense"]
            self["actual-rank-percentage-conv"] = self["--rank-percentage-conv"]

        self["--nb-stack"] = int(self["--nb-stack"]) if self["--nb-stack"] is not None else None

        self["--final-sparsity"] = float(self["--final-sparsity"]) if self["--final-sparsity"] is not None else None

        self["--sparsity-factor"] = int(self["--sparsity-factor"]) if self["--sparsity-factor"] is not None else None
        self["--nb-factor"] = int(self["--nb-factor"]) if self["--nb-factor"] is not None else None

        if "--train-val-split" in self.keys() and self["--train-val-split"] is not None:
            self["--train-val-split"] = float(self["--train-val-split"]) if self["--train-val-split"] is not None else None
            assert 0 <= self["--train-val-split"] <= 1, f"Train-val split should be comprise between 0 and 1. {self['--train-val-split']}"

        self["--batch-size"] = int(self["--batch-size"]) if self["--batch-size"] is not None else None

        self["--lr-configuration-file"] = Path(self["--lr-configuration-file"]) if self["--lr-configuration-file"] is not None else None

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

        if self["--seed"] is not None:
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


def get_params_optimizer():

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

    if paraman["tucker"]:
        str_method = "tucker"
    elif paraman["tensortrain"]:
        str_method = "tensortrain"
    elif paraman["deepfried"]:
        str_method = "deepfried"
    elif paraman["magnitude"]:
        str_method = "magnitude"
    elif paraman["random"]:
        str_method = "random"
    else:
        raise ValueError("Unknown compression method")

    params_optimizer = param_train_dataset.params_optimizer

    # str_keep_first = "-keep_first" if paraman["--keep-first-layer"] else ""
    # str_config_for_lr = f"{str_data_param}-{str_model_param}-{str_method}" + str_keep_first

    if paraman["--lr-configuration-file"] is not None:
        with open(str(paraman["--lr-configuration-file"]), 'r') as f:
            dct_config_lr = yaml.full_load(f)

        params_optimizer["lr"] = load_function_lr(dct_res_new=dct_config_lr,
                                                  dataset=str_data_param,
                                                  model=str_model_param,
                                                  compression=str_method,
                                                  sparsity_value_magnitude=paraman["--final-sparsity"],
                                                  sparsity_value_random=paraman["--sparsity-factor"],
                                                  nb_fac_random=paraman["--nb-factor"],
                                                  order_value_tensortrain=paraman["--order"],
                                                  rank_value_tensortrain=paraman["--rank-value"],
                                                  rank_value_tucker=paraman["--rank-percentage-dense"])
    else:
        params_optimizer["lr"] = paraman["--lr"] if paraman["--lr"] is not None else params_optimizer["lr"]

    dct_optimizer = {
        "RMSProp": keras.optimizers.RMSprop,
        "RMSprop": keras.optimizers.RMSprop,
        "Adam": keras.optimizers.Adam,
        "SGD": keras.optimizers.SGD
    }

    model_compilation_params = {
        "loss": param_train_dataset.loss,
        "optimizer": dct_optimizer[param_train_dataset.optimizer.__name__](**params_optimizer),
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

def define_callbacks(param_train_dataset, x_train):
    call_backs = []

    model_checkpoint_callback = SafeModelCheckpoint(str(paraman["output_file_modelprinter"]), monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto',
                                                    period=1)
    call_backs.append(model_checkpoint_callback)

    actual_min_lr = param_train_dataset.min_lr if paraman["--min-lr"] is None else paraman["--min-lr"]
    actual_max_lr = param_train_dataset.max_lr if paraman["--max-lr"] is None else paraman["--max-lr"]

    if paraman["--use-clr"]:
        clr_cb = CyclicLR(base_lr=actual_min_lr,
                          max_lr=actual_max_lr,
                          step_size=(paraman["--epoch-step-size"] * (x_train.shape[0] // param_train_dataset.batch_size)),
                          logrange=True)
        call_backs.append(clr_cb)

    csvcallback = CSVLoggerByBatch(str(paraman["output_file_csvcbprinter"]), n_batch_between_display=100, separator=',', append=True)
    call_backs.append(csvcallback)

    if paraman["magnitude"]:
        call_backs.append(sparsity.UpdatePruningStep())

    dct_results = {
        "actual-batch-size": param_train_dataset.batch_size,
        "actual-nb-epochs": param_train_dataset.epochs if paraman["--nb-epoch"] is None else paraman["--nb-epoch"],
        "actual-min-lr": actual_min_lr,
        "actual-max-lr": actual_max_lr,
    }

    resprinter.add(dct_results)

    return call_backs

def get_and_evaluate_base_model(model_compilation_params, x_train, y_train, x_val, y_val, x_test, y_test):
    base_model = paraman.get_model()
    if paraman["magnitude"]:
        base_model = translate_keras_to_tf_model(base_model)
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

def compress_and_evaluate_model(base_model, model_compilation_params, param_train_dataset, x_train, y_train, x_val, y_val, x_test, y_test):
    if paraman["tucker"]:
        layer_replacer = LayerReplacerTucker(keep_last_layer=paraman["--keep-last-layer"], keep_first_layer=paraman["--keep-first-layer"], only_dense=paraman["--only-dense"],
                                             rank_percentage_dense=paraman["actual-rank-percentage-dense"], rank_percentage_conv=paraman["actual-rank-percentage-conv"])
    elif paraman["tensortrain"]:
        layer_replacer = LayerReplacerTT(keep_last_layer=paraman["--keep-last-layer"], keep_first_layer=paraman["--keep-first-layer"], only_dense=paraman["--only-dense"],
                                         rank_value=paraman["--rank-value"], order=paraman["--order"], use_pretrained=paraman["--use-pretrained"], tt_rank0_conv_1=paraman["--rank0-1-conv"])
    elif paraman["deepfried"]:
        layer_replacer = LayerReplacerDeepFried(keep_last_layer=paraman["--keep-last-layer"], keep_first_layer=paraman["--keep-first-layer"],
                                                nb_stack=paraman["--nb-stack"], only_dense=paraman["--only-dense"])
    elif paraman["random"]:
        randomizer = Randomizer(sparsity_fac=paraman["--sparsity-factor"],
                                nb_factor=paraman["--nb-factor"])

        layer_replacer = LayerReplacerRandomSparseFacto(only_mask=True, sparse_factorizer=randomizer,
                                                        keep_last_layer=paraman["--keep-last-layer"],
                                                        keep_first_layer=paraman["--keep-first-layer"],
                                                        only_dense=paraman["--only-dense"]
                                                        )
    elif paraman["magnitude"]:
        # base_model = translate_keras_to_tf_model(base_model)
        epochs = param_train_dataset.epochs if paraman["--nb-epoch"] is None else paraman["--nb-epoch"]
        end_step = np.ceil(1.0 * x_train.shape[0] / param_train_dataset.batch_size).astype(np.int32) * epochs
        layer_replacer = LayerReplacerMagnitudePruning(final_sparsity=paraman["--final-sparsity"], end_step=end_step,
                                                       keep_last_layer=paraman["--keep-last-layer"], keep_first_layer=paraman["--keep-first-layer"],
                                                       keras_module=tf.keras, only_dense=paraman["--only-dense"])
    else:
        raise ValueError("Unknown compression method.")

    start_replace = time.time()
    new_model = layer_replacer.fit_transform(base_model)
    stop_replace = time.time()
    new_model.compile(**model_compilation_params)
    test_score_compressed, test_acc_compressed = new_model.evaluate(x_test, y_test, verbose=0)
    train_score_compressed, train_acc_compressed = new_model.evaluate(x_train, y_train, verbose=0)
    val_score_compressed, val_acc_compressed = new_model.evaluate(x_val, y_val, verbose=0)
    actual_learning_rate = K.eval(new_model.optimizer.lr)

    dct_results = {
        "actual-lr": actual_learning_rate,
        "decomposition_time": stop_replace - start_replace,
        "test_accuracy_compressed_model": test_acc_compressed,
        "test_loss_compressed_model": test_score_compressed,
        "train_accuracy_compressed_model": train_acc_compressed,
        "train_loss_compressed_model": train_score_compressed,
        "val_accuracy_compressed_model": val_acc_compressed,
        "val_loss_compressed_model": val_score_compressed,
    }
    resprinter.add(dct_results)

    return new_model


def count_models_parameters(new_model, base_model=None):
    if base_model is None:
        df_layerby_layer = pd.read_csv(paraman["output_file_layerbylayer"])
        new_model_nb_param = get_nb_learnable_weights_from_model(new_model, nnz=True)
        dct_results_param = {
            "new_model_nb_param": new_model_nb_param
        }
        resprinter.add(dct_results_param)
    else:
        df_layerby_layer = None
        new_model_nb_param = get_nb_learnable_weights_from_model(new_model, nnz=True)
        base_model_nb_param = get_nb_learnable_weights_from_model(base_model, nnz=True)
        dct_results_param = {
            "base_model_nb_param": base_model_nb_param,
            "new_model_nb_param": new_model_nb_param
        }
        resprinter.add(dct_results_param)

        if len(base_model.layers) < len(new_model.layers):
            base_model = Model(inputs=base_model.inputs, outputs=base_model.outputs)

    dct_results_matrices = defaultdict(lambda: [])

    for idx_layer, compressed_layer in enumerate(new_model.layers):

        if any(isinstance(compressed_layer, _class) for _class in (PruneLowMagnitude, TuckerLayerConv,
                                                                   TTLayerConv, TTLayerDense, LowRankDense, FastFoodLayerDense, FastFoodLayerConv,
                                                                   SparseFactorisationDense, SparseFactorisationConv2D, tf.keras.layers.Conv2D, tf.keras.layers.Dense,
                                                                   Conv2D, Dense)):
            if df_layerby_layer is None:
                base_layer = base_model.layers[idx_layer]
                row_layer = None
            else:
                base_layer = None
                row_layer = df_layerby_layer[df_layerby_layer["idx-layer"] == idx_layer]
            log_memory_usage("Start secondary loop")

            # get informations to identify the layer (and do cross references)
            dct_results_matrices["idx-expe"].append(paraman["identifier"])
            dct_results_matrices["hash"].append(paraman["hash"])
            dct_results_matrices["layer-name-compressed"].append(compressed_layer.name)
            dct_results_matrices["layer-name-base"].append(base_layer.name if base_layer is not None else row_layer["layer-name-base"].values[0])
            dct_results_matrices["idx-layer"].append(idx_layer)
            dct_results_matrices["layer-class-name"].append(base_layer.__class__.__name__ if base_layer is not None else row_layer["layer-class-name"].values[0])
            dct_results_matrices["layer-compressed-class-name"].append(compressed_layer.__class__.__name__)

            # complexity analysis #
            # get nb val base layer and comrpessed layer
            if base_layer is not None:
                nb_weights_base_layer = get_nb_learnable_weights(base_layer, nnz=True)
            else:
                nb_weights_base_layer = row_layer["nb-non-zero-base"].values[0]

            dct_results_matrices["nb-non-zero-base"].append(nb_weights_base_layer)

            nb_weights_compressed_layer = get_nb_learnable_weights(compressed_layer, nnz=True)
            dct_results_matrices["nb-non-zero-compressed"].append(nb_weights_compressed_layer)
            dct_results_matrices["nb-non-zero-compression-rate"].append(nb_weights_base_layer / nb_weights_compressed_layer)

    df_results_layers = pd.DataFrame.from_dict(dct_results_matrices)
    df_results_layers.to_csv(paraman["output_file_layerbylayer"])


def load_model_from_disc():
    if paraman["magnitude"]:
        with_obj = sparsity.prune_scope
    else:
        with_obj = DummyWith

    with with_obj():
        new_model = keras.models.load_model(paraman["output_file_modelprinter"], custom_objects={
            'TuckerLayerConv': TuckerLayerConv,
            'LowRankDense': LowRankDense,
            'TTLayerConv': TTLayerConv,
            "TTLayerDense": TTLayerDense,
            'FastFoodLayerDense': FastFoodLayerDense,
            'FastFoodLayerConv': FastFoodLayerConv,
            "SparseFactorisationDense": SparseFactorisationDense,
            "SparseFactorisationConv2D": SparseFactorisationConv2D
        })

    return new_model


def get_or_load_new_model(model_compilation_params, param_train_dataset, x_train, y_train, x_val, y_val, x_test, y_test):
    if os.path.exists(paraman["output_file_notfinishedprinter"]):
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

        # if paraman["tucker"]:
        #     new_model = keras.models.load_model(paraman["output_file_modelprinter"],custom_objects={'TuckerLayerConv': TuckerLayerConv, 'LowRankDense': LowRankDense})
        # elif paraman["tensortrain"]:
        #     new_model = keras.models.load_model(paraman["output_file_modelprinter"], custom_objects={'TTLayerConv': TTLayerConv, "TTLayerDense": TTLayerDense})
        # else:
        #     raise ValueError("Unknown compression.")
    else:
        # Base Model #
        base_model = get_and_evaluate_base_model(model_compilation_params,
                                                x_train=x_train,
                                                y_train=y_train,
                                                x_val=x_val,
                                                y_val=y_val,
                                                x_test=x_test,
                                                y_test=y_test)

        # New model compression #
        new_model = compress_and_evaluate_model(base_model, model_compilation_params, param_train_dataset,
                                                x_train=x_train,
                                                y_train=y_train,
                                                x_val=x_val,
                                                y_val=y_val,
                                                x_test=x_test,
                                                y_test=y_test)

        # count the number of parameter
        count_models_parameters(new_model, base_model)

        new_model.save(str(paraman["output_file_modelprinter"]))
        del new_model
        del base_model
        keras.backend.clear_session()
        init_nb_epoch = 0

    new_model = load_model_from_disc()

    return new_model, init_nb_epoch


def fit_new_model(new_model, param_train_dataset, init_nb_epoch, call_backs, x_train, y_train, x_test, y_test, x_val, y_val):
    open(paraman["output_file_notfinishedprinter"], 'w').close()
    batch_size = param_train_dataset.batch_size if paraman["--batch-size"] is None else paraman["--batch-size"]

    new_model.fit(param_train_dataset.image_data_generator.flow(x_train, y_train,
                                                                batch_size=batch_size),
                  epochs=(param_train_dataset.epochs if paraman["--nb-epoch"] is None else paraman["--nb-epoch"]) - init_nb_epoch,
                  # epochs=2 - init_nb_epoch,
                  verbose=2,
                  # validation_data=(x_test, y_test),
                  callbacks=param_train_dataset.callbacks + call_backs)

    if paraman["magnitude"]:
        count_models_parameters(new_model)
        # new_model = sparsity.strip_pruning(new_model)

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
        "actual_batch_size": batch_size
    }

    resprinter.add(dct_results)

def main():

    # Params optimizer #
    model_compilation_params, param_train_dataset = get_params_optimizer()
    # Dataset #
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = get_dataset()
    # Do compression or load #
    new_model, init_nb_epoch = get_or_load_new_model(model_compilation_params, param_train_dataset,
                                                x_train=x_train,
                                                y_train=y_train,
                                                x_val=x_val,
                                                y_val=y_val,
                                                x_test=x_test,
                                                y_test=y_test)
    # Write results before finetuning #
    resprinter.print()
    # Callbacks definition #
    call_backs = define_callbacks(param_train_dataset, x_train)
    # New model Finetuning #
    fit_new_model(new_model, param_train_dataset, init_nb_epoch, call_backs,
                                                x_train=x_train,
                                                y_train=y_train,
                                                x_val=x_val,
                                                y_val=y_val,
                                                x_test=x_test,
                                                y_test=y_test)


if __name__ == "__main__":
    logger.info("Command line: " + " ".join(sys.argv))
    log_memory_usage("Memory at startup")
    arguments = docopt.docopt(__doc__)
    paraman = ParameterManagerTensotrainAndTuckerDecomposition(arguments)
    set_random_seed(paraman["seed"])
    initialized_results = dict((v, None) for v in lst_results_header)
    if paraman["magnitude"]:
        keras = tf.keras
    else:
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