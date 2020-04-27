"""
This script is for running compression of convolutional networks using tucker decomposition.

Usage:
    script.py tucker [-h] [-v|-vv] [--rank-percentage-dense float] [--keep-first-layer] [--keep-last-layer] [--lr float] [--nb-epoch int] [--use-clr] [--min-lr float] [--max-lr float] [--epoch-step-size int] (--mnist|--svhn|--cifar10|--cifar100|--test-data) [--cifar100-resnet50|--cifar100-resnet20|--mnist-500|--mnist-lenet|--test-model|--cifar10-vgg19|--cifar100-vgg19|--svhn-vgg19]
    script.py tensortrain [-h] [-v|-vv] [--rank-value int] [--order int] [--keep-first-layer] [--keep-last-layer] [--lr float] [--nb-epoch int] [--use-clr] [--min-lr float] [--max-lr float] [--epoch-step-size int] (--mnist|--svhn|--cifar10|--cifar100|--test-data) [--cifar100-resnet50|--cifar100-resnet20|--mnist-500|--mnist-lenet|--test-model|--cifar10-vgg19|--cifar100-vgg19|--svhn-vgg19]

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
  --cifar100-resnet20                   Use model resnet20 pretrained on cifar100.

Compression specific options:
    --keep-first-layer                  Tell the replacer to keep the first layer of the network
    --keep-last-layer                   Tell the replacer to keep the last layer (classification) of the network

Tucker specific option:
    --rank-percentage-dense float       Tell tucker to replace Dense layers by low rank decomposition with rank equal percentage of base rank
    --rank-percentage-conv float        Tell tucker to use a percentage of in_channel and out_channel to determine the rank in convolution and not VBMF.

Tensortrain specific option:
    --rank-value int                    The values for r0, r1 r... rk (exemple: 2, 4, 6)
    --order int                         The value for k (number of cores)

Finetuning options:
    --lr float                          Overide learning rate for optimization

    --min-lr float                      Set min lr if use-clr is used learning rate for optimization
    --max-lr float                      Set max lr if use-clr is used learning rate for optimization
    --epoch-step-size int               Number of epoch before an half cycle.
    --use-clr                           Tells to use cyclical learning rate instead of standard learning rate, lr and clr can't be set together

    --nb-epoch int                      Overide the number of epochs
"""
import logging
import sys
import time
import numpy as np
import keras
import docopt
import os
from collections import defaultdict
import pandas as pd
from pathlib import Path
from keras.models import Model
import zlib

from palmnet.core.layer_replacer_TT import LayerReplacerTT
from palmnet.core.layer_replacer_tucker import LayerReplacerTucker
from palmnet.data import Mnist, Cifar10, Cifar100, Svhn, Test
from palmnet.experiments.utils import ResultPrinter, ParameterManager
from palmnet.layers.low_rank_dense_layer import LowRankDense
from palmnet.layers.tt_layer_conv import TTLayerConv
from palmnet.layers.tt_layer_dense import TTLayerDense
from palmnet.layers.tucker_layer import TuckerLayerConv
from palmnet.utils import CyclicLR, CSVLoggerByBatch, get_nb_learnable_weights_from_model, get_nb_learnable_weights
from skluc.utils import logger, log_memory_usage

lst_results_header = [
    "decomposition_time",
    "test_accuracy_base_model",
    "test_accuracy_compressed_model",
    "test_accuracy_finetuned_model",
    "test_loss_base_model",
    "test_loss_compressed_model",
    "test_loss_finetuned_model",
    "base_model_nb_param,"
    "new_model_nb_param"
]


class ParameterManagerTensotrainAndTuckerDecomposition(ParameterManager):
    def __init__(self, dct_params, **kwargs):
        super().__init__(self, **dct_params, **kwargs)
        self["--min-lr"] = float(self["--min-lr"]) if self["--min-lr"] is not None else None
        self["--max-lr"] = float(self["--max-lr"]) if self["--max-lr"] is not None else None
        self["--lr"] = float(self["--lr"]) if self["--lr"] is not None else None
        self["--nb-epoch"] = int(self["--nb-epoch"]) if self["--nb-epoch"] is not None else None
        self["--epoch-step-size"] = int(self["--epoch-step-size"]) if self["--epoch-step-size"] is not None else None

        # tensortrain parameters
        self["--rank-value"] = int(self["--rank-value"]) if self["--rank-value"] is not None else None
        self["--order"] = int(self["--order"]) if self["--order"] is not None else None

        # tucker parameters
        if "--rank-percentage-dense" in self:
            self["--rank-percentage-dense"] = float(self["--rank-percentage-dense"]) if self["--rank-percentage-dense"] is not None else None
        if "--rank-percentage-conv" in self:
            self["--rank-percentage-conv"] = float(self["--rank-percentage-conv"]) if self["--rank-percentage-conv"] is not None else None

        self.__init_hash_expe()
        self.__init_output_file()

    def __init_output_file(self):
        self["output_file_resprinter"] = Path(self["hash"] + "_results.csv")
        self["output_file_modelprinter"] = Path(self["hash"] + "_model.h5")
        self["output_file_notfinishedprinter"] = Path(self["hash"] + ".notfinished")
        self["output_file_csvcbprinter"] = Path(self["hash"] + "_history.csv")
        self["output_file_layerbylayer"] = Path(self["hash"] + "_layerbylayer.csv")

    def __init_hash_expe(self):
        lst_elem_to_remove_for_hash = [
            'identifier',
            '-v',
            '--help',
            "output_file_resprinter",
            "output_file_modelprinter",
            "output_file_notfinishedprinter",
            "output_file_csvcbprinter",
            "output_file_layerbylayer"
        ]

        keys_expe = sorted(self.keys())
        any(keys_expe.remove(item) for item in lst_elem_to_remove_for_hash if item in keys_expe)
        val_expe = [self[k] for k in keys_expe]
        str_expe = [str(val) for pair in zip(keys_expe, val_expe) for val in pair]
        self["hash"] = hex(zlib.crc32(str.encode("".join(str_expe))))


def get_params_optimizer():
    # designed using cyclical learning rate with evaluation of different learning rates on 10 epochs
    dct_config_lr = {'--cifar10---cifar10-vgg19-tensortrain-4-10': 0.0001,
 '--cifar10---cifar10-vgg19-tensortrain-4-10-keep_first': 0.0001,
 '--cifar10---cifar10-vgg19-tensortrain-4-12': 0.0001,
 '--cifar10---cifar10-vgg19-tensortrain-4-12-keep_first': 0.001,
 '--cifar10---cifar10-vgg19-tensortrain-4-14': 0.0001,
 '--cifar10---cifar10-vgg19-tensortrain-4-14-keep_first': 0.0001,
 '--cifar10---cifar10-vgg19-tensortrain-4-2': 0.001,
 '--cifar10---cifar10-vgg19-tensortrain-4-2-keep_first': 0.001,
 '--cifar10---cifar10-vgg19-tensortrain-4-6': 0.0001,
 '--cifar10---cifar10-vgg19-tensortrain-4-6-keep_first': 1e-06,
 '--cifar100---cifar100-resnet20-tensortrain-4-10': 0.001,
 '--cifar100---cifar100-resnet20-tensortrain-4-10-keep_first': 0.001,
 '--cifar100---cifar100-resnet20-tensortrain-4-12': 0.001,
 '--cifar100---cifar100-resnet20-tensortrain-4-12-keep_first': 0.001,
 '--cifar100---cifar100-resnet20-tensortrain-4-14': 0.001,
 '--cifar100---cifar100-resnet20-tensortrain-4-14-keep_first': 0.001,
 '--cifar100---cifar100-resnet20-tensortrain-4-2': 0.001,
 '--cifar100---cifar100-resnet20-tensortrain-4-2-keep_first': 0.01,
 '--cifar100---cifar100-resnet20-tensortrain-4-6': 0.001,
 '--cifar100---cifar100-resnet20-tensortrain-4-6-keep_first': 0.001,
 '--cifar100---cifar100-resnet50-tensortrain-4-10': 0.001,
 '--cifar100---cifar100-resnet50-tensortrain-4-10-keep_first': 0.001,
 '--cifar100---cifar100-resnet50-tensortrain-4-2': 0.01,
 '--cifar100---cifar100-resnet50-tensortrain-4-2-keep_first': 0.01,
 '--cifar100---cifar100-resnet50-tensortrain-4-6': 0.01,
 '--cifar100---cifar100-resnet50-tensortrain-4-6-keep_first': 0.001,
 '--cifar100---cifar100-vgg19-tensortrain-4-10': 0.001,
 '--cifar100---cifar100-vgg19-tensortrain-4-10-keep_first': 0.001,
 '--cifar100---cifar100-vgg19-tensortrain-4-12': 0.0001,
 '--cifar100---cifar100-vgg19-tensortrain-4-12-keep_first': 1e-05,
 '--cifar100---cifar100-vgg19-tensortrain-4-14': 1e-05,
 '--cifar100---cifar100-vgg19-tensortrain-4-14-keep_first': 0.0001,
 '--cifar100---cifar100-vgg19-tensortrain-4-2': 0.001,
 '--cifar100---cifar100-vgg19-tensortrain-4-2-keep_first': 0.001,
 '--cifar100---cifar100-vgg19-tensortrain-4-6': 0.0001,
 '--cifar100---cifar100-vgg19-tensortrain-4-6-keep_first': 0.001,
 '--mnist---mnist-lenet-tensortrain-4-10': 1e-06,
 '--mnist---mnist-lenet-tensortrain-4-10-keep_first': 1e-05,
 '--mnist---mnist-lenet-tensortrain-4-12': 1e-05,
 '--mnist---mnist-lenet-tensortrain-4-12-keep_first': 1e-05,
 '--mnist---mnist-lenet-tensortrain-4-14': 1e-06,
 '--mnist---mnist-lenet-tensortrain-4-14-keep_first': 1e-06,
 '--mnist---mnist-lenet-tensortrain-4-2': 0.0001,
 '--mnist---mnist-lenet-tensortrain-4-2-keep_first': 0.0001,
 '--mnist---mnist-lenet-tensortrain-4-6': 1e-05,
 '--mnist---mnist-lenet-tensortrain-4-6-keep_first': 1e-05,
 '--svhn---svhn-vgg19-tensortrain-4-10': 0.1,
 '--svhn---svhn-vgg19-tensortrain-4-10-keep_first': 1e-05,
 '--svhn---svhn-vgg19-tensortrain-4-12': 1e-05,
 '--svhn---svhn-vgg19-tensortrain-4-12-keep_first': 0.0001,
 '--svhn---svhn-vgg19-tensortrain-4-14': 0.0001,
 '--svhn---svhn-vgg19-tensortrain-4-14-keep_first': 1e-05,
 '--svhn---svhn-vgg19-tensortrain-4-2': 0.0001,
 '--svhn---svhn-vgg19-tensortrain-4-2-keep_first': 0.0001,
 '--svhn---svhn-vgg19-tensortrain-4-6': 0.0001,
 '--svhn---svhn-vgg19-tensortrain-4-6-keep_first': 0.001,
 '--cifar10---cifar10-vgg19-tucker': 1e-06,
 '--cifar10---cifar10-vgg19-tucker-keep_first': 1e-05,
 '--cifar100---cifar100-resnet20-tucker': 1e-05,
 '--cifar100---cifar100-resnet20-tucker-keep_first': 1e-05,
 '--cifar100---cifar100-resnet50-tucker': 1e-05,
 '--cifar100---cifar100-resnet50-tucker-keep_first': 1e-05,
 '--cifar100---cifar100-vgg19-tucker': 1e-06,
 '--cifar100---cifar100-vgg19-tucker-keep_first': 1e-06,
 '--mnist---mnist-lenet-tucker': 1e-06,
 '--mnist---mnist-lenet-tucker-keep_first': 1e-06,
 '--svhn---svhn-vgg19-tucker': 0.0001,
 '--svhn---svhn-vgg19-tucker-keep_first': 1e-06}

    if paraman["--mnist-lenet"]:
        param_train_dataset = Mnist.get_model_param_training()
        str_data_param = "--mnist"
        str_model_param = "--mnist-lenet"
    elif paraman["--mnist-500"]:
        param_train_dataset = Mnist.get_model_param_training("mnist_500")
        str_data_param = ""
        str_model_param = ""
    elif paraman["--cifar10-vgg19"]:
        param_train_dataset = Cifar10.get_model_param_training()
        str_data_param = "--cifar10"
        str_model_param = "--cifar10-vgg19"
    elif paraman["--cifar100-vgg19"]:
        param_train_dataset = Cifar100.get_model_param_training()
        str_data_param = "--cifar100"
        str_model_param = "--cifar100-vgg19"
    elif paraman["--cifar100-resnet20"]:
        param_train_dataset = Cifar100.get_model_param_training("cifar100_resnet")
        str_data_param = "--cifar100"
        str_model_param = "--cifar100-resnet20"
    elif paraman["--cifar100-resnet50"]:
        param_train_dataset = Cifar100.get_model_param_training("cifar100_resnet")
        str_data_param = "--cifar100"
        str_model_param = "--cifar100-resnet50"
    elif paraman["--svhn-vgg19"]:
        param_train_dataset = Svhn.get_model_param_training()
        str_data_param = "--svhn"
        str_model_param = "--svhn-vgg19"
    elif paraman["--test-model"]:
        param_train_dataset = Test.get_model_param_training()
        str_data_param = ""
        str_model_param = ""
    else:
        raise NotImplementedError("No dataset specified.")

    if paraman["tucker"]:
        str_method = "tucker"
    elif paraman["tensortrain"]:
        str_method = f"tensortrain-{int(paraman['--order'])}-{int(paraman['--rank-value'])}"
    else:
        raise ValueError("Unknown compression method")


    params_optimizer = param_train_dataset.params_optimizer

    str_keep_first = "-keep_first" if paraman["--keep-first-layer"] else ""
    str_config_for_lr = f"{str_data_param}-{str_model_param}-{str_method}" + str_keep_first

    if str_config_for_lr in dct_config_lr:
        params_optimizer["lr"] = paraman["--lr"] if paraman["--lr"] is not None else dct_config_lr[str_config_for_lr]
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

    return (x_train, y_train), (x_test, y_test)

def define_callbacks(param_train_dataset, x_train):
    call_backs = []

    model_checkpoint_callback = keras.callbacks.ModelCheckpoint(str(paraman["output_file_modelprinter"]), monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto',
                                                                period=1)
    call_backs.append(model_checkpoint_callback)

    if paraman["--use-clr"]:
        clr_cb = CyclicLR(base_lr=param_train_dataset.min_lr if paraman["--min-lr"] is None else paraman["--min-lr"],
                          max_lr=param_train_dataset.max_lr if paraman["--max-lr"] is None else paraman["--max-lr"],
                          step_size=(paraman["--epoch-step-size"] * (x_train.shape[0] // param_train_dataset.batch_size)),
                          logrange=True)
        call_backs.append(clr_cb)

    csvcallback = CSVLoggerByBatch(str(paraman["output_file_csvcbprinter"]), n_batch_between_display=100, separator=',', append=True)
    call_backs.append(csvcallback)

    return call_backs

def get_and_evaluate_base_model(model_compilation_params, x_test, y_test):
    base_model = paraman.get_model()
    base_model.compile(**model_compilation_params)

    score_base, acc_base = base_model.evaluate(x_test, y_test, verbose=0)

    dct_results = {
        "test_accuracy_base_model": acc_base,
        "test_loss_base_model": score_base,
    }
    resprinter.add(dct_results)

    return base_model

def compress_and_evaluate_model(base_model, model_compilation_params, x_test, y_test):
    if paraman["tucker"]:
        layer_replacer = LayerReplacerTucker(keep_last_layer=paraman["--keep-last-layer"], keep_first_layer=paraman["--keep-first-layer"],
                                             rank_percentage_dense=paraman["--rank-percentage-dense"], rank_percentage_conv=paraman["--rank-percentage-conv"])
    elif paraman["tensortrain"]:
        layer_replacer = LayerReplacerTT(keep_last_layer=paraman["--keep-last-layer"], keep_first_layer=paraman["--keep-first-layer"],
                                         rank_value=paraman["--rank-value"], order=paraman["--order"])
    else:
        raise ValueError("Unknown compression method.")
    start_replace = time.time()
    new_model = layer_replacer.fit_transform(base_model)
    stop_replace = time.time()
    new_model.compile(**model_compilation_params)
    score_compressed, acc_compressed = new_model.evaluate(x_test, y_test, verbose=0)

    dct_results = {
        "decomposition_time": stop_replace - start_replace,
        "test_accuracy_compressed_model": acc_compressed,
        "test_loss_compressed_model": score_compressed
    }
    resprinter.add(dct_results)

    return new_model


def count_models_parameters(base_model, new_model):
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
        if any(isinstance(compressed_layer, _class) for _class in (TuckerLayerConv, TTLayerConv, TTLayerDense, LowRankDense)):
            base_layer = base_model.layers[idx_layer]
            log_memory_usage("Start secondary loop")

            # get informations to identify the layer (and do cross references)
            dct_results_matrices["idx-expe"].append(paraman["identifier"])
            dct_results_matrices["hash"].append(paraman["hash"])
            dct_results_matrices["layer-name-compressed"].append(compressed_layer.name)
            dct_results_matrices["layer-name-base"].append(base_layer.name)
            dct_results_matrices["idx-layer"].append(idx_layer)

            # complexity analysis #
            # get nb val base layer and comrpessed layer
            nb_weights_base_layer = get_nb_learnable_weights(base_layer)
            dct_results_matrices["nb-non-zero-base"].append(nb_weights_base_layer)
            nb_weights_compressed_layer = get_nb_learnable_weights(compressed_layer)
            dct_results_matrices["nb-non-zero-compressed"].append(nb_weights_compressed_layer)
            dct_results_matrices["nb-non-zero-compression-rate"].append(nb_weights_base_layer / nb_weights_compressed_layer)

    df_results_layers = pd.DataFrame.from_dict(dct_results_matrices)
    df_results_layers.to_csv(paraman["output_file_layerbylayer"])


def get_or_load_new_model(model_compilation_params, x_test, y_test):
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

        if paraman["tucker"]:
            new_model = keras.models.load_model(paraman["output_file_modelprinter"],custom_objects={'TuckerLayerConv': TuckerLayerConv, 'LowRankDense': LowRankDense})
        elif paraman["tensortrain"]:
            new_model = keras.models.load_model(paraman["output_file_modelprinter"], custom_objects={'TTLayerConv': TTLayerConv, "TTLayerDense": TTLayerDense})
        else:
            raise ValueError("Unknown compression.")
    else:
        # Base Model #
        base_model = get_and_evaluate_base_model(model_compilation_params, x_test, y_test)

        # New model compression #
        new_model = compress_and_evaluate_model(base_model, model_compilation_params, x_test, y_test)

        # count the number of parameter
        count_models_parameters(base_model, new_model)

        del base_model
        init_nb_epoch = 0

    return new_model, init_nb_epoch


def fit_new_model(new_model, param_train_dataset, init_nb_epoch, call_backs, x_train, y_train, x_test, y_test):
    open(paraman["output_file_notfinishedprinter"], 'w').close()

    new_model.fit(param_train_dataset.image_data_generator.flow(x_train, y_train, batch_size=param_train_dataset.batch_size),
         epochs=(param_train_dataset.epochs if paraman["--nb-epoch"] is None else paraman["--nb-epoch"]) - init_nb_epoch,
         # epochs=2 - init_nb_epoch,
         verbose=2,
         # validation_data=(x_test, y_test),
         callbacks=param_train_dataset.callbacks + call_backs)

    score_finetuned, acc_finetuned = new_model.evaluate(x_test, y_test, verbose=0)

    if os.path.exists(paraman["output_file_notfinishedprinter"]):
        os.remove(paraman["output_file_notfinishedprinter"])

    dct_results = {
        "test_accuracy_finetuned_model": acc_finetuned,
        "test_loss_finetuned_model": score_finetuned
    }

    resprinter.add(dct_results)

def main():

    # Params optimizer #
    model_compilation_params, param_train_dataset = get_params_optimizer()
    # Dataset #
    (x_train, y_train), (x_test, y_test) = get_dataset()
    # Do compression or load #
    new_model, init_nb_epoch = get_or_load_new_model(model_compilation_params, x_test, y_test)
    # Write results before finetuning #
    resprinter.print()
    # Callbacks definition #
    call_backs = define_callbacks(param_train_dataset, x_train)
    # New model Finetuning #
    fit_new_model(new_model, param_train_dataset, init_nb_epoch, call_backs, x_train, y_train, x_test, y_test)


if __name__ == "__main__":
    logger.info("Command line: " + " ".join(sys.argv))
    log_memory_usage("Memory at startup")
    arguments = docopt.docopt(__doc__)
    paraman = ParameterManagerTensotrainAndTuckerDecomposition(arguments)
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