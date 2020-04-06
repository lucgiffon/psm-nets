"""
This script is for running compression of convolutional networks using tucker decomposition.

Usage:
    script.py tucker [-h] [-v|-vv] [--keep-first-layer] [--keep-last-layer] [--lr float] [--nb-epoch int] [--use-clr] [--min-lr float] [--max-lr float] [--epoch-step-size int] (--mnist|--svhn|--cifar10|--cifar100|--test-data) [--cifar100-resnet50|--cifar100-resnet20|--mnist-500|--mnist-lenet|--test-model|--cifar10-vgg19|--cifar100-vgg19|--svhn-vgg19]
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
import pickle
import sys
import time
import numpy as np
import keras
import docopt
import os
import pandas as pd

from palmnet.core.layer_replacer_TT import LayerReplacerTT
from palmnet.core.layer_replacer_tucker import LayerReplacerTucker
from palmnet.data import Mnist, Cifar10, Cifar100, Svhn, Test
from palmnet.experiments.utils import ResultPrinter, ParameterManagerTensotrainAndTuckerDecomposition
from palmnet.layers.tt_layer_conv import TTLayerConv
from palmnet.layers.tt_layer_dense import TTLayerDense
from palmnet.layers.tucker_layer import TuckerLayerConv
from palmnet.utils import CyclicLR, CSVLoggerByBatch
from skluc.utils import logger, log_memory_usage

lst_results_header = [
    "decomposition_time",
    "test_accuracy_base_model",
    "test_accuracy_compressed_model",
    "test_accuracy_finetuned_model",
    "test_loss_base_model",
    "test_loss_compressed_model",
    "test_loss_finetuned_model"
]

def get_params_optimizer():
    # designed using cyclical learning rate with evaluation of different learning rates on 10 epochs
    dct_config_lr = {'--cifar10---cifar10-vgg19-tensortrain-4-10': 0.0001,
 '--cifar10---cifar10-vgg19-tensortrain-4-10-keep_first': 0.0001,
 '--cifar10---cifar10-vgg19-tensortrain-4-2': 0.001,
 '--cifar10---cifar10-vgg19-tensortrain-4-2-keep_first': 0.001,
 '--cifar10---cifar10-vgg19-tensortrain-4-6': 0.0001,
 '--cifar10---cifar10-vgg19-tensortrain-4-6-keep_first': 1e-06,
 '--cifar100---cifar100-resnet20-tensortrain-4-10': 0.001,
 '--cifar100---cifar100-resnet20-tensortrain-4-10-keep_first': 0.001,
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
 '--cifar100---cifar100-vgg19-tensortrain-4-2': 0.001,
 '--cifar100---cifar100-vgg19-tensortrain-4-2-keep_first': 0.001,
 '--cifar100---cifar100-vgg19-tensortrain-4-6': 0.0001,
 '--cifar100---cifar100-vgg19-tensortrain-4-6-keep_first': 0.001,
 '--mnist---mnist-lenet-tensortrain-4-10': 1e-06,
 '--mnist---mnist-lenet-tensortrain-4-10-keep_first': 1e-05,
 '--mnist---mnist-lenet-tensortrain-4-2': 0.0001,
 '--mnist---mnist-lenet-tensortrain-4-2-keep_first': 0.0001,
 '--mnist---mnist-lenet-tensortrain-4-6': 1e-05,
 '--mnist---mnist-lenet-tensortrain-4-6-keep_first': 1e-05,
 '--svhn---svhn-vgg19-tensortrain-4-10': 0.0001,  # modified by hand
 '--svhn---svhn-vgg19-tensortrain-4-10-keep_first': 1e-05,
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
        layer_replacer = LayerReplacerTucker(keep_last_layer=paraman["--keep-last-layer"], keep_first_layer=paraman["--keep-first-layer"])
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
            new_model = keras.models.load_model(paraman["output_file_modelprinter"],custom_objects={'TuckerLayerConv':TuckerLayerConv})
        elif paraman["tensortrain"]:
            new_model = keras.models.load_model(paraman["output_file_modelprinter"], custom_objects={'TTLayerConv': TTLayerConv, "TTLayerDense": TTLayerDense})
        else:
            raise ValueError("Unknown compression.")
    else:
        # Base Model #
        base_model = get_and_evaluate_base_model(model_compilation_params, x_test, y_test)

        # New model compression #
        new_model = compress_and_evaluate_model(base_model, model_compilation_params, x_test, y_test)
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