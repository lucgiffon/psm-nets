"""
This script is for running compression of convolutional networks using tucker decomposition.

Usage:
    script.py [-h] [-v|-vv] [--keep-first-layer] [--keep-last-layer] [--lr float] [--nb-epoch int] [--use-clr] [--min-lr float] [--max-lr float] [--epoch-step-size int] (--mnist|--svhn|--cifar10|--cifar100|--test-data) [--cifar100-resnet50|--cifar100-resnet20|--mnist-500|--mnist-lenet|--test-model|--cifar10-vgg19|--cifar100-vgg19|--svhn-vgg19]

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

from palmnet.core.layer_replacer_tucker import LayerReplacerTucker
from palmnet.data import Mnist, Cifar10, Cifar100, Svhn, Test
from palmnet.experiments.utils import ResultPrinter, ParameterManagerTensotrainAndTuckerDecomposition
from palmnet.layers.tucker_layer import TuckerLayerConv
from palmnet.utils import CyclicLR, CSVLoggerByBatch
from skluc.utils import logger, log_memory_usage

lst_results_header = [
    "tucker_decomposition_time",
    "test_accuracy_base_model",
    "test_accuracy_compressed_model",
    "test_accuracy_finetuned_model"
    "test_loss_base_model",
    "test_loss_compressed_model",
    "test_loss_finetuned_model"
]

def get_params_optimizer():
    if paraman["--mnist-lenet"]:
        param_train_dataset = Mnist.get_model_param_training()
    elif paraman["--mnist-500"]:
        param_train_dataset = Mnist.get_model_param_training("mnist_500")
    elif paraman["--cifar10-vgg19"]:
        param_train_dataset = Cifar10.get_model_param_training()
    elif paraman["--cifar100-vgg19"]:
        param_train_dataset = Cifar100.get_model_param_training()
    elif paraman["--cifar100-resnet20"] or paraman["--cifar100-resnet50"]:
        param_train_dataset = Cifar100.get_model_param_training("cifar100_resnet")
    elif paraman["--svhn-vgg19"]:
        param_train_dataset = Svhn.get_model_param_training()
    elif paraman["--test-model"]:
        param_train_dataset = Test.get_model_param_training()
    else:
        raise NotImplementedError("No dataset specified.")

    params_optimizer = param_train_dataset.params_optimizer
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
    layer_replacer = LayerReplacerTucker(keep_last_layer=paraman["--keep-last-layer"], keep_first_layer=paraman["--keep-first-layer"])
    start_replace = time.time()
    new_model = layer_replacer.fit_transform(base_model)
    stop_replace = time.time()
    new_model.compile(**model_compilation_params)
    score_compressed, acc_compressed = new_model.evaluate(x_test, y_test, verbose=0)

    dct_results = {
        "tucker_decomposition_time": stop_replace - start_replace,
        "test_accuracy_compressed_model": acc_compressed,
        "test_loss_compressed_model": score_compressed
    }
    resprinter.add(dct_results)

    return new_model

def main():

    # Params optimizer #
    model_compilation_params, param_train_dataset = get_params_optimizer()

    # Dataset #
    (x_train, y_train), (x_test, y_test) = get_dataset()

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

        new_model = keras.models.load_model(paraman["output_file_modelprinter"],custom_objects={'TuckerLayerConv':TuckerLayerConv})

    else:
        # Base Model #
        base_model = get_and_evaluate_base_model(model_compilation_params, x_test, y_test)

        # New model compression #
        new_model = compress_and_evaluate_model(base_model, model_compilation_params, x_test, y_test)
        del base_model

        # Write results before finetuning #
        resprinter.print()

        init_nb_epoch = 0

    ########################
    # Callbacks definition #
    ########################
    call_backs = define_callbacks(param_train_dataset, x_train)

    ########################
    # New model Finetuning #
    ########################
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