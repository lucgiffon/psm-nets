"""
This script finds a palminized model with given arguments then finetune it.

Usage:
    script.py (palm|faust) --input-dir path [-h] [-v|-vv] [--seed int] [--train-val-split float] [--batchnorm] [--keep-last-layer] [--lr float] [--use-clr policy] [--min-lr float --max-lr float] [--epoch-step-size int] [--nb-epoch int] [--only-mask] [--tb] (--mnist|--svhn|--cifar10|--cifar100|--test-data) [--cifar100-resnet50|--cifar100-resnet20|--mnist-500|--mnist-lenet|--test-model|--cifar10-vgg19|--cifar100-vgg19|--svhn-vgg19] --sparsity-factor=int [--nb-iteration-palm=int] [--delta-threshold=float] [--hierarchical] [--nb-factor=int]

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
  --keep-last-layer                     Do not compress classification layer.
  --train-val-split float               Tells the proportion of validation data. If not specified, validation data is test data.


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

Palm-Specifc options:
  --sparsity-factor=int                 Integer coefficient from which is computed the number of value in each factor.
  --nb-iteration-palm=int               Number of iterations in the inner palm4msa calls. [default: 300]
  --delta-threshold=float               Threshold value before stopping palm iterations. [default: 1e-6]
  --hierarchical                        Tells if palm should use the hierarchical euristic or not. Muhc longer but better approximation results.
  --nb-factor=int                       Tells the number of sparse factor for palm
  --only-mask                           Use only sparsity mask given by palm but re-initialize weights.
  --batchnorm                           Intertwine batchnorm between factors of dense layers.
"""
import logging
import os

import docopt
import keras
import keras.backend as K
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from palmnet.core import palminizable
from palmnet.core.faustizer import Faustizer
from palmnet.core.layer_replacer_faust import LayerReplacerFaust
from palmnet.core.layer_replacer_palm import LayerReplacerPalm
from palmnet.core.palminizer import Palminizer
from palmnet.data import Mnist, Test, Svhn, Cifar100, Cifar10
from palmnet.experiments.utils import ParameterManagerPalminizeFinetune, ResultPrinter
# from palmnet.layers.sparse_tensor import SparseFactorisationDense#, SparseFactorisationConv2DDensify
from palmnet.layers.sparse_facto_conv2D_masked import SparseFactorisationConv2D
from palmnet.layers.sparse_facto_dense_masked import SparseFactorisationDense
from palmnet.utils import CSVLoggerByBatch
from palmnet.utils import CyclicLR
from skluc.utils import logger, log_memory_usage

palminizable.Palminizer = Palminizer
import sys
sys.modules["palmnet.core.palminize"] = palminizable

# noinspection PyUnreachableCode
lst_results_header = [
    "test_accuracy_finetuned_model",
    "test_accuracy_base_model",
    "test_accuracy_compressed_model",
    "test_loss_base_model",
    "test_loss_compressed_model",
    "actual-lr",
    "actual-batch-size",
    "actual-nb-epochs",
    "actual-min-lr",
    "actual-max-lr",
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
    elif paraman["--cifar100-resnet20"]:
        param_train_dataset = Cifar100.get_model_param_training("cifar100_resnet")
    elif paraman["--cifar100-resnet50"]:
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

    if paraman["--train-val-split"] is not None:
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=paraman["--train-val-split"], random_state=paraman["--seed"])
    else:
        x_val, y_val = x_test, y_test

    return (x_train, y_train), (x_val, y_val), (x_test, y_test)

def compress_and_evaluate_model(base_model, model_compilation_params, x_test, y_test):
    if paraman["palm"]:
        layer_replacer = LayerReplacerPalm(keep_last_layer=paraman["--keep-last-layer"], sparse_factorizer=Palminizer(),
                                           only_mask=paraman["--only-mask"], path_checkpoint_file=paraman["input_model_path"],
                                           intertwine_batchnorm=paraman["--batchnorm"])
    else:
        layer_replacer = LayerReplacerFaust(keep_last_layer=paraman["--keep-last-layer"], sparse_factorizer=Faustizer(),
                                           only_mask=paraman["--only-mask"], path_checkpoint_file=paraman["input_model_path"],
                                           intertwine_batchnorm=paraman["--batchnorm"])

    layer_replacer.load_dct_name_compression()
    new_model = layer_replacer.transform(base_model)
    new_model.compile(**model_compilation_params)
    score_compressed, acc_compressed = new_model.evaluate(x_test, y_test, verbose=0)
    actual_learning_rate = K.eval(new_model.optimizer.lr)

    dct_results = {
        "actual-lr": actual_learning_rate,
        "test_accuracy_compressed_model": acc_compressed,
        "test_loss_compressed_model": score_compressed
    }
    resprinter.add(dct_results)

    return new_model

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

        new_model = keras.models.load_model(paraman["output_file_modelprinter"], custom_objects={'SparseFactorisationConv2D': SparseFactorisationConv2D,
                                                                                                 "SparseFactorisationDense": SparseFactorisationDense})
    else:
        # Base Model #
        base_model = get_and_evaluate_base_model(model_compilation_params, x_test, y_test)

        # New model compression #
        new_model = compress_and_evaluate_model(base_model, model_compilation_params, x_test, y_test)
        del base_model
        init_nb_epoch = 0

    return new_model, init_nb_epoch


def define_callbacks(param_train_dataset, x_train):
    call_backs = []

    model_checkpoint_callback = keras.callbacks.ModelCheckpoint(str(paraman["output_file_modelprinter"]), monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto',
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

    new_model.fit(param_train_dataset.image_data_generator.flow(x_train, y_train, batch_size=param_train_dataset.batch_size),
         epochs=(param_train_dataset.epochs if paraman["--nb-epoch"] is None else paraman["--nb-epoch"]) - init_nb_epoch,
         # epochs=2 - init_nb_epoch,
         verbose=2,
         validation_data=(x_val, y_val),
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
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = get_dataset()
    # Do compression or load #
    new_model, init_nb_epoch = get_or_load_new_model(model_compilation_params, x_test, y_test)
    # Callbacks definition #
    call_backs = define_callbacks(param_train_dataset, x_train)
    # Write results before finetuning #
    resprinter.print()
    # New model Finetuning #
    fit_new_model(new_model, param_train_dataset, init_nb_epoch, call_backs, x_train, y_train, x_test, y_test, x_val, y_val)



if __name__ == "__main__":
    logger.info("Command line: " + " ".join(sys.argv))
    log_memory_usage("Memory at startup")
    arguments = docopt.docopt(__doc__)
    paraman = ParameterManagerPalminizeFinetune(arguments)
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