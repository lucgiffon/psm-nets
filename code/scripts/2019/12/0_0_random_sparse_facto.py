"""
This script finds a palminized model with given arguments then finetune it.

Usage:
    script.py [-h] [-v|-vv] --walltime int  --sparsity-factor=int --nb-factor=intorstr [--no-permutation] [--tb] (--mnist|--svhn|--cifar10|--cifar100|--test-data) [--mnist-lenet|--test-model|--cifar10-vgg19|--cifar100-vgg19|--svhn-vgg19]

Options:
  -h --help                             Show this screen.
  -vv                                   Set verbosity to debug.
  -v                                    Set verbosity to info.
  --walltime int                        The number of hour before training is stopped.
  --tb                                  Tell if tensorboard should be printed.

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

Sparsity options:
  --sparsity-factor=int                 Integer coefficient from which is computed the number of value in each factor.
  --nb-factor=intorstr                  Integer telling how many factors should be used or list of int telling for each layer the number of factor ("int,int,int").
  --no-permutation                      Bool tells to not apply permutation to factors so that there is only block diag matrix
"""
import logging
import os
import pandas as pd
import sys

import keras
import signal
import docopt

from palmnet.core.palminize import Palminizable
from palmnet.data import Mnist, Test, Svhn, Cifar100, Cifar10
from palmnet.layers.sparse_tensor import RandomSparseFactorisationDense, RandomSparseFactorisationConv2D
from palmnet.models import sparse_random_vgg19_model, sparse_random_lenet_model
from palmnet.utils import timeout_signal_handler
from palmnet.experiments.utils import ResultPrinter, ParameterManagerRandomSparseFacto
from skluc.utils import logger, log_memory_usage

lst_results_header = [
    "test_accuracy_finetuned_model",
    "base_score",
    "finetuned_score",
    "nb_param",
    "nb_flop"
]

def main():

    (x_train, y_train), (x_test, y_test) = paraman.get_dataset()


    if paraman["--mnist-lenet"]:
        param_train_dataset = Mnist.get_model_param_training()
        base_model = sparse_random_lenet_model(x_train[0].shape, 10, sparsity_factor=paraman["--sparsity-factor"], nb_sparse_factors=paraman["--nb-factor"])
    elif paraman["--cifar10-vgg19"]:
        param_train_dataset = Cifar10.get_model_param_training()
        base_model = sparse_random_vgg19_model(x_train[0].shape, 10, permutation=not paraman["--no-permutation"], sparsity_factor=paraman["--sparsity-factor"], nb_sparse_factors=paraman["--nb-factor"])
    elif paraman["--cifar100-vgg19"]:
        param_train_dataset = Cifar100.get_model_param_training()
        base_model = sparse_random_vgg19_model(x_train[0].shape, 100, permutation=not paraman["--no-permutation"], sparsity_factor=paraman["--sparsity-factor"], nb_sparse_factors=paraman["--nb-factor"])
    elif paraman["--svhn-vgg19"]:
        param_train_dataset = Svhn.get_model_param_training()
        base_model = sparse_random_vgg19_model(x_train[0].shape, 10, permutation=not paraman["--no-permutation"], sparsity_factor=paraman["--sparsity-factor"], nb_sparse_factors=paraman["--nb-factor"])
    elif paraman["--test-model"]:
        param_train_dataset = Test.get_model_param_training()
        base_model = sparse_random_lenet_model(x_train[0].shape, 10, sparsity_factor=paraman["--sparsity-factor"], nb_sparse_factors=paraman["--nb-factor"])
    else:
        raise NotImplementedError("No dataset specified.")


    if os.path.exists(paraman["output_file_notfinishedprinter"]):
        df = pd.read_csv(paraman["output_file_resprinter"])
        try:
            init_nb_epoch = len(pd.read_csv(paraman["output_file_csvcbprinter"]))
        except Exception as e:
            logger.error("Caught exception while reading csv history: {}".format(str(e)))
            init_nb_epoch = 0
        base_score = float(df["base_score"])
        base_model = keras.models.load_model(paraman["output_file_modelprinter"],custom_objects={'RandomSparseFactorisationConv2D': RandomSparseFactorisationConv2D,
                                                                            "RandomSparseFactorisationDense": RandomSparseFactorisationDense})
        nb_param_model = int(df["nb_param"])
        nb_flop_model = int(df["nb_flop"])

    else:
        init_nb_epoch = 0

        base_model.compile(loss=param_train_dataset.loss,
                                 optimizer=param_train_dataset.optimizer,
                                 metrics=['categorical_accuracy'])
        base_score = base_model.evaluate(x_test, y_test, verbose=1)[1]
        print(base_score)
        nb_param_model, _, nb_flop_model, _, param_by_layer, flop_by_layer = Palminizable.count_model_param_and_flops_(base_model)
        print(nb_param_model, nb_flop_model)

        # results must be already printed once in case process is killed afterward
    dct_results = {
        "finetuned_score": None,
        "base_score": base_score,
        "nb_flop": nb_flop_model,
        "nb_param": nb_param_model,
    }
    resprinter.add(dct_results)
    resprinter.print()

    base_model.summary()

    call_backs = []

    model_checkpoint_callback = keras.callbacks.ModelCheckpoint(str(paraman["output_file_modelprinter"]), monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)
    call_backs.append(model_checkpoint_callback)
    if paraman["--tb"]:
        tbCallBack = keras.callbacks.TensorBoard(log_dir=str(paraman["output_file_tensorboardprinter"]), histogram_freq=20, write_graph=False, write_images=False, batch_size=param_train_dataset.batch_size, write_grads=True, update_freq="epoch")
        call_backs.append(tbCallBack)
    csvcallback = keras.callbacks.callbacks.CSVLogger(str(paraman["output_file_csvcbprinter"]), separator=',', append=True)
    call_backs.append(csvcallback)


    signal.signal(signal.SIGALRM, timeout_signal_handler)
    signal.alarm(int(paraman["--walltime"] * 3600))  # start alarm
    finetuned_score = None
    try:
        open(paraman["output_file_notfinishedprinter"], 'w').close()

        history = base_model.fit(param_train_dataset.image_data_generator.flow(x_train, y_train, batch_size=param_train_dataset.batch_size),
                                       epochs=param_train_dataset.epochs - init_nb_epoch,
                                       # epochs=2 - init_nb_epoch,
                                       verbose=1,
                                       validation_data=(x_test, y_test),
                                       callbacks=param_train_dataset.callbacks + call_backs)
        signal.alarm(0)  # stop alarm for next evaluation
        finetuned_score = base_model.evaluate(x_test, y_test, verbose=1)[1]
        print(finetuned_score)

        if os.path.exists(paraman["output_file_notfinishedprinter"]):
            os.remove(paraman["output_file_notfinishedprinter"])
    # except TimeoutError as te:
    except Exception as e:
        logging.error("Caught exception: {}".format(e))
        finetuned_score = None
    finally:
        dct_results = {
            "finetuned_score": finetuned_score,
            "base_score": base_score,
            "nb_flop": nb_flop_model,
            "nb_param": nb_param_model,
        }
        base_model.save(str(paraman["output_file_modelprinter"]))
        resprinter.add(dct_results)


if __name__ == "__main__":
    logger.info("Command line: " + " ".join(sys.argv))
    log_memory_usage("Memory at startup")
    arguments = docopt.docopt(__doc__)
    paraman = ParameterManagerRandomSparseFacto(arguments)
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