"""
This script finds a palminized model with given arguments then finetune it.

Usage:
    script.py --input-dir path [-h] [-v|-vv] [--keep-last-layer] [--lr float] [--use-clr [--min-lr float --max-lr float] [--epoch-step-size int]] [--nb-epoch int] [--only-mask] [--tb] (--mnist|--svhn|--cifar10|--cifar100|--test-data) [--cifar100-resnet50|--cifar100-resnet20|--mnist-500|--mnist-lenet|--test-model|--cifar10-vgg19|--cifar100-vgg19|--svhn-vgg19] --sparsity-factor=int [--nb-iteration-palm=int] [--delta-threshold=float] [--hierarchical] [--nb-factor=int]

Options:
  -h --help                             Show this screen.
  -vv                                   Set verbosity to debug.
  -v                                    Set verbosity to info.
  --input-dir path                      Path to input directory where to find previously generated results.
  --tb                                  Tell if tensorboard should be printed.
  --lr float                            Flat lr to be used (Overidable)
  --min-lr float                        Tells the min reasonable lr (Overide everything else).
  --max-lr float                        Tells the max reasonable lr (Overide everything else).
  --nb-epoch int                        Number of epochs of training (Overide everything else).
  --epoch-step-size int                 Number of epochs for an half cycle of CLR.
  --use-clr                             Tell to use clr.
  --keep-last-layer                     Do not compress classification layer.

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
"""
import logging
import os
import pickle
import pandas as pd
import sys
from collections import defaultdict

import time
from copy import deepcopy
import keras
from keras.engine import Model, InputLayer
import signal
import docopt
from scipy.sparse import coo_matrix
from palmnet.utils import CyclicLR

from palmnet.core.palminizer import Palminizer
from palmnet.core.palminizable import Palminizable
from palmnet.data import Mnist, Test, Svhn, Cifar100, Cifar10
# from palmnet.layers.sparse_tensor import SparseFactorisationDense#, SparseFactorisationConv2DDensify
from palmnet.layers.sparse_facto_conv2D_masked import SparseFactorisationConv2D
from palmnet.layers.sparse_facto_dense_masked import SparseFactorisationDense
from palmnet.utils import get_sparsity_pattern, insert_layer_nonseq, timeout_signal_handler, get_lr_metric, CSVLoggerByBatch
from palmnet.experiments.utils import ParameterManagerPalminize, ParameterManagerPalminizeFinetune, ResultPrinter
from skluc.utils import logger, log_memory_usage
from keras.layers import Dense, Conv2D
import numpy as np
import keras.backend as K
from palmnet.core import palminizable
from palmnet.core.palminizer import Palminizer
palminizable.Palminizer = Palminizer
import sys
sys.modules["palmnet.core.palminize"] = palminizable
lst_results_header = [
    "test_accuracy_finetuned_model"
]

def get_idx_last_dense_layer(model):
    idx_last_dense_layer = -1
    for i, layer in enumerate(model.layers):
        if isinstance(layer, Dense):
            idx_last_dense_layer = i
    if idx_last_dense_layer == -1:
        logger.warning("No dense layer found")
    return idx_last_dense_layer

def replace_layers_with_sparse_facto(model, dct_name_facto):
    new_model = deepcopy(model)
    log_memory_usage("After copy model")
    lst_tpl_str_bool_new_model_layers = []
    dct_new_layer_attr = defaultdict(lambda: {})

    idx_last_dense_layer = get_idx_last_dense_layer(new_model) if paraman["--keep-last-layer"] else -1

    for i, layer in enumerate(new_model.layers):
        layer_name = layer.name
        sparse_factorization = dct_name_facto[layer_name]
        logger.info('Prepare layer {}'.format(layer.name))
        # if sparse_factorization != (None, None) and (i != idx_last_dense_layer and paraman["--keep-last-layer"]):
        if sparse_factorization != (None, None) and not (i == idx_last_dense_layer and paraman["--keep-last-layer"]):
            # scaling = 1.
            if paraman["--only-mask"]:
                scaling = []
            else:
                scaling = [np.array(sparse_factorization[0])[None]]
            # factors_sparse = [coo_matrix(fac.toarray()) for fac in sparse_factorization[1].get_list_of_factors()]
            factors = [fac.toarray() for fac in sparse_factorization[1].get_list_of_factors()]
            # sparsity_patterns = [get_sparsity_pattern(w.toarray()) for w in factors]
            sparsity_patterns = [get_sparsity_pattern(w) for w in factors]
            nb_val_sparse_factors = np.sum([np.sum(fac) for fac in sparsity_patterns])
            # factor_data_sparse = [f.data for f in factors_sparse]
            factor_data = factors
            reconstructed_matrix = np.linalg.multi_dot(factors) * scaling[0]
            nb_val_full_matrix = np.prod(reconstructed_matrix.shape)

            if nb_val_full_matrix <= nb_val_sparse_factors:
                logger.info("Less values in full matrix than factorization. Keep full matrix. {} <= {}".format(nb_val_full_matrix, nb_val_sparse_factors))
                dct_new_layer_attr[layer_name]["modified"] = False
                lst_tpl_str_bool_new_model_layers.append((layer_name, False))
                dct_new_layer_attr[layer_name]["layer_obj"] = layer
                continue

            base_palminized_matrix = np.reshape(layer.get_weights()[0], reconstructed_matrix.shape)
            diff = np.linalg.norm(base_palminized_matrix - reconstructed_matrix) / np.linalg.norm(base_palminized_matrix)
            # assert np.allclose(diff, 0, atol=1e-5), "Reconstructed  is different than base"

            # create new layer
            if isinstance(layer, Dense):
                logger.debug("Dense layer treatment")
                hidden_layer_dim = layer.units
                activation = layer.activation
                regularizer = layer.kernel_regularizer
                replacing_layer = SparseFactorisationDense(use_scaling=not paraman["--only-mask"], units=hidden_layer_dim, sparsity_patterns=sparsity_patterns, use_bias=layer.use_bias, activation=activation, kernel_regularizer=regularizer)
                replacing_weights = scaling + factor_data + [layer.get_weights()[-1]] if layer.use_bias else []
                # new_model = insert_layer_nonseq(new_model, layer_name, lambda: replacing_layer, position="replace")
                # replacing_layer.set_weights(replacing_weights)

            elif isinstance(layer, Conv2D):
                logger.debug("Conv2D layer treatment")
                nb_filters = layer.filters
                strides = layer.strides
                kernel_size = layer.kernel_size
                activation = layer.activation
                padding = layer.padding
                regularizer = layer.kernel_regularizer
                replacing_layer = SparseFactorisationConv2D(use_scaling=not paraman["--only-mask"], strides=strides, filters=nb_filters, kernel_size=kernel_size, sparsity_patterns=sparsity_patterns, use_bias=layer.use_bias, activation=activation, padding=padding, kernel_regularizer=regularizer)
                replacing_weights = scaling + factor_data + [layer.get_weights()[-1]] if layer.use_bias else []
                # new_model = insert_layer_nonseq(new_model, layer_name, lambda: replacing_layer, position="replace")
                # replacing_layer.set_weights(replacing_weights)

            else:
                raise ValueError("unknown layer class")

            dct_new_layer_attr[layer_name]["layer_weights"] = replacing_weights
            dct_new_layer_attr[layer_name]["sparsity_pattern"] = sparsity_patterns
            dct_new_layer_attr[layer_name]["layer_obj"] = replacing_layer
            dct_new_layer_attr[layer_name]["modified"] = True

            lst_tpl_str_bool_new_model_layers.append((layer_name, True))
        else:
            dct_new_layer_attr[layer_name]["modified"] = False
            lst_tpl_str_bool_new_model_layers.append((layer_name, False))
            dct_new_layer_attr[layer_name]["layer_obj"] = layer

    log_memory_usage("After prepare all sparse layers ")

    network_dict = {'input_layers_of': defaultdict(lambda: []), 'new_output_tensor_of': defaultdict(lambda: [])}

    if not isinstance(new_model.layers[0], InputLayer):
        new_model = Model(input=new_model.input, output=new_model.output)

    # Set the input layers of each layer
    for layer in new_model.layers:
        # each layer is set as `input` layer of all its outbound layers
        for node in layer._outbound_nodes:
            outbound_layer_name = node.outbound_layer.name
            # if outbound_layer_name not in network_dict
            # network_dict['input_layers_of'].update({outbound_layer_name: [layer.name]})
            network_dict['input_layers_of'][outbound_layer_name].append(layer.name)

    # Set the output tensor of the input layer
    network_dict['new_output_tensor_of'].update(
        {new_model.layers[0].name: new_model.input})

    for layer in new_model.layers[1:]:
        log_memory_usage("Before layer {}".format(layer.name))
        layer_name = layer.name

        layer_input = [network_dict['new_output_tensor_of'][layer_aux] for layer_aux in network_dict['input_layers_of'][layer.name]]

        if len(layer_input) == 1:
            layer_input = layer_input[0]

        proxy_new_layer_attr = dct_new_layer_attr[layer_name]

        if proxy_new_layer_attr["modified"]:
            x = layer_input

            new_layer = proxy_new_layer_attr["layer_obj"] # type: keras.layers.Layer
            new_layer.name = '{}_{}'.format(layer.name,
                                            new_layer.name)
            x = new_layer(x)

            if not paraman["--only-mask"]:
                if layer.use_bias:
                    reconstructed_matrix = np.linalg.multi_dot(proxy_new_layer_attr["layer_weights"][1:-1]) * proxy_new_layer_attr["layer_weights"][0]
                else:
                    reconstructed_matrix = np.linalg.multi_dot(proxy_new_layer_attr["layer_weights"][1:]) * proxy_new_layer_attr["layer_weights"][0]

                base_palminized_matrix = np.reshape(layer.get_weights()[0], reconstructed_matrix.shape)
                diff = np.linalg.norm(base_palminized_matrix - reconstructed_matrix) / np.linalg.norm(base_palminized_matrix)
                # assert np.allclose(diff, 0, atol=1e-5), "Reconstructed  is different than base"
                del base_palminized_matrix

                new_layer.set_weights(proxy_new_layer_attr["layer_weights"])

            else:
                masked_weights = []
                i = 0
                for w in new_layer.get_weights():
                    if len(w.shape) > 1:
                        new_weight = w * proxy_new_layer_attr["sparsity_pattern"][i]
                        i += 1
                    else:
                        new_weight = w
                    masked_weights.append(new_weight)
                new_layer.set_weights(masked_weights)

            logger.info('Layer {} modified into {}'.format(layer.name, new_layer.name))
        else:
            x = layer(layer_input)
            logger.info('Layer {} unmodified'.format(layer.name))

        network_dict['new_output_tensor_of'].update({layer.name: x})

        del dct_new_layer_attr[layer_name]

    new_model = Model(inputs=new_model.inputs, outputs=x)

    return new_model

def main():

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

    (x_train, y_train), (x_test, y_test) = paraman.get_dataset().load_data()

    if paraman["--mnist-500"]:
        x_test = np.reshape(x_test, (-1, 784))
        x_train = np.reshape(x_train, (-1, 784))

    # noinspection PyUnreachableCode
    if os.path.exists(paraman["output_file_notfinishedprinter"]):
        df = pd.read_csv(paraman["output_file_resprinter"])
        init_nb_epoch = pd.read_csv(paraman["output_file_csvcbprinter"])["epoch"].max() -1
        base_score = float(df["base_score"])
        before_finetuned_score = float(df["before_finetuned_score"])
        palminized_score = float(df["palminized_score"])
        fine_tuned_model = keras.models.load_model(paraman["output_file_modelprinter"],custom_objects={'SparseFactorisationConv2D':SparseFactorisationConv2D,
                                                                            "SparseFactorisationDense": SparseFactorisationDense})
    else:
        init_nb_epoch = 0

        mypalminizedmodel = pickle.load(open(paraman["input_model_path"], "rb"))  # type: Palminizable
        log_memory_usage("After load mypalminized model")
        base_model = mypalminizedmodel.base_model
        dct_name_facto = mypalminizedmodel.sparsely_factorized_layers
        base_score = base_model.evaluate(x_test, y_test, verbose=0)[1]
        print(base_score)
        palminized_model = mypalminizedmodel.compressed_model
        palminized_score = palminized_model.evaluate(x_test, y_test, verbose=1)[1]
        print(palminized_score)
        fine_tuned_model = replace_layers_with_sparse_facto(palminized_model, dct_name_facto)
        log_memory_usage("After get_finetuned_model")
        # fine_tuned_model = palminized_model

        input_by_shape = {(32,32,3): x_test[:3]}

        # for i, layer in enumerate(palminized_model.layers[1:]):
        #     i = i+1
        #     print("Start with layer {}".format(layer.name))
        #     dense_palm_layer = layer
        #     sparsefacto_palm_layer = fine_tuned_model.layers[i]
        #
        #     dense_layer_output_function = K.function([dense_palm_layer.input],
        #                                              [dense_palm_layer.output])
        #
        #     sparsefacto_layer_outut_function = K.function([sparsefacto_palm_layer.get_input_at(-1)],
        #                                              [sparsefacto_palm_layer.get_output_at(-1)])
        #
        #     necessary_input_shapes = [tuple(inpt.shape.as_list()[1:]) for inpt in dense_layer_output_function.inputs]
        #     input_data_layer = [input_by_shape[shap] for shap in necessary_input_shapes]
        #
        #     dense_layer_output = dense_layer_output_function(input_data_layer)[0]
        #     sparsefacto_layer_output = sparsefacto_layer_outut_function(input_data_layer)[0]
        #
        #     # try:
        #     assert np.allclose(np.linalg.norm(dense_layer_output - sparsefacto_layer_output) / np.linalg.norm(dense_layer_output), 0, atol=1e-5)
        #     # except:
        #     #     print("error")
        #     input_by_shape[dense_layer_output.shape[1:]] = dense_layer_output

        params_optimizer = param_train_dataset.params_optimizer

        params_optimizer["lr"] = paraman["--lr"] if paraman["--lr"] is not None else params_optimizer["lr"]

        fine_tuned_model.compile(loss=param_train_dataset.loss,
                                 optimizer=param_train_dataset.optimizer(**params_optimizer),
                                 metrics=['categorical_accuracy'])
                                 # metrics=['categorical_accuracy', get_lr_metric(param_train_dataset.optimizer)])

        before_finetuned_score = fine_tuned_model.evaluate(x_test, y_test, verbose=1)[1]
        print(before_finetuned_score)

    # results must be already printed once in case process is killed afterward
    dct_results = {
        "finetuned_score": None,
        "before_finetuned_score": before_finetuned_score,
        "base_score": base_score,
        "palminized_score": palminized_score,
    }
    resprinter.add(dct_results)
    resprinter.print()

    # if paraman["--hierarchical"]:
    # if not paraman["--only-mask"]:
    #     assert before_finetuned_score == palminized_score, \
    #     "the reconstructed model with sparse facto should equal in perf to the reconstructed model with dense product. {} != {}".format(before_finetuned_score, palminized_score)
    # else: # small fix for a bug where when I wasn't using hierarchical palm returned a matrix that wasn't multiplied by lambda
    #     # this should pass until results are generated without bug..
    #     assert before_finetuned_score != palminized_score, \
    #         "the reconstructed model with sparse facto should equal in perf to the reconstructed model with dense product. {} != {}".format(before_finetuned_score, palminized_score)
    fine_tuned_model.summary()

    call_backs = []

    model_checkpoint_callback = keras.callbacks.ModelCheckpoint(str(paraman["output_file_modelprinter"]), monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)
    call_backs.append(model_checkpoint_callback)
    if paraman["--tb"]:
        tbCallBack = keras.callbacks.TensorBoard(log_dir=str(paraman["output_file_tensorboardprinter"]), histogram_freq=20, write_graph=False, write_images=False, batch_size=param_train_dataset.batch_size, write_grads=True, update_freq="epoch")
        call_backs.append(tbCallBack)

    if paraman["--use-clr"]:
        clr_cb = CyclicLR(base_lr=param_train_dataset.min_lr if paraman["--min-lr"] is None else paraman["--min-lr"],
                          max_lr=param_train_dataset.max_lr if paraman["--max-lr"] is None else paraman["--max-lr"],
                          step_size=(paraman["--epoch-step-size"]*(x_train.shape[0] // param_train_dataset.batch_size)),
                          logrange=True)
        call_backs.append(clr_cb)

    csvcallback = CSVLoggerByBatch(str(paraman["output_file_csvcbprinter"]), n_batch_between_display=100, separator=',', append=True)
    call_backs.append(csvcallback)

    finetuned_score = None

    open(paraman["output_file_notfinishedprinter"], 'w').close()

    history = fine_tuned_model.fit(param_train_dataset.image_data_generator.flow(x_train, y_train, batch_size=param_train_dataset.batch_size),
                                   epochs=(param_train_dataset.epochs if paraman["--nb-epoch"] is None else paraman["--nb-epoch"]) - init_nb_epoch,
                                   # epochs=2 - init_nb_epoch,
                                   verbose=2,
                                   # validation_data=(x_test, y_test),
                                   callbacks=param_train_dataset.callbacks + call_backs)

    finetuned_score = fine_tuned_model.evaluate(x_test, y_test, verbose=1)[1]
    print(finetuned_score)

    if os.path.exists(paraman["output_file_notfinishedprinter"]):
        os.remove(paraman["output_file_notfinishedprinter"])


    dct_results = {
        "finetuned_score": finetuned_score,
        "before_finetuned_score": before_finetuned_score,
        "base_score": base_score,
        "palminized_score": palminized_score,
    }
    fine_tuned_model.save(str(paraman["output_file_modelprinter"]))
    resprinter.add(dct_results)


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