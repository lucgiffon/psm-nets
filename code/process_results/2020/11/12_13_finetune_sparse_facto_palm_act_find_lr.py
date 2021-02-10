from collections import defaultdict

import keras.backend as K
import pickle
import pathlib
import pandas as pd
import scipy.special
import scipy.stats
from keras.models import Model
import gc
import palmnet.hunt
from palmnet.core.faustizer import Faustizer
from palmnet.core.layer_replacer_faust import LayerReplacerFaust
from palmnet.core.layer_replacer_palm import LayerReplacerPalm
from palmnet.data import param_training, image_data_generator_cifar_svhn, image_data_generator_mnist
from palmnet.experiments.utils import get_line_of_interest, ParameterManager
from palmnet.utils import get_sparsity_pattern, get_nb_learnable_weights, get_nb_learnable_weights_from_model
from palmnet.visualization.utils import get_palminized_model_and_df, get_df
import numpy as np
import logging
from palmnet.core import palminizable
from palmnet.core.palminizer import Palminizer
palminizable.Palminizer = Palminizer
import sys
sys.modules["palmnet.core.palminize"] = palminizable

from skluc.utils import logger, log_memory_usage
import keras

mpl_logger = logging.getLogger('matplotlib')
mpl_logger.setLevel(logging.ERROR)

logger.setLevel(logging.DEBUG)


def get_singular_values_info(matrix):
    U, S, V = np.linalg.svd(matrix)
    mean_sv = np.mean(S)
    softmax_S = scipy.special.softmax(S)
    entropy_S = scipy.stats.entropy(softmax_S)
    entropy_sv = entropy_S
    nb_sv = len(S)
    entropy_sv_normalized = entropy_S / scipy.stats.entropy(scipy.special.softmax(np.ones(len(S))))
    percent_sv_above_mean = np.sum(S > mean_sv) / len(S)
    return entropy_sv, nb_sv, entropy_sv_normalized, percent_sv_above_mean

def get_df_from_expe_path(expe_path):
    src_dir = root_source_dir / expe_path
    df = get_df(src_dir)
    df = df.assign(results_dir=[str(src_dir.absolute())] * len(df))
    df = df.rename(columns={"--tol": "--delta-threshold"})

    if expe_path == "2020/07/11_12_finetune_sparse_facto_fix_replicates":
        df = df[df["--only-mask"] == False]
    return df


columns_not_to_num = ['hash', 'output_file_csvcbprinter', "--use-clr",
                      "--input-dir", "input_model_path", "output_file_csvcvprinter",
                      "output_file_finishedprinter", "output_file_layerbylayer",
                      "output_file_modelprinter", "output_file_notfinishedprinter",
                      "output_file_resprinter", "output_file_tensorboardprinter", "results_dir"]


def cast_to_num(df):
    for col in df.columns.difference(columns_not_to_num):
        if col in df.columns.values:
            df.loc[:, col] = df.loc[:, col].apply(pd.to_numeric, errors='coerce')
    return df

if __name__ == "__main__":
    root_source_dir = pathlib.Path("/home/luc/PycharmProjects/palmnet/results/")
    expe_path = "2020/11/12_13_finetune_sparse_facto_palm_act_find_lr"

    lst_path_finetune = [
        "2020/11/12_13_finetune_sparse_facto_palm_act_find_lr"
    ]

    # lst_path_compression = [
    #     "2020/05/3_4_compression_palm_not_log_all",
    # ]

    df_finetune = pd.concat(list(map(get_df_from_expe_path, lst_path_finetune)))
    # df_finetune = get_df_from_expe_path(lst_path_finetune[0])
    df_finetune = df_finetune.dropna(subset=["failure"])
    df_finetune = df_finetune[df_finetune["failure"] == False]
    df_finetune = df_finetune.drop(columns="oar_id").drop_duplicates()
    df_finetune = cast_to_num(df_finetune)
    df_finetune = df_finetune[~df_finetune["test_accuracy_finetuned_model"].isnull()]

    # df_compression = pd.concat(list(map(get_df_from_expe_path, lst_path_compression)))
    # df_compression = get_df_from_expe_path(lst_path_compression[0])
    # df_compression = cast_to_num(df_compression)

    root_output_dir = pathlib.Path("/home/luc/PycharmProjects/palmnet/results/processed/")
    output_dir = root_output_dir / expe_path
    output_dir.mkdir(parents=True, exist_ok=True)

    dct_attributes = defaultdict(lambda: [])
    dct_results_matrices = defaultdict(lambda: [])

    length_df = len(df_finetune)

    for idx, (_, row) in enumerate(df_finetune.iterrows()):
        # if df_results_tmp is not None and row["hash"] in df_results_tmp["hash"].values:
        #     continue
        bool_resnet_new = row["--cifar100-resnet20-new"] or row["--cifar100-resnet50-new"]
        if row["--cifar100-resnet50-new"] and np.isnan(row["test_loss_finetuned_model"]) and row["--only-mask"] :
            print("FOUND NAN 50 :(")
            continue
            pass
        if row["--cifar100-resnet20-new"] and np.isnan(row["test_loss_finetuned_model"]) and row["--only-mask"] :
            print("FOUND NAN 20 :(")
            continue
            pass

        # if np.isnan(row["test_loss_finetuned_model"]) :
        #     print("FOUND NAN 2 :(")
        #     continue

        log_memory_usage("Start loop")
        print("row {}/{}".format(idx, length_df))
        dct_attributes["idx-expe"].append(idx)
        dct_attributes["hash"].append(row["hash"])

        # get corresponding row in the palminize results directory #
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
                            "--nb-factor"
                            ]

        if row["--cifar100-resnet50"] or row["--cifar100-resnet20"]:
            keys_of_interest.extend([
                '--cifar100-resnet50',
                '--cifar100-resnet20',
            ])

        # row_before_finetune = get_line_of_interest(df_compression, keys_of_interest, row).iloc[0]
        # this is the row of results for the model before finetuning

        ############################################
        # Global informations about the experiment #
        ############################################

        if row["--cifar10"]:
            dct_attributes["dataset"].append("cifar10")
        elif row["--cifar100"]:
            dct_attributes["dataset"].append("cifar100")
        elif row["--mnist"]:
            dct_attributes["dataset"].append("mnist")
        elif row["--svhn"]:
            dct_attributes["dataset"].append("svhn")
        else:
            raise ValueError("Unknown dataset")

        if row["--cifar100-vgg19"] or row["--cifar10-vgg19"] or row["--svhn-vgg19"]:
            dct_attributes["model"].append("vgg19")
        elif row["--mnist-lenet"]:
            dct_attributes["model"].append("lenet")
        elif row["--mnist-500"]:
            dct_attributes["model"].append("fc500")
        elif row["--cifar100-resnet20"]:
            dct_attributes["model"].append("resnet20")
        elif row["--cifar100-resnet50"]:
            dct_attributes["model"].append("resnet50")
        elif row["--cifar100-resnet20-new"]:
            dct_attributes["model"].append("resnet20")
        elif row["--cifar100-resnet50-new"]:
            dct_attributes["model"].append("resnet50")
        else:
            raise ValueError("Unknown model")

        if row["faust"]:
            dct_attributes["method"].append("faust")
        elif row["palm"]:
            dct_attributes["method"].append("pyqalm")
        else:
            raise NotImplementedError

        # palm informations #
        dct_attributes["delta-threshold"].append(float(row["--delta-threshold"]))
        dct_attributes["hierarchical"].append(bool(row["--hierarchical"]))
        dct_attributes["nb-factor"].append(int(row["--nb-factor"]) if not np.isnan(row["--nb-factor"]) else np.nan)
        dct_attributes["nb-iteration-palm"].append(int(row["--nb-iteration-palm"]))
        dct_attributes["sparsity-factor"].append(int(row["--sparsity-factor"]))

        # finetuning informations
        dct_attributes["use-clr"].append(row["--use-clr"])  # this must be first because used in other attributes
        dct_attributes["only-mask"].append(bool(row["--only-mask"]))
        dct_attributes["keep-last-layer"].append(bool(row["--keep-last-layer"]))
        dct_attributes["keep-first-layer"].append(bool(row["--keep-first-layer"]))
        dct_attributes["only-dense"].append(bool(row["--only-dense"]))
        # beware of this line here because the params_optimizer may change between experiments
        dct_attributes["epoch-step-size"].append(float(row["--epoch-step-size"]) if dct_attributes["use-clr"][-1] else np.nan)
        dct_attributes["actual-batch-size"].append(int(row["actual-batch-size"]) if row["actual-batch-size"] is not None else None)
        dct_attributes["actual-nb-epochs"].append(int(row["actual-nb-epochs"]) if row["actual-nb-epochs"] is not None else None)
        dct_attributes["actual-min-lr"].append(float(row["actual-min-lr"]) if row["actual-min-lr"] is not None else None)
        dct_attributes["actual-max-lr"].append(float(row["actual-max-lr"]) if row["actual-max-lr"] is not None else None)
        dct_attributes["actual-lr"].append(float(row["actual-lr"]) if row["actual-lr"] is not None else None)

        # score informations
        dct_attributes["base-model-score"].append(float(row["test_accuracy_base_model"]))
        dct_attributes["before-finetune-score"].append(float(row["test_accuracy_compressed_model"]))
        dct_attributes["finetuned-score"].append(float(row["test_accuracy_finetuned_model"]))
        dct_attributes["base-model-loss"].append(float(row["test_loss_base_model"]))
        dct_attributes["before-finetune-loss"].append(float(row["test_loss_compressed_model"]))
        dct_attributes["finetuned-loss"].append(float(row["test_loss_finetuned_model"]))
        dct_attributes["finetuned-score-val"].append(float(row["val_accuracy_finetuned_model"]))

        # palm act info
        dct_attributes["batch-size"].append(int(row["--batch-size"]) if row["--batch-size"] is not None else None)
        dct_attributes["max-cum-batch-size"].append(int(row["--max-cum-batch-size"]) if row["--max-cum-batch-size"] is not None else None)
        dct_attributes["nb-epochs"].append(int(row["--nb-epochs"]) if row["--nb-epochs"] is not None else None)
        dct_attributes["activations"].append(bool(row["--activations"]) if row["--activations"] is not None else False)
        dct_attributes["full-model-approx"].append(bool(row["--full-model-approx"]) if row["--full-model-approx"] is not None else False)
        dct_attributes["only-one-batch"].append(bool(row["--only-one-batch"]) if row["--only-one-batch"] is not None else False)

        # store path informations
        # path_model_compressed = pathlib.Path(row_before_finetune["results_dir"]) / row_before_finetune["output_file_modelprinter"]
        path_history = pathlib.Path(row["results_dir"]) / row["output_file_csvcbprinter"]
        dct_attributes["path-learning-history"].append(path_history)
        # dct_attributes["path-model-compressed"].append(path_model_compressed)
        ##############################
        # Layer by Layer information #
        ##############################

        nb_param_dense_base = 0
        nb_param_dense_compressed = 0
        nb_param_conv_base = 0
        nb_param_conv_compressed = 0

        if type(row["output_file_layerbylayer"]) == str:
            dct_attributes["nb-param-base-total"].append(int(row["base_model_nb_param"]))
            dct_attributes["nb-param-compressed-total"].append(int(row["new_model_nb_param"]))
            dct_attributes["param-compression-rate-total"].append(row["base_model_nb_param"]/row["new_model_nb_param"])

            path_layer_by_layer = pathlib.Path(row["results_dir"]) / row["output_file_layerbylayer"]
            df_csv_layerbylayer = pd.read_csv(str(path_layer_by_layer))

            for idx_row_layer, row_layer in df_csv_layerbylayer.iterrows():
                dct_results_matrices["idx-expe"].append(idx)
                dct_results_matrices["model"].append(dct_attributes["model"][-1])
                layer_name_compressed = row_layer["layer-name-compressed"]
                is_dense = "sparse_factorisation_dense" in layer_name_compressed
                dct_results_matrices["layer-name-base"].append(row_layer["layer-name-base"])
                dct_results_matrices["layer-name-compressed"].append(row_layer["layer-name-compressed"])
                dct_results_matrices["idx-layer"].append(row_layer["idx-layer"])
                dct_results_matrices["data"].append(dct_attributes["dataset"][-1])
                dct_results_matrices["keep-last-layer"].append(dct_attributes["keep-last-layer"][-1])
                dct_results_matrices["use-clr"].append(dct_attributes["use-clr"][-1])

                dct_results_matrices["diff-approx"].append(row_layer["diff-approx"])

                # get nb val base layer and comrpessed layer
                dct_results_matrices["nb-non-zero-base"].append(row_layer["nb-non-zero-base"])
                dct_results_matrices["nb-non-zero-compressed"].append(row_layer["nb-non-zero-compressed"])
                dct_results_matrices["nb-non-zero-compression-rate"].append(row_layer["nb-non-zero-compression-rate"])
                if is_dense:
                    nb_param_dense_base += row_layer["nb-non-zero-base"]
                    nb_param_dense_compressed += row_layer["nb-non-zero-compressed"]
                else:
                    nb_param_conv_base += row_layer["nb-non-zero-base"]
                    nb_param_conv_compressed += row_layer["nb-non-zero-compressed"]

                # get palm setting options
                dct_results_matrices["nb-factor-param"].append(dct_attributes["nb-factor"][-1])
                # dct_results_matrices["nb-factor-actual"].append(len(sparsity_patterns))
                dct_results_matrices["sparsity-factor"].append(dct_attributes["sparsity-factor"][-1])
                dct_results_matrices["hierarchical"].append(dct_attributes["hierarchical"][-1])
        else:
            exit("Need compression path")
            palmnet.hunt.show_most_common_types(limit=20)
            log_memory_usage("Before pickle")
            layer_replacer = LayerReplacerFaust(only_mask=False, keep_last_layer=dct_attributes["keep-last-layer"][-1], path_checkpoint_file=path_model_compressed, sparse_factorizer=Faustizer())
            layer_replacer.load_dct_name_compression()
            log_memory_usage("After pickle")
            paraman = ParameterManager(row.to_dict())
            base_model = paraman.get_model()
            palmnet.hunt.show_most_common_types(limit=20)
            compressed_model = layer_replacer.transform(base_model)

            palmnet.hunt.show_most_common_types(limit=20)
            log_memory_usage("After transform")

            if len(base_model.layers) < len(compressed_model.layers):
                base_model = Model(inputs=base_model.inputs, outputs=base_model.outputs)

            assert len(base_model.layers) == len(compressed_model.layers)

            # model complexity informations obtained from the reconstructed model
            nb_learnable_weights_base_model = get_nb_learnable_weights_from_model(base_model)
            nb_learnable_weights_compressed_model = get_nb_learnable_weights_from_model(compressed_model)
            dct_attributes["nb-param-base-total"].append(int(nb_learnable_weights_base_model))
            dct_attributes["nb-param-compressed-total"].append(int(nb_learnable_weights_compressed_model))
            dct_attributes["param-compression-rate-total"].append(nb_learnable_weights_base_model/nb_learnable_weights_compressed_model)

            dct_name_facto = None
            dct_name_facto = layer_replacer.dct_name_compression

            for idx_layer, base_layer in enumerate(base_model.layers):
                log_memory_usage("Start secondary loop")
                sparse_factorization = dct_name_facto.get(base_layer.name, (None, None))
                if sparse_factorization != (None, None) and sparse_factorization != None:
                    print(base_layer.name)
                    compressed_layer = None
                    compressed_layer = compressed_model.layers[idx_layer]

                    # get informations to identify the layer (and do cross references)
                    dct_results_matrices["idx-expe"].append(idx)
                    dct_results_matrices["model"].append(dct_attributes["model"][-1])
                    dct_results_matrices["layer-name-base"].append(base_layer.name)
                    layer_name_compressed = compressed_layer.name
                    is_dense = "sparse_factorisation_dense" in layer_name_compressed
                    dct_results_matrices["layer-name-compressed"].append(compressed_layer.name)
                    dct_results_matrices["idx-layer"].append(idx_layer)
                    dct_results_matrices["data"].append(dct_attributes["dataset"][-1])
                    dct_results_matrices["keep-last-layer"].append(dct_attributes["keep-last-layer"][-1])
                    dct_results_matrices["use-clr"].append(dct_attributes["use-clr"][-1])

                    # get sparse factorization
                    scaling = sparse_factorization['lambda']
                    factors = Faustizer.get_factors_from_op_sparsefacto(sparse_factorization['sparse_factors'])
                    sparsity_patterns = [get_sparsity_pattern(w) for w in factors]
                    factor_data = factors

                    # rebuild full matrix to allow comparisons
                    reconstructed_matrix = np.linalg.multi_dot(factors) * scaling
                    base_matrix = np.reshape(base_layer.get_weights()[0], reconstructed_matrix.shape)

                    # normalized approximation errors
                    diff = np.linalg.norm(base_matrix - reconstructed_matrix) / np.linalg.norm(base_matrix)
                    dct_results_matrices["diff-approx"].append(diff)

                    # # measures "singular values" #
                    # # base matrix
                    # base_entropy_sv, base_nb_sv, base_entropy_sv_normalized, base_percent_sv_above_mean = get_singular_values_info(base_matrix)
                    # dct_results_matrices["entropy-base-sv"].append(base_entropy_sv)
                    # dct_results_matrices["nb-sv-base"].append(base_nb_sv)
                    # dct_results_matrices["entropy-base-sv-normalized"].append(base_entropy_sv_normalized)
                    # dct_results_matrices["percent-sv-base-above-mean"].append(base_percent_sv_above_mean)
                    # # reconstructed matrix
                    # recons_entropy_sv, recons_nb_sv, recons_entropy_sv_normalized, recons_percent_sv_above_mean = get_singular_values_info(reconstructed_matrix)
                    # dct_results_matrices["entropy-recons-sv"].append(recons_entropy_sv)
                    # dct_results_matrices["nb-sv-recons"].append(recons_nb_sv)
                    # dct_results_matrices["entropy-recons-sv-normalized"].append(recons_entropy_sv_normalized)
                    # dct_results_matrices["percent-sv-recons-above-mean"].append(recons_percent_sv_above_mean)

                    # complexity analysis #
                    # get nb val of the full reconstructed matrix
                    sparsity_pattern_reconstructed = get_sparsity_pattern(reconstructed_matrix)
                    nb_non_zero = int(np.sum(sparsity_pattern_reconstructed))
                    size_bias = len(base_layer.get_weights()[-1]) if base_layer.use_bias else 0
                    # dct_results_matrices["nb-non-zero-reconstructed"].append(nb_non_zero + size_bias)
                    # get nb val base layer and comrpessed layers
                    nb_weights_base_layer = get_nb_learnable_weights(base_layer)
                    dct_results_matrices["nb-non-zero-base"].append(nb_weights_base_layer)
                    nb_weights_compressed_layer = get_nb_learnable_weights(compressed_layer)
                    dct_results_matrices["nb-non-zero-compressed"].append(nb_weights_compressed_layer)
                    dct_results_matrices["nb-non-zero-compression-rate"].append(nb_weights_base_layer/nb_weights_compressed_layer)
                    if is_dense:
                        nb_param_dense_base += nb_weights_base_layer
                        nb_param_dense_compressed += nb_weights_compressed_layer
                    else:
                        nb_param_conv_base += nb_weights_base_layer
                        nb_param_conv_compressed += nb_weights_compressed_layer
                    # get palm setting options
                    dct_results_matrices["nb-factor-param"].append(dct_attributes["nb-factor"][-1])
                    # dct_results_matrices["nb-factor-actual"].append(len(sparsity_patterns))
                    dct_results_matrices["sparsity-factor"].append(dct_attributes["sparsity-factor"][-1])
                    dct_results_matrices["hierarchical"].append(dct_attributes["hierarchical"][-1])

            gc.collect()
            palmnet.hunt.show_most_common_types(limit=20)
            log_memory_usage("Before dels")
            del dct_name_facto
            del base_model
            del compressed_model
            del base_layer
            del compressed_layer
            del sparse_factorization
            K.clear_session()

            gc.collect()
            log_memory_usage("After dels")
            palmnet.hunt.show_most_common_types(limit=20)

        dct_attributes["nb-param-base-dense"].append(int(nb_param_dense_base))
        dct_attributes["nb-param-base-conv"].append(int(nb_param_conv_base))
        dct_attributes["nb-param-compressed-dense"].append(int(nb_param_dense_compressed))
        dct_attributes["nb-param-compressed-conv"].append(int(nb_param_conv_compressed))

        dct_attributes["nb-param-compression-rate-dense"].append(dct_attributes["nb-param-base-dense"][-1] / dct_attributes["nb-param-compressed-dense"][-1])
        try:
            dct_attributes["nb-param-compression-rate-conv"].append(dct_attributes["nb-param-base-conv"][-1] / dct_attributes["nb-param-compressed-conv"][-1])
        except ZeroDivisionError:
            dct_attributes["nb-param-compression-rate-conv"].append(np.nan)
    df_results = pd.DataFrame.from_dict(dct_attributes)
    # if df_results_tmp is not None:
    #     df_results = pd.concat([df_results, df_results_tmp])
    df_results.to_csv(output_dir / "results.csv")

    df_results_layers = pd.DataFrame.from_dict(dct_results_matrices)
    # if df_results_layers_tmp is not None:
    #     df_results_layers = pd.concat([df_results_layers, df_results_layers_tmp])
    df_results_layers.to_csv(output_dir / "results_layers.csv")
