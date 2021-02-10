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
from palmnet.core.layer_replacer_TT import LayerReplacerTT
from palmnet.core.layer_replacer_palm import LayerReplacerPalm
from palmnet.data import param_training, image_data_generator_cifar_svhn, image_data_generator_mnist
from palmnet.experiments.utils import get_line_of_interest, ParameterManager
from palmnet.layers.low_rank_dense_layer import LowRankDense
from palmnet.layers.tt_layer_conv import TTLayerConv
from palmnet.layers.tt_layer_dense import TTLayerDense
from palmnet.layers.tucker_layer import TuckerLayerConv
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


columns_not_to_num = [
    'hash',
    'output_file_csvcbprinter',
    'results_dir',
    'output_file_modelprinter',
    'output_file_resprinter',
    'output_file_tensorboardprinter',
    'oar_id',
    'input_model_path',
    'output_file_finishedprinter',
    'output_file_notfinishedprinter',
    "output_file_layerbylayer",
    "output_file_objectives",
    "ouput_file_objectives"
]

def get_df_from_expe_path(expe_path):
    src_dir = root_source_dir / expe_path
    df = get_df(src_dir)
    df = df.assign(results_dir=[str(src_dir.absolute())] * len(df))
    return df

def cast_to_num(df):
    for col in df.columns.difference(columns_not_to_num):
        if col in df.columns.values:
            df.loc[:, col] = df.loc[:, col].apply(pd.to_numeric, errors='coerce')
    return df

if __name__ == "__main__":
    root_source_dir = pathlib.Path("/home/luc/PycharmProjects/palmnet/results/")
    experiment_name = "2020/11/11_12_compression_palm_act"

    lst_paths_finetune = [
        "2020/09/6_7_compression_palm_act",
        "2020/09/6_7_compression_palm_act_cifar100",
        "2020/09/8_9_compression_palm_act_full_net",
        "2020/09/9_10_compression_palm_act_only_one_batch",
        "2020/09/9_10_compression_palm_act_full_net_only_1_batch_300",
        "2020/11/9_10_compression_palm_act_full_net_1iter",
        "2020/11/9_10_compression_palm_act_full_net_1iterx10",
        "2020/11/11_12_compression_palm_epsilon_act",
        "2020/11/11_12_compression_sparse_facto_epsilon_act_no_act",
        "2020/11/11_12_compression_palm_one_batch_other_sizes"
    ]

    df_compression = pd.concat(list(map(get_df_from_expe_path, lst_paths_finetune)))
    # df_compression = get_df_from_expe_path(lst_paths_finetune[0])
    df_compression = df_compression.dropna(subset=["failure"])
    df_compression = df_compression[df_compression["failure"] == False]
    df_compression = df_compression.drop(columns="oar_id").drop_duplicates()
    df_compression = cast_to_num(df_compression)


    root_output_dir = pathlib.Path("/home/luc/PycharmProjects/palmnet/results/processed/")
    output_dir = root_output_dir / experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)


    dct_attributes = defaultdict(lambda: [])
    dct_results_matrices = defaultdict(lambda: [])

    length_df = len(df_compression)

    for idx, (_, row) in enumerate(df_compression.iterrows()):

        log_memory_usage("Start loop")
        print("row {}/{}".format(idx, length_df))
        dct_attributes["idx-expe"].append(idx)
        dct_attributes["hash"].append(row["hash"])

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
        elif row["--cifar100-resnet20"] or row["--cifar100-resnet20-new"]:
            dct_attributes["model"].append("resnet20")
        elif row["--cifar100-resnet50"] or row["--cifar100-resnet50-new"]:
            dct_attributes["model"].append("resnet50")
        else:
            raise ValueError("Unknown model")

        dct_attributes["activations"].append(row["--activations"])
        dct_attributes["nb-epochs"].append(int(row["--nb-epochs"]) if not np.isnan(row["--nb-epochs"]) else np.nan)
        dct_attributes["batch-size"].append(int(row["--batch-size"]) if not np.isnan(row["--batch-size"]) else np.nan)
        # dct_attributes["max-cum-batch-size"].append(int(row["--max-cum-batch-size"]) if not np.isnan(row["--max-cum-batch-size"]) else np.nan)
        dct_attributes["nb-fac"].append(int(row["--nb-factor"]))
        dct_attributes["sparsity-level"].append(int(row["--sparsity-factor"]))
        dct_attributes["nb-iteration-palm"].append(int(row["--nb-iteration-palm"]))
        dct_attributes["train-val-split"].append(float(row["--train-val-split"]))
        dct_attributes["only-one-batch"].append(bool(row["--only-one-batch"]) if not np.isnan(row["--only-one-batch"]) else False)
        dct_attributes["epsilon-lr"].append(float(row["--epsilon-learning-rate"]) if not np.isnan(row["--epsilon-learning-rate"]) else None)
        print(dct_attributes["nb-iteration-palm"][-1])
        dct_attributes["full-net-approx"].append(bool(row["--full-model-approx"]) if not np.isnan(row["--full-model-approx"]) else np.nan)

        dct_attributes["nb-param-compressed-total"].append(int(row["new_model_nb_param"]))
        dct_attributes["nb-param-base-total"].append(int(row["base_model_nb_param"]))
        dct_attributes["param-compression-rate-total"].append(row["base_model_nb_param"] / row["new_model_nb_param"])

        nb_param_dense_base = 0
        nb_param_dense_compressed = 0
        nb_param_conv_base = 0
        nb_param_conv_compressed = 0

        if type(row["output_file_layerbylayer"]) == str:

            path_layer_by_layer = pathlib.Path(row["results_dir"]) / row["output_file_layerbylayer"]
            df_csv_layerbylayer = pd.read_csv(str(path_layer_by_layer))

            for idx_row_layer, row_layer in df_csv_layerbylayer.iterrows():
                # get informations to identify the layer (and do cross references)
                dct_results_matrices["idx-expe"].append(idx)
                dct_results_matrices["model"].append(dct_attributes["model"][-1])
                layer_name_compressed = row_layer["layer-name-compressed"]
                is_dense = "dense" in layer_name_compressed or "fc" in layer_name_compressed or "predictions_cifa10" in layer_name_compressed
                dct_results_matrices["layer-name-compressed"].append(row_layer["layer-name-compressed"])
                dct_results_matrices["layer-name-base"].append(row_layer["layer-name-base"])
                dct_results_matrices["idx-layer"].append(row_layer["idx-layer"])
                dct_results_matrices["data"].append(dct_attributes["dataset"][-1])
                try:
                    dct_results_matrices["diff-only-layer-processing-train"].append(row_layer["diff-only-layer-processing-train"])
                    dct_results_matrices["diff-total-processing-train"].append(row_layer["diff-total-processing-train"])

                except KeyError:
                    dct_results_matrices["diff-only-layer-processing-train"].append(None)
                    dct_results_matrices["diff-total-processing-train"].append(None)

                try:
                    dct_results_matrices["diff-only-layer-processing-val"].append(row_layer["diff-only-layer-processing-val"])
                    dct_results_matrices["diff-total-processing-val"].append(row_layer["diff-total-processing-val"])
                except KeyError:
                    dct_results_matrices["diff-only-layer-processing-val"].append(row_layer["diff-only-layer-processing"])
                    dct_results_matrices["diff-total-processing-val"].append(row_layer["diff-total-processing"])

                dct_results_matrices["diff-approx"].append(row_layer["diff-approx"])
                dct_results_matrices["nb-iteration-palm"].append(int(row["--nb-iteration-palm"]))
                dct_results_matrices["train-val-split"].append(float(row["--train-val-split"]))
                dct_results_matrices["nb-epochs"].append(int(row["--nb-epochs"]) if not np.isnan(row["--nb-epochs"]) else np.nan)
                dct_results_matrices["batch-size"].append(int(row["--batch-size"]) if not np.isnan(row["--batch-size"]) else np.nan)
                dct_results_matrices["only-one-batch"].append(dct_attributes["only-one-batch"][-1])
                dct_results_matrices["epsilon-lr"].append(dct_attributes["epsilon-lr"][-1])

                dct_results_matrices["sparsity-level"].append(dct_attributes["sparsity-level"][-1])
                dct_results_matrices["nb-fac"].append(dct_attributes["nb-fac"][-1])
                dct_results_matrices["activations"].append(dct_attributes["activations"][-1])
                dct_results_matrices["full-net-approx"].append(dct_attributes["full-net-approx"][-1])
                # complexity analysis #
                # get nb val base layer and comrpessed layer
                dct_results_matrices["nb-non-zero-base"].append(int(row_layer["nb-non-zero-base"]))
                dct_results_matrices["nb-non-zero-compressed"].append(int(row_layer["nb-non-zero-compressed"]))
                dct_results_matrices["nb-non-zero-compression-rate"].append(row_layer["nb-non-zero-compression-rate"])
                if is_dense:
                    nb_param_dense_base += row_layer["nb-non-zero-base"]
                    nb_param_dense_compressed += row_layer["nb-non-zero-compressed"]
                else:
                    nb_param_conv_base += row_layer["nb-non-zero-base"]
                    nb_param_conv_compressed += row_layer["nb-non-zero-compressed"]

        dct_attributes["ouptut_file_objectives"].append(pathlib.Path(row["results_dir"])  / row["ouput_file_objectives"])

        dct_attributes["nb-param-base-dense"].append(int(nb_param_dense_base))
        dct_attributes["nb-param-base-conv"].append(int(nb_param_conv_base))
        dct_attributes["nb-param-compressed-dense"].append(int(nb_param_dense_compressed))
        dct_attributes["nb-param-compressed-conv"].append(int(nb_param_conv_compressed))

        try:
            dct_attributes["nb-param-compression-rate-dense"].append(dct_attributes["nb-param-base-dense"][-1] / dct_attributes["nb-param-compressed-dense"][-1])
        except ZeroDivisionError:
            dct_attributes["nb-param-compression-rate-dense"].append(np.nan)
        try:
            dct_attributes["nb-param-compression-rate-conv"].append(dct_attributes["nb-param-base-conv"][-1] / dct_attributes["nb-param-compressed-conv"][-1])
        except ZeroDivisionError:
            dct_attributes["nb-param-compression-rate-conv"].append(np.nan)

    df_results = pd.DataFrame.from_dict(dct_attributes)
    df_results.to_csv(output_dir / "results.csv")

    df_results_layers = pd.DataFrame.from_dict(dct_results_matrices)
    df_results_layers.to_csv(output_dir / "results_layers.csv")
