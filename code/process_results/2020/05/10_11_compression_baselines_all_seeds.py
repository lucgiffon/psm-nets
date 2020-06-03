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

# copy pasta from the palmnet.data file at the time of the experiment
cifar100_param_training = param_training(
    batch_size=64,
    epochs=300,
    optimizer=keras.optimizers.Adam,
    params_optimizer={"lr":0.0001},
    min_lr=0.000005,
    max_lr=0.001,
    loss="categorical_crossentropy",
    image_data_generator=image_data_generator_cifar_svhn,
    # callbacks=[LearningRateScheduler(scheduler)]
    callbacks=[]
)
cifar100_resnet_param_training = param_training(
    batch_size=128,
    epochs=300,
    # optimizer=optimizers.SGD(lr=.1, momentum=0.9, nesterov=True),
    optimizer=keras.optimizers.Adam,
    params_optimizer={"lr": 0.00005},
    min_lr=0.000005,
    max_lr=0.001,
    loss="categorical_crossentropy",
    image_data_generator=image_data_generator_cifar_svhn,
    # callbacks=[LearningRateScheduler(scheduler_cifar100_resnet)],
    callbacks=[]
)

mnist_param_training = param_training(
    batch_size=32,
    epochs=100,
    optimizer=keras.optimizers.RMSprop,
    params_optimizer={"lr":0.0001, "decay":1e-6},
    loss="categorical_crossentropy",
    image_data_generator=image_data_generator_mnist,
    callbacks=[],
    min_lr=0.0001,
    max_lr=0.0001,
)

dct_param_train_model = {
    "resnet20": cifar100_resnet_param_training,
    "resnet50": cifar100_resnet_param_training,
    "vgg19": cifar100_param_training,
    "lenet": mnist_param_training
}

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
    "output_file_layerbylayer"
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
    experiment_name = "2020/05/10_11_compression_baselines_full"

    lst_paths_finetune = [
        "2020/05/10_11_compression_baselines_full",
        "2020/05/11_12_compression_baselines_magnitude_resnet"
    ]

    df_tucker_tt = pd.concat(list(map(get_df_from_expe_path, lst_paths_finetune)))
    df_tucker_tt = df_tucker_tt.dropna(subset=["failure"])
    df_tucker_tt = df_tucker_tt[df_tucker_tt["failure"] == False]
    df_tucker_tt = df_tucker_tt.drop(columns="oar_id").drop_duplicates()
    df_tucker_tt = cast_to_num(df_tucker_tt)

    root_output_dir = pathlib.Path("/home/luc/PycharmProjects/palmnet/results/processed/")
    output_dir = root_output_dir / experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)


    dct_attributes = defaultdict(lambda: [])
    dct_results_matrices = defaultdict(lambda: [])

    length_df = len(df_tucker_tt)

    for idx, (_, row) in enumerate(df_tucker_tt.iterrows()):
        cifar100_resnet = (row["--cifar100-resnet20-new"] is True or row["--cifar100-resnet50-new"] is True)
        if cifar100_resnet and np.isnan(row["test_loss_finetuned_model"]):
            continue

        if row["tucker"] is True:
            if cifar100_resnet and "2020/05/9_10_compression_tucker_only_resnet_grid_lr" not in row["results_dir"]:
                continue
            dct_attributes["compression"].append("tucker")
            # continue
        elif row["deepfried"] is True:
            if "2020/05/9_10_compression_deepfried_grid_lr" not in row["results_dir"]:
                continue
            dct_attributes["compression"].append("deepfried")
        elif row["magnitude"] is True:
            dct_attributes["compression"].append("magnitude")
        elif row["random"] is True:
            dct_attributes["compression"].append("random")
        else:
            if cifar100_resnet and "2020/05/9_10_compression_tensortrain_smaller_batch_size" not in row["results_dir"]:
                continue
            elif not cifar100_resnet and "2020/05/9_10_compression_tensortrain_smaller_batch_size" in row["results_dir"]:
                continue


            dct_attributes["compression"].append("tensortrain")

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

        # finetuning informations
        dct_attributes["use-clr"].append(bool(row["--use-clr"]))  # this must be first because used in other attributes
        dct_attributes["keep-last-layer"].append(bool(row["--keep-last-layer"]))
        dct_attributes["keep-first-layer"].append(bool(row["--keep-first-layer"]))
        # beware of this line here because the params_optimizer may change between experiments
        dct_attributes["learning-rate"].append(float(row["actual-lr"]))
        dct_attributes["min-lr"].append(float(dct_param_train_model[dct_attributes["model"][-1]].min_lr) if dct_attributes["use-clr"][-1] else np.nan)
        dct_attributes["max-lr"].append(float(dct_param_train_model[dct_attributes["model"][-1]].max_lr) if dct_attributes["use-clr"][-1] else np.nan)
        dct_attributes["nb-epoch"].append(int(dct_param_train_model[dct_attributes["model"][-1]].epochs))
        dct_attributes["epoch-step-size"].append(float(row["--epoch-step-size"]) if dct_attributes["use-clr"][-1] else np.nan)
        # dct_attributes['use-pretrained'].append(bool(row["--use-pretrained"]) if not np.isnan(row["--use-pretrained"]) else False)
        dct_attributes['only-dense'].append(bool(row["--only-dense"]) if not np.isnan(row["--only-dense"]) else False)
        # tensortrain informations
        dct_attributes["rank-value"].append(int(row["--rank-value"]) if not np.isnan(row["--rank-value"]) else np.nan)
        dct_attributes["order"].append(int(row["--order"]) if not np.isnan(row["--order"]) else np.nan)

        # tucker informations
        dct_attributes["rank-percentage-dense"].append(float(row["--rank-percentage-dense"]) if not np.isnan(row["--rank-percentage-dense"]) else np.nan)
        dct_attributes["rank-percentage-conv"].append(float(row["--rank-percentage-conv"]) if not np.isnan(row["--rank-percentage-conv"]) else np.nan)
        dct_attributes["rank-percentage"].append(float(row["--rank-percentage"]) if not np.isnan(row["--rank-percentage"]) else np.nan)

        # deepfried informations
        dct_attributes["nb-stack"].append(int(row["--nb-stack"]) if not np.isnan(row["--nb-stack"]) else np.nan)


        # magnitude informations
        dct_attributes["final-sparsity"].append(float(row["--final-sparsity"]) if not np.isnan(row["--final-sparsity"]) else np.nan)


        # random informations
        dct_attributes["nb-fac"].append(int(row["--nb-factor"]) if not np.isnan(row["--nb-factor"]) else np.nan)
        dct_attributes["sparsity-factor"].append(int(row["--sparsity-factor"]) if not np.isnan(row["--sparsity-factor"]) else np.nan)

        # score informations
        dct_attributes["base-model-score"].append(float(row["test_accuracy_base_model"]))
        dct_attributes["before-finetune-score"].append(float(row["test_accuracy_compressed_model"]))
        dct_attributes["finetuned-score"].append(float(row["test_accuracy_finetuned_model"]))
        dct_attributes["finetuned-score-val"].append(float(row["val_accuracy_finetuned_model"]))

        # store path informations
        path_history = pathlib.Path(row["results_dir"]) / row["output_file_csvcbprinter"]
        dct_attributes["path-learning-history"].append(path_history)
        path_model = pathlib.Path(row["results_dir"]) / row["output_file_modelprinter"]
        dct_attributes["path-model"].append(path_model)

        nb_param_dense_base = 0
        nb_param_dense_compressed = 0
        nb_param_conv_base = 0
        nb_param_conv_compressed = 0

        if type(row["output_file_layerbylayer"]) == str:
            dct_attributes["nb-param-compressed-total"].append(int(row["new_model_nb_param"]))
            dct_attributes["nb-param-base-total"].append(int(row["base_model_nb_param"]))
            dct_attributes["param-compression-rate-total"].append(row["base_model_nb_param"] / row["new_model_nb_param"])

            path_layer_by_layer = pathlib.Path(row["results_dir"]) / row["output_file_layerbylayer"]
            df_csv_layerbylayer = pd.read_csv(str(path_layer_by_layer))
            a=1

            for idx_row_layer, row_layer in df_csv_layerbylayer.iterrows():
                # get informations to identify the layer (and do cross references)
                dct_results_matrices["idx-expe"].append(idx)
                dct_results_matrices["model"].append(dct_attributes["model"][-1])
                dct_results_matrices["compression"].append(dct_attributes["compression"][-1])
                layer_name_compressed = row_layer["layer-name-compressed"]
                is_dense = "dense" in layer_name_compressed or "fc" in layer_name_compressed or "predictions_cifa10"
                dct_results_matrices["layer-name-compressed"].append(row_layer["layer-name-compressed"])
                dct_results_matrices["layer-name-base"].append(row_layer["layer-name-base"])
                dct_results_matrices["idx-layer"].append(row_layer["idx-layer"])
                dct_results_matrices["data"].append(dct_attributes["dataset"][-1])
                dct_results_matrices["keep-last-layer"].append(dct_attributes["keep-last-layer"][-1])
                dct_results_matrices["keep-first-layer"].append(dct_attributes["keep-first-layer"][-1])
                dct_results_matrices["use-clr"].append(dct_attributes["use-clr"][-1])

                # complexity analysis #
                # get nb val base layer and comrpessed layer
                dct_results_matrices["nb-non-zero-base"].append(int(row_layer["nb-non-zero-base"]))
                dct_results_matrices["nb-non-zero-compressed"].append(int(row_layer["nb-non-zero-compressed"]))
                dct_results_matrices["nb-non-zero-compression-rate"].append(int(row_layer["nb-non-zero-compression-rate"]))
                if is_dense:
                    nb_param_dense_base += row_layer["nb-non-zero-base"]
                    nb_param_dense_compressed += row_layer["nb-non-zero-compressed"]
                else:
                    nb_param_conv_base += row_layer["nb-non-zero-base"]
                    nb_param_conv_compressed += row_layer["nb-non-zero-compressed"]
                # tensortrain informations
                dct_results_matrices["rank-value"].append(int(row["--rank-value"]) if not np.isnan(row["--rank-value"]) else np.nan)
                dct_results_matrices["order"].append(int(row["--order"]) if not np.isnan(row["--order"]) else np.nan)

                # tucker informations
                dct_results_matrices["rank-percentage-dense"].append(int(row["--rank-percentage-dense"]) if not np.isnan(row["--rank-percentage-dense"]) else np.nan)

        else:
            # continue
            paraman = ParameterManager(row.to_dict())
            base_model = paraman.get_model()

            try:
                compressed_model = None
                compressed_model = keras.models.load_model(str(path_model.absolute()), custom_objects={'TuckerLayerConv': TuckerLayerConv,
                                                                                                   'TTLayerConv': TTLayerConv,
                                                                                                   'TTLayerDense': TTLayerDense,
                                                                                                   'LowRankDense': LowRankDense})
            except Exception as e:
                print(e)
                layer_replacer = LayerReplacerTT(rank_value=dct_attributes["rank-value"][-1], order=dct_attributes["order"][-1], keep_last_layer=dct_attributes["keep-last-layer"][-1],
                                                 keep_first_layer=dct_attributes["keep-first-layer"][-1])
                compressed_model = None
                compressed_model = layer_replacer.fit_transform(base_model)
            # load model
            # if row["tucker"] is True:
            #     compressed_model = keras.models.load_model(str(path_model.absolute()), custom_objects={'TuckerLayerConv': TuckerLayerConv,
            #                                                                                            'TTLayerConv': TTLayerConv,
            #                                                                                            'TTLayerDense': TTLayerDense})
            # else:

            #     layer_replacer = LayerReplacerTT(rank_value=dct_attributes["rank-value"][-1], order=dct_attributes["order"][-1], keep_last_layer=dct_attributes["keep-last-layer"][-1],
            #                     keep_first_layer=dct_attributes["keep-first-layer"][-1])
            #     compressed_model = layer_replacer.fit_transform(base_model)


            nb_learnable_weights_compressed_model = get_nb_learnable_weights_from_model(compressed_model)
            nb_learnable_weights_base_model = get_nb_learnable_weights_from_model(base_model)
            dct_attributes["nb-param-compressed-total"].append(int(nb_learnable_weights_compressed_model))
            dct_attributes["nb-param-base-total"].append(int(nb_learnable_weights_base_model))
            dct_attributes["param-compression-rate-total"].append(nb_learnable_weights_base_model/nb_learnable_weights_compressed_model)

            if len(base_model.layers) < len(compressed_model.layers):
                base_model = Model(inputs=base_model.inputs, outputs=base_model.outputs)

            assert len(base_model.layers) == len(compressed_model.layers)

            for idx_layer, compressed_layer in enumerate(compressed_model.layers):
                if any(isinstance(compressed_layer, _class) for _class in (TuckerLayerConv, TTLayerConv, TTLayerDense, LowRankDense)):
                    base_layer = base_model.layers[idx_layer]
                    log_memory_usage("Start secondary loop")

                    # get informations to identify the layer (and do cross references)
                    dct_results_matrices["idx-expe"].append(idx)
                    dct_results_matrices["model"].append(dct_attributes["model"][-1])
                    dct_results_matrices["compression"].append(dct_attributes["compression"][-1])
                    layer_name_compressed = compressed_layer.name
                    is_dense = "sparse_factorisation_dense" in layer_name_compressed
                    dct_results_matrices["layer-name-compressed"].append(compressed_layer.name)
                    dct_results_matrices["layer-name-base"].append(base_layer.name)
                    dct_results_matrices["idx-layer"].append(idx_layer)
                    dct_results_matrices["data"].append(dct_attributes["dataset"][-1])
                    dct_results_matrices["keep-last-layer"].append(dct_attributes["keep-last-layer"][-1])
                    dct_results_matrices["keep-first-layer"].append(dct_attributes["keep-first-layer"][-1])
                    dct_results_matrices["use-clr"].append(dct_attributes["use-clr"][-1])

                    # complexity analysis #
                    # get nb val base layer and comrpessed layer
                    nb_weights_base_layer = get_nb_learnable_weights(base_layer)
                    dct_results_matrices["nb-non-zero-base"].append(nb_weights_base_layer)
                    nb_weights_compressed_layer = get_nb_learnable_weights(compressed_layer)
                    dct_results_matrices["nb-non-zero-compressed"].append(nb_weights_compressed_layer)
                    dct_results_matrices["nb-non-zero-compression-rate"].append(nb_weights_base_layer / nb_weights_compressed_layer)
                    if is_dense:
                        nb_param_dense_base += nb_weights_base_layer
                        nb_param_dense_compressed += nb_weights_compressed_layer
                    else:
                        nb_param_conv_base += nb_weights_base_layer
                        nb_param_conv_compressed += nb_weights_compressed_layer
                    # tensortrain informations
                    dct_results_matrices["rank-value"].append(int(row["--rank-value"]) if not np.isnan(row["--rank-value"]) else np.nan)
                    dct_results_matrices["order"].append(int(row["--order"]) if not np.isnan(row["--order"]) else np.nan)

                    # tucker informations
                    dct_results_matrices["rank-percentage-dense"].append(int(row["--rank-percentage-dense"]) if not np.isnan(row["--rank-percentage-dense"]) else np.nan)

            gc.collect()
            palmnet.hunt.show_most_common_types(limit=20)
            log_memory_usage("Before dels")
            del compressed_model
            del compressed_layer
            K.clear_session()
            gc.collect()
            log_memory_usage("After dels")
            palmnet.hunt.show_most_common_types(limit=20)

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
