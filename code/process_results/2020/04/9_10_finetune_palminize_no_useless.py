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
from palmnet.core.layer_replacer_palm import LayerReplacerPalm
from palmnet.data import param_training, image_data_generator_cifar_svhn, image_data_generator_mnist
from palmnet.experiments.utils import get_line_of_interest
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
    experiment_name = "2020/04/9_10_finetune_palminized_no_useless"

    lst_paths_finetune = [
        "2020/03/9_10_finetune_palminized_no_useless",
        "2020/04/9_10_finetune_palminized_no_useless_only_vgg19_cifar100",
        "2020/04/9_10_finetune_palminized_no_useless_only_resnet",
        "2020/04/9_10_finetune_palminized_no_useless_only_vgg19_cifar100_errors",
        "2020/04/9_10_finetune_palminized_no_useless_only_resnet_errors",
    ]

    lst_paths_before_finetune = [
        "2020/03/2_3_palminize_from_scratch"
    ]

    df_palminized_no_useless = pd.concat(list(map(get_df_from_expe_path, lst_paths_finetune)))
    df_palminized_no_useless = df_palminized_no_useless.dropna(subset=["failure"])
    df_palminized_no_useless = df_palminized_no_useless[df_palminized_no_useless["failure"] == False]
    df_palminized_no_useless = df_palminized_no_useless.drop(columns="oar_id").drop_duplicates()
    df_palminized_no_useless = cast_to_num(df_palminized_no_useless)
    df_palminized_no_useless = df_palminized_no_useless[~df_palminized_no_useless["finetuned_score"].isnull()]

    df_palminize_from_scratch = get_df_from_expe_path(*lst_paths_before_finetune)
    df_palminize_from_scratch = cast_to_num(df_palminize_from_scratch)

    root_output_dir = pathlib.Path("/home/luc/PycharmProjects/palmnet/results/processed/")
    output_dir = root_output_dir / experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)

    dct_attributes = defaultdict(lambda: [])
    dct_results_matrices = defaultdict(lambda: [])

    length_df = len(df_palminized_no_useless)

    # if (output_dir / "results.csv").exists():
    #     df_results_tmp = pd.read_csv(output_dir / "results.csv")
    # else:
    #     df_results_tmp = None
    #
    # if (output_dir / "results_layers.csv").exists():
    #     df_results_layers_tmp = pd.read_csv(output_dir / "results_layers.csv")
    # else:
    #     df_results_layers_tmp = None

    for idx, (_, row) in enumerate(df_palminized_no_useless.iterrows()):
        # if df_results_tmp is not None and row["hash"] in df_results_tmp["hash"].values:
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
        row_before_finetune = get_line_of_interest(df_palminize_from_scratch, keys_of_interest, row).iloc[0]
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
        else:
            raise ValueError("Unknown model")

        # palm informations #
        dct_attributes["delta-threshold"].append(float(row["--delta-threshold"]))
        dct_attributes["hierarchical"].append(bool(row["--hierarchical"]))
        dct_attributes["nb-factor"].append(int(row["--nb-factor"]) if not np.isnan(row["--nb-factor"]) else np.nan)
        dct_attributes["nb-iteration-palm"].append(int(row["--nb-iteration-palm"]))
        dct_attributes["sparsity-factor"].append(int(row["--sparsity-factor"]))

        # finetuning informations
        dct_attributes["use-clr"].append(bool(row["--use-clr"]))  # this must be first because used in other attributes
        dct_attributes["only-mask"].append(bool(row["--only-mask"]))
        dct_attributes["keep-last-layer"].append(bool(row["--keep-last-layer"]))
        # beware of this line here because the params_optimizer may change between experiments
        dct_attributes["learning-rate"].append(float(dct_param_train_model[dct_attributes["model"][-1]].params_optimizer["lr"]))
        dct_attributes["min-lr"].append(float(dct_param_train_model[dct_attributes["model"][-1]].min_lr) if dct_attributes["use-clr"][-1] else np.nan)
        dct_attributes["max-lr"].append(float(dct_param_train_model[dct_attributes["model"][-1]].max_lr) if dct_attributes["use-clr"][-1] else np.nan)
        dct_attributes["nb-epoch"].append(int(dct_param_train_model[dct_attributes["model"][-1]].epochs))
        dct_attributes["epoch-step-size"].append(float(row["--epoch-step-size"]) if dct_attributes["use-clr"][-1] else np.nan)

        # score informations
        dct_attributes["base-model-score"].append(float(row["base_score"]))
        dct_attributes["before-finetune-score"].append(float(row["before_finetuned_score"]))
        dct_attributes["finetuned-score"].append(float(row["finetuned_score"]))

        # store path informations
        path_pickle = pathlib.Path(row_before_finetune["results_dir"]) / row_before_finetune["output_file_modelprinter"]
        path_history = pathlib.Path(row["results_dir"]) / row["output_file_csvcbprinter"]
        dct_attributes["path-learning-history"].append(path_history)
        dct_attributes["path-pickle-model-object"].append(path_pickle)
        ##############################
        # Layer by Layer information #
        ##############################

        # matrices analysis

        palmnet.hunt.show_most_common_types(limit=20)
        log_memory_usage("Before pickle")
        model_obj = None
        with open(path_pickle, 'rb') as pkle:
            model_obj = pickle.load(pkle)
        log_memory_usage("After pickle")
        base_model = None
        base_model = model_obj.base_model
        palmnet.hunt.show_most_common_types(limit=20)
        layer_replacer = None
        layer_replacer = LayerReplacerPalm(only_mask=False, keep_last_layer=dct_attributes["keep-last-layer"][-1], dct_name_compression=model_obj.sparsely_factorized_layers,
                                           sparse_factorizer=Palminizer())
        compressed_model = None
        compressed_model = layer_replacer.transform(base_model)
        del layer_replacer
        palmnet.hunt.show_most_common_types(limit=20)
        log_memory_usage("After transform")

        if len(base_model.layers) < len(compressed_model.layers):
            base_model = Model(inputs=base_model.inputs, outputs=base_model.outputs)

        assert len(base_model.layers) == len(compressed_model.layers)

        # model complexity informations obtained from the base palminized model
        dct_attributes["nb-flop-base"].append(int(row_before_finetune["nb_flops_base_layers_conv_dense"]))
        dct_attributes["nb-flop-compressed"].append(int(row_before_finetune["nb_flops_compressed_layers_conv_dense"]))
        dct_attributes["nb-param-base"].append(int(row_before_finetune["nb_param_base_layers_conv_dense"]))
        dct_attributes["nb-param-compressed"].append(int(row_before_finetune["nb_param_compressed_layers_conv_dense"]))
        dct_attributes["palminization-time"].append(float(row_before_finetune["palminization_time"]))
        dct_attributes["param-compression-rate"].append(dct_attributes["nb-param-base"][-1] / dct_attributes["nb-param-compressed"][-1])
        dct_attributes["flop-compression-rate"].append(dct_attributes["nb-flop-base"][-1] / dct_attributes["nb-flop-compressed"][-1])
        # model complexity informations obtained from the reconstructed model
        nb_learnable_weights_base_model = get_nb_learnable_weights_from_model(base_model)
        nb_learnable_weights_compressed_model = get_nb_learnable_weights_from_model(compressed_model)
        dct_attributes["nb-param-base-total"].append(int(nb_learnable_weights_base_model))
        dct_attributes["nb-param-compressed-total"].append(int(nb_learnable_weights_compressed_model))
        dct_attributes["param-compression-rate-total"].append(nb_learnable_weights_base_model/nb_learnable_weights_compressed_model)

        dct_name_facto = None
        dct_name_facto = model_obj.sparsely_factorized_layers

        for idx_layer, base_layer in enumerate(base_model.layers):
            log_memory_usage("Start secondary loop")
            sparse_factorization = dct_name_facto.get(base_layer.name, (None, None))
            if sparse_factorization != (None, None):
                print(base_layer.name)
                compressed_layer = None
                compressed_layer = compressed_model.layers[idx_layer]

                # get informations to identify the layer (and do cross references)
                dct_results_matrices["idx-expe"].append(idx)
                dct_results_matrices["model"].append(dct_attributes["model"][-1])
                dct_results_matrices["layer-name-base"].append(base_layer.name)
                dct_results_matrices["layer-name-compressed"].append(compressed_layer.name)
                dct_results_matrices["idx-layer"].append(idx_layer)
                dct_results_matrices["data"].append(dct_attributes["dataset"][-1])
                dct_results_matrices["keep-last-layer"].append(dct_attributes["keep-last-layer"][-1])
                dct_results_matrices["use-clr"].append(dct_attributes["use-clr"][-1])

                # get sparse factorization
                scaling = sparse_factorization[0]
                factors = [fac.toarray() for fac in sparse_factorization[1].get_list_of_factors()]
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
                dct_results_matrices["nb-non-zero-reconstructed"].append(nb_non_zero + size_bias)
                # get nb val base layer and comrpessed layer
                nb_weights_base_layer = get_nb_learnable_weights(base_layer)
                dct_results_matrices["nb-non-zero-base"].append(nb_weights_base_layer)
                nb_weights_compressed_layer = get_nb_learnable_weights(compressed_layer)
                dct_results_matrices["nb-non-zero-compressed"].append(nb_weights_compressed_layer)
                dct_results_matrices["nb-non-zero-compression-rate"].append(nb_weights_base_layer/nb_weights_compressed_layer)

                # get palm setting options
                dct_results_matrices["nb-factor-param"].append(dct_attributes["nb-factor"][-1])
                dct_results_matrices["nb-factor-actual"].append(len(sparsity_patterns))
                dct_results_matrices["sparsity-factor"].append(dct_attributes["sparsity-factor"][-1])
                dct_results_matrices["hierarchical"].append(dct_attributes["hierarchical"][-1])

        gc.collect()
        palmnet.hunt.show_most_common_types(limit=20)
        log_memory_usage("Before dels")
        del dct_name_facto
        del model_obj
        del base_model
        del compressed_model
        del base_layer
        del compressed_layer
        del sparse_factorization
        K.clear_session()

        gc.collect()
        log_memory_usage("After dels")
        palmnet.hunt.show_most_common_types(limit=20)

    df_results = pd.DataFrame.from_dict(dct_attributes)
    # if df_results_tmp is not None:
    #     df_results = pd.concat([df_results, df_results_tmp])
    df_results.to_csv(output_dir / "results.csv")

    df_results_layers = pd.DataFrame.from_dict(dct_results_matrices)
    # if df_results_layers_tmp is not None:
    #     df_results_layers = pd.concat([df_results_layers, df_results_layers_tmp])
    df_results_layers.to_csv(output_dir / "results_layers.csv")
