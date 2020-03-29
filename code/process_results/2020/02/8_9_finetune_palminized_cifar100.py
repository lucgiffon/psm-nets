from collections import defaultdict

import pickle
import pathlib
import pandas as pd
import scipy.special
import scipy.stats
from palmnet.data import param_training, image_data_generator_cifar_svhn
from palmnet.experiments.utils import get_line_of_interest
from palmnet.utils import get_sparsity_pattern
from palmnet.visualization.utils import get_palminized_model_and_df, get_df
import matplotlib.pyplot as plt
import numpy as np
import logging
import plotly.graph_objects as go

from skluc.utils import logger
import keras

mpl_logger = logging.getLogger('matplotlib')
mpl_logger.setLevel(logging.ERROR)
logger.setLevel(logging.ERROR)

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

dct_param_train_model = {
    "resnet20": cifar100_resnet_param_training,
    "resnet50": cifar100_resnet_param_training,
    "vgg19": cifar100_param_training
}


if __name__ == "__main__":
    root_source_dir = pathlib.Path("/home/luc/PycharmProjects/palmnet/results/")

    expe_path_palminized_cifar100_resnet = "2020/02/8_9_finetune_palminized_resnet_new_lr"
    expe_path_palminized_cifar100_resnet_before_finetune = "2020/02/2_3_palminize_cifar_100_resnet"
    expe_path_palminized_cifar100_palm_before_finetune = "2020/02/1_2_palminize_cifar_100"
    expe_path_palminized_not_hierarchical_2_3_factors_before_finetune = "2019/12/0_1_palmnet_patient_zero"

    src_results_dir_palminized_cifar100_resnet = root_source_dir / expe_path_palminized_cifar100_resnet
    src_results_dir_palminized_cifar100_resnet_before_finetune = root_source_dir / expe_path_palminized_cifar100_resnet_before_finetune
    src_results_dir_palminized_cifar100_palm_before_finetune = root_source_dir / expe_path_palminized_cifar100_palm_before_finetune
    src_results_dir_palminized_not_hierarchical_2_3_factors_before_finetune = root_source_dir / expe_path_palminized_not_hierarchical_2_3_factors_before_finetune

    df_palminized = get_df(src_results_dir_palminized_cifar100_resnet)
    df_palminized = df_palminized.dropna(subset=["failure"])
    df_palminized = df_palminized[df_palminized["failure"] == False]
    df_palminized = df_palminized[df_palminized["finetuned_score"] != "None"]
    df_palminized = df_palminized.drop(columns="oar_id").drop_duplicates()

    df_palminized_before_finetune_cifar100_palm_before_finetune = get_df(src_results_dir_palminized_cifar100_palm_before_finetune)
    df_palminized_before_finetune_cifar100_palm_before_finetune = df_palminized_before_finetune_cifar100_palm_before_finetune.assign(results_dir=[str(src_results_dir_palminized_cifar100_palm_before_finetune.absolute())] * len(df_palminized_before_finetune_cifar100_palm_before_finetune))
    df_palminized_not_hierarchical_2_3_factors_before_finetune = get_df(src_results_dir_palminized_not_hierarchical_2_3_factors_before_finetune)
    df_palminized_not_hierarchical_2_3_factors_before_finetune = df_palminized_not_hierarchical_2_3_factors_before_finetune.assign(results_dir=[str(src_results_dir_palminized_not_hierarchical_2_3_factors_before_finetune.absolute())] * len(df_palminized_not_hierarchical_2_3_factors_before_finetune))
    df_palminized_resnet_before_finetune = get_df(src_results_dir_palminized_cifar100_resnet_before_finetune)
    df_palminized_resnet_before_finetune = df_palminized_resnet_before_finetune.assign(results_dir=[str(src_results_dir_palminized_cifar100_resnet_before_finetune.absolute())] * len(df_palminized_resnet_before_finetune))
    df_palminized_before_finetune = pd.concat([df_palminized_resnet_before_finetune, df_palminized_not_hierarchical_2_3_factors_before_finetune, df_palminized_before_finetune_cifar100_palm_before_finetune])

    root_output_dir = pathlib.Path("/home/luc/PycharmProjects/palmnet/results/processed/")
    output_dir = root_output_dir / expe_path_palminized_cifar100_resnet
    output_dir.mkdir(parents=True, exist_ok=True)

    columns_not_to_num = ['results_dir', 'output_file_modelprinter']
    df_palminized = df_palminized.apply(pd.to_numeric, errors='coerce')
    df_palminized_before_finetune.loc[:, df_palminized_before_finetune.columns.difference(columns_not_to_num)] = df_palminized_before_finetune.loc[:, df_palminized_before_finetune.columns.difference(columns_not_to_num)].apply(pd.to_numeric, errors='coerce')

    dct_attributes = defaultdict(lambda: [])

    dct_results_matrices = defaultdict(lambda: [])
    length_df = len(df_palminized)
    for idx, (_, row) in enumerate(df_palminized.iterrows()):
        print("row {}/{}".format(idx, length_df))
        dct_attributes["idx-expe"].append(idx)

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
        row_before_finetune = get_line_of_interest(df_palminized_before_finetune, keys_of_interest, row).iloc[0]


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

        dct_attributes["delta-threshold"].append(float(row["--delta-threshold"]))
        dct_attributes["hierarchical"].append(bool(row["--hierarchical"]))
        dct_attributes["nb-factor"].append(int(row["--nb-factor"]) if not np.isnan(row["--nb-factor"]) else np.nan)
        dct_attributes["nb-iteration-palm"].append(int(row["--nb-iteration-palm"]))
        dct_attributes["only-mask"].append(bool(row["--only-mask"]))
        dct_attributes["sparsity-factor"].append(int(row["--sparsity-factor"]))
        dct_attributes["use-clr"].append(bool(row["--use-clr"]))
        dct_attributes["base-model-score"].append(float(row["base_score"]))
        dct_attributes["before-finetune-score"].append(float(row["before_finetuned_score"]))
        dct_attributes["finetuned-score"].append(float(row["finetuned_score"]))
        dct_attributes["epoch-step-size"].append(float(row["--epoch-step-size"]) if dct_attributes["use-clr"][-1] else np.nan)

        dct_attributes["nb-flop-base"].append(int(row_before_finetune["nb_flops_base_layers_conv_dense"]))
        dct_attributes["nb-flop-compressed"].append(int(row_before_finetune["nb_flops_compressed_layers_conv_dense"]))
        dct_attributes["nb-param-base"].append(int(row_before_finetune["nb_param_base_layers_conv_dense"]))
        dct_attributes["nb-param-compressed"].append(int(row_before_finetune["nb_param_compressed_layers_conv_dense"]))
        dct_attributes["palminization-time"].append(float(row_before_finetune["palminization_time"]))
        dct_attributes["param-compression-rate"].append(dct_attributes["nb-param-base"][-1] / dct_attributes["nb-param-compressed"][-1])
        dct_attributes["flop-compression-rate"].append(dct_attributes["nb-flop-base"][-1] / dct_attributes["nb-flop-compressed"][-1])

        dct_attributes["learning-rate"].append(float(dct_param_train_model[dct_attributes["model"][-1]].params_optimizer["lr"]))
        dct_attributes["min-lr"].append(float(dct_param_train_model[dct_attributes["model"][-1]].min_lr) if dct_attributes["use-clr"][-1] else np.nan)
        dct_attributes["max-lr"].append(float(dct_param_train_model[dct_attributes["model"][-1]].max_lr) if dct_attributes["use-clr"][-1] else np.nan)
        dct_attributes["nb-epoch"].append(int(dct_param_train_model[dct_attributes["model"][-1]].epochs))


        # matrices analysis
        path_pickle = pathlib.Path(row_before_finetune["results_dir"]) / row_before_finetune["output_file_modelprinter"]
        model_obj = pickle.load(open(path_pickle, 'rb'))
        base_model = model_obj.base_model
        dct_name_facto = model_obj.sparsely_factorized_layers

        for idx_layer, layer in enumerate(base_model.layers):
            sparse_factorization = dct_name_facto[layer.name]
            if sparse_factorization != (None, None):
                print(layer.name)
                dct_results_matrices["idx-expe"].append(idx)
                dct_results_matrices["model"].append(dct_attributes["model"][-1])
                dct_results_matrices["layer-name"].append(layer.name)
                dct_results_matrices["idx-layer"].append(idx_layer)
                dct_results_matrices["data"].append(dct_attributes["dataset"][-1])
                # scaling = 1.
                scaling = sparse_factorization[0]
                # factors_sparse = [coo_matrix(fac.toarray()) for fac in sparse_factorization[1].get_list_of_factors()]
                factors = [fac.toarray() for fac in sparse_factorization[1].get_list_of_factors()]
                # sparsity_patterns = [get_sparsity_pattern(w.toarray()) for w in factors]
                sparsity_patterns = [get_sparsity_pattern(w) for w in factors]
                # factor_data_sparse = [f.data for f in factors_sparse]
                factor_data = factors
                reconstructed_matrix = np.linalg.multi_dot(factors) * scaling
                base_matrix = np.reshape(layer.get_weights()[0], reconstructed_matrix.shape)
                diff = np.linalg.norm(base_matrix - reconstructed_matrix) / np.linalg.norm(base_matrix)
                dct_results_matrices["diff-approx"].append(diff)

                U, S, V = np.linalg.svd(base_matrix)
                mean_sv = np.mean(S)
                quantiles = np.percentile(S, np.linspace(0, 100, 9))
                softmax_S = scipy.special.softmax(S)
                entropy_S = scipy.stats.entropy(softmax_S)
                dct_results_matrices["entropy-base-sv"].append(entropy_S)
                dct_results_matrices["nb-sv-base"].append(len(S))
                dct_results_matrices["entropy-base-sv-normalized"].append(entropy_S / scipy.stats.entropy(scipy.special.softmax(np.ones(len(S)))))
                dct_results_matrices["percent-sv-base-above-mean"].append(np.sum(S > mean_sv)/len(S))

                U, S_recons, V = np.linalg.svd(reconstructed_matrix)
                mean_sv_recons = np.mean(S_recons)
                softmax_S_recons = scipy.special.softmax(S_recons)
                entropy_S_recons = scipy.stats.entropy(softmax_S_recons)
                dct_results_matrices["entropy-recons-sv"].append(entropy_S_recons)
                dct_results_matrices["nb-sv-recons"].append(len(S_recons))
                dct_results_matrices["entropy-recons-sv-normalized"].append(entropy_S_recons / scipy.stats.entropy(scipy.special.softmax(np.ones(len(S_recons)))))
                dct_results_matrices["percent-sv-recons-above-mean"].append(np.sum(S_recons > mean_sv_recons) / len(S))

                sparsity_pattern_reconstructed = get_sparsity_pattern(reconstructed_matrix)
                nb_non_zero = int(np.sum(sparsity_pattern_reconstructed))
                size_bias = len(layer.get_weights()[-1])
                dct_results_matrices["nb-non-zero-reconstructed"].append(nb_non_zero + size_bias)
                max_possible_non_zero = np.prod(reconstructed_matrix.shape) + size_bias
                dct_results_matrices["nb-non-zero-base"].append(max_possible_non_zero)

                nb_val_sparse_facto = np.sum([np.sum(w) for w in sparsity_patterns]) + 1
                dct_results_matrices["nb-non-zero-compressed"].append(nb_val_sparse_facto + size_bias)

                ratio = dct_results_matrices["nb-non-zero-base"][-1] / dct_results_matrices["nb-non-zero-compressed"][-1]

                dct_results_matrices["nb-factor-param"].append(dct_attributes["nb-factor"][-1])
                dct_results_matrices["nb-factor-actual"].append(len(sparsity_patterns))

                dct_results_matrices["sparsity-factor"].append(dct_attributes["sparsity-factor"][-1])
                dct_results_matrices["hierarchical"].append(dct_attributes["hierarchical"][-1])


    df_results = pd.DataFrame.from_dict(dct_attributes)
    df_results.to_csv(output_dir / "results.csv")

    df_results_layers = pd.DataFrame.from_dict(dct_results_matrices)
    df_results_layers["compression-rate"] = df_results_layers["nb-non-zero-base"] / df_results_layers["nb-non-zero-compressed"]
    df_results_layers["non-zero-rate"] = df_results_layers["nb-non-zero-base"] / df_results_layers["nb-non-zero-reconstructed"]
    df_results_layers["non-zero-prop"] = df_results_layers["nb-non-zero-reconstructed"] / df_results_layers["nb-non-zero-base"]
    df_results_layers.to_csv(output_dir / "results_layers.csv")
