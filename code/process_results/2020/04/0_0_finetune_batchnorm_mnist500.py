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
    "lenet": mnist_param_training
}


if __name__ == "__main__":
    root_source_dir = pathlib.Path("/home/luc/PycharmProjects/palmnet/results/")
    root_source_dir_backup = pathlib.Path("/home/luc/ownCloudLIS/PycharmProjects/palmnet/results/")

    # load results standard palm mnist500 #
    #######################################
    # path after finetune standard palm facto #
    # expe_path_palminized_mnist_500 = "2020/02/3_4_finetune_palminized_mnist_500"
    # expe_path_palminized_great_fac_number = "2020/02/3_4_finetune_palminized_greater_fac_number_mnist_only"
    # expe_path_palminized_7_fac = "2020/02/3_4_finetune_palminized_7_fac_mnist_only"

    # path before finetune standard palm facto #
    expe_path_palminized_mnist_500_before_finetune = "2020/02/1_2_palminize_mnist500"
    expe_path_palminized_great_fac_number_before_finetune = "2020/02/1_2_palmnet_zero_greater_fac_number"
    expe_path_palminized_great_7_fac_before_finetune = "2020/02/1_2_palmnet_zero_7_fac"

    # results dir after finetune
    # src_results_dir_palminized_great_fac_number = root_source_dir / expe_path_palminized_great_fac_number
    # src_results_dir_palminized_7_fac = root_source_dir / expe_path_palminized_7_fac
    # src_results_dir_palminized_mnist_500 = root_source_dir / expe_path_palminized_mnist_500

    # results dir before finetune
    src_results_dir_palminized_7_fac_before_finetune = root_source_dir / expe_path_palminized_great_7_fac_before_finetune
    src_results_dir_palminized_mnist_500_before_finetune = root_source_dir / expe_path_palminized_mnist_500_before_finetune
    src_results_dir_palminized_great_fac_number_before_finetune = root_source_dir / expe_path_palminized_great_fac_number_before_finetune

    # df_palminized_great_fac_number = get_df(src_results_dir_palminized_great_fac_number)
    # df_palminized_7_fac_number = get_df(src_results_dir_palminized_7_fac)
    # df_palminized_mnist_500 = get_df(src_results_dir_palminized_mnist_500)
    # df_palminized = pd.concat([df_palminized_7_fac_number, df_palminized_mnist_500, df_palminized_great_fac_number])
    # df_palminized = df_palminized.dropna(subset=["failure"])
    # df_palminized = df_palminized[df_palminized["finetuned_score"] != "None"]
    # df_palminized = df_palminized.drop(columns="oar_id").drop_duplicates()

    df_palminized_great_fac_number_before_finetune = get_df(src_results_dir_palminized_great_fac_number_before_finetune)
    df_palminized_7_fac_number_before_finetune = get_df(src_results_dir_palminized_7_fac_before_finetune)
    df_palminized_mnist_500_before_finetune = get_df(src_results_dir_palminized_mnist_500_before_finetune)
    df_palminized_before_finetune = pd.concat([df_palminized_7_fac_number_before_finetune, df_palminized_mnist_500_before_finetune, df_palminized_great_fac_number_before_finetune])
    df_palminized_before_finetune = df_palminized_before_finetune.apply(pd.to_numeric, errors='coerce')
    # load results batchnorm palm mnist500 #
    ########################################
    # after finetune
    expe_path_batchnorm = "2020/04/0_0_finetune_layerreplacerpalminizer_batchnorm_mnist500"

    src_results_batchnorm = root_source_dir / expe_path_batchnorm

    df_batchnorm = get_df(src_results_batchnorm)
    df_batchnorm = df_batchnorm.dropna(subset=["failure"])
    df_batchnorm = df_batchnorm[df_batchnorm["failure"] == False]
    df_batchnorm = df_batchnorm[df_batchnorm["test_accuracy_finetuned_model"] != "None"]
    df_batchnorm = df_batchnorm.drop(columns="oar_id").drop_duplicates()

    root_output_dir = pathlib.Path("/home/luc/PycharmProjects/palmnet/results/processed/")
    output_dir = root_output_dir / expe_path_batchnorm
    output_dir.mkdir(parents=True, exist_ok=True)

    df_all = df_batchnorm
    # df_all = pd.concat([df_batchnorm, df_palminized])

    columns_not_to_num = ['hash', 'output_file_csvcbprinter']
    # df_palminized_no_useless = df_palminized_no_useless.apply(pd.to_numeric, errors='coerce')
    df_all.loc[:, df_all.columns.difference(columns_not_to_num)] = df_all.loc[:, df_all.columns.difference(columns_not_to_num)].apply(pd.to_numeric, errors='coerce')
    df_all = df_all.sort_values(by=["hash"])

    dct_attributes = defaultdict(lambda: [])
    dct_results_matrices = defaultdict(lambda: [])

    length_df = len(df_all)

    for idx, (_, row) in enumerate(df_all.iterrows()):

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
                            '--mnist-500',
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
        dct_attributes["batchnorm"].append(bool(row["--batchnorm"]) if not np.isnan(row["--batchnorm"]) else False)

        # score informations
        # dct_attributes["base-model-score"].append(float(row["base_score"]) if not np.isnan(row["base_score"]) else float(row["test_accuracy_base_model"]))
        # dct_attributes["before-finetune-score"].append(float(row["before_finetuned_score"]) if not np.isnan(row["before_finetuned_score"]) else float(row["test_accuracy_compressed_model"]))
        # dct_attributes["finetuned-score"].append(float(row["finetuned_score"]) if not np.isnan(row["finetuned_score
        dct_attributes["base-model-score"].append(float(row["test_accuracy_base_model"]))
        dct_attributes["before-finetune-score"].append(float(row["test_accuracy_compressed_model"]))
        dct_attributes["finetuned-score"].append(float(row["test_accuracy_finetuned_model"]))

    df_results = pd.DataFrame.from_dict(dct_attributes)
    df_results.to_csv(output_dir / "results.csv")