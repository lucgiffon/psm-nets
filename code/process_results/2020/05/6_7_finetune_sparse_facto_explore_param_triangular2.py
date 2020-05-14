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


if __name__ == "__main__":
    root_source_dir = pathlib.Path("/home/luc/PycharmProjects/palmnet/results/")

    expe_path_explore_params ="2020/05/12_13_finetune_palminized_explore_param_bis_bis"
    expe_path_explore_params_errors ="2020/05/12_13_finetune_palminized_explore_param_bis_bis_errors"

    lst_expe_path = [
        "2020/05/12_13_finetune_palminized_explore_param_bis_bis",
        "2020/05/12_13_finetune_palminized_explore_param_bis_bis_errors",
        "2020/05/12_13_finetune_palminized_explore_param_triangular2",
        "2020/05/12_13_finetune_palminized_explore_param_triangular2_more_epochs",
        "2020/05/6_7_finetune_sparse_facto_explore_param_triangular2_even_more_epochs"
    ]

    # src_results_dir_explore_params = root_source_dir / expe_path_explore_params
    # src_results_dir_explore_params_errors = root_source_dir / expe_path_explore_params_errors

    get_results_from_path = lambda x: get_df(root_source_dir / x).assign(results_dir=root_source_dir / x)

    df_explore_params = pd.concat(list(map(get_results_from_path, lst_expe_path)))

    # df_explore_params = get_df(src_results_dir_explore_params)
    # df_explore_params = df_explore_params.assign(results_dir=[str(src_results_dir_explore_params.absolute())] * len(df_explore_params))
    # df_explore_params_errors = get_df(src_results_dir_explore_params_errors)
    # df_explore_params_errors = df_explore_params_errors.assign(results_dir=[str(src_results_dir_explore_params_errors.absolute())] * len(df_explore_params_errors))
    # df_explore_params = pd.concat([df_explore_params, df_explore_params_errors])

    df_explore_params = df_explore_params.dropna(subset=["failure"])
    df_explore_params = df_explore_params[df_explore_params["failure"] == False]
    df_explore_params = df_explore_params[df_explore_params["finetuned_score"] != "None"]
    df_explore_params = df_explore_params.drop(columns="oar_id").drop_duplicates()

    root_output_dir = pathlib.Path("/home/luc/PycharmProjects/palmnet/results/processed/")
    output_dir = root_output_dir / expe_path_explore_params
    output_dir.mkdir(parents=True, exist_ok=True)

    columns_not_to_num = ['hash', 'output_file_csvcbprinter', 'results_dir', 'output_file_modelprinter', "--use-clr", "output_file_csvcbprinter_epoch"]
    # df_palminized_no_useless = df_palminized_no_useless.apply(pd.to_numeric, errors='coerce')
    df_explore_params.loc[:, df_explore_params.columns.difference(columns_not_to_num)] = df_explore_params.loc[:, df_explore_params.columns.difference(columns_not_to_num)].apply(pd.to_numeric, errors='coerce')
    df_explore_params = df_explore_params.sort_values(by=["hash"])

    dct_attributes = defaultdict(lambda: [])

    length_df = len(df_explore_params)

    for idx, (_, row) in enumerate(df_explore_params.iterrows()):
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
        dct_attributes["use-clr"].append(row["--use-clr"])  # this must be first because used in other attributes
        dct_attributes["only-mask"].append(bool(row["--only-mask"]))
        dct_attributes["keep-last-layer"].append(bool(row["--keep-last-layer"]))
        # beware of this line here because the params_optimizer may change between experiments
        dct_attributes["learning-rate"].append(float(dct_param_train_model[dct_attributes["model"][-1]].params_optimizer["lr"]))
        dct_attributes["min-lr"].append(float(dct_param_train_model[dct_attributes["model"][-1]].min_lr) if dct_attributes["use-clr"][-1] else np.nan)
        dct_attributes["max-lr"].append(float(dct_param_train_model[dct_attributes["model"][-1]].max_lr) if dct_attributes["use-clr"][-1] else np.nan)
        dct_attributes["nb-epoch"].append(int(dct_param_train_model[dct_attributes["model"][-1]].epochs))
        dct_attributes["epoch-step-size"].append(float(row["--epoch-step-size"]) if dct_attributes["use-clr"][-1] else np.nan)
        dct_attributes["actual-batch-size"].append(int(row["actual-batch-size"]) if row["actual-batch-size"] is not None else None)
        dct_attributes["actual-nb-epochs"].append(int(row["actual-nb-epochs"]) if row["actual-nb-epochs"] is not None else None)
        dct_attributes["actual-min-lr"].append(float(row["actual-min-lr"]) if row["actual-min-lr"] is not None else None)
        dct_attributes["actual-max-lr"].append(float(row["actual-max-lr"]) if row["actual-max-lr"] is not None else None)
        dct_attributes["actual-lr"].append(float(row["actual-lr"]) if row["actual-lr"] is not None else None)

        dct_attributes["batchnorm"].append(bool(row["--batchnorm"]) if not np.isnan(row["--batchnorm"]) else np.nan)

        # score informations
        dct_attributes["base-model-score"].append(float(row["base_score"]))
        dct_attributes["before-finetune-score"].append(float(row["before_finetuned_score"]))
        dct_attributes["finetuned-score"].append(float(row["finetuned_score"]))
        dct_attributes["logrange"].append(bool(row["--logrange-clr"]))
        # store path informations
        dct_attributes["path-learning-history"].append(pathlib.Path(row["results_dir"]) / row["output_file_csvcbprinter"])
        dct_attributes["path-learning-history-epoch"].append(pathlib.Path(row["results_dir"]) / row["output_file_csvcbprinter_epoch"])

    df_results = pd.DataFrame.from_dict(dct_attributes)
    df_results.to_csv(output_dir / "results.csv")
