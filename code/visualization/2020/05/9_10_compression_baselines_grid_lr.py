import pathlib
import pandas as pd
from palmnet.visualization.utils import get_palminized_model_and_df, get_df
import matplotlib.pyplot as plt
import numpy as np
import logging
import plotly.graph_objects as go
import plotly.io as pio
import yaml

mpl_logger = logging.getLogger('matplotlib')
mpl_logger.setLevel(logging.ERROR)
pio.templates.default = "plotly_white"

dataset = {
    "Cifar10": "--cifar10",
    "Cifar100": "--cifar100",
    # "SVHN": "--svhn",
    "MNIST": "--mnist"
}



models_data = {
    "Cifar10": ["--cifar10-vgg19"],
    # "Cifar100": ["--cifar100-resnet20", "--cifar100-resnet50"],
    "Cifar100": ["--cifar100-vgg19", "--cifar100-resnet20", "--cifar100-resnet50"],
    "SVHN": ["--svhn-vgg19"],
    "MNIST":["--mnist-lenet"],
}

color_bars_sparsity = {
    2: "g",
    3: "c",
    4: "b",
    5: "y"
}

tasks = {
    "nb-param-compressed-total",
    "finetuned-score",
    "param-compression-rate-total"
}
ylabel_task = {
    "nb-param-compressed-total": "log(# non-zero value)",
    "finetuned-score": "Accuracy",
    "param-compression-rate-total": "Compression Rate"
}

scale_tasks = {
    "nb-param-compressed-total": "log",
    "finetuned-score": "linear",
    "param-compression-rate-total": "linear"
}



if __name__ == "__main__":
    root_source_dir = pathlib.Path("/home/luc/PycharmProjects/palmnet/results/processed")


    SHOW_FAUST = False
    SHOW_KEEP_FIRST_ONLY = True
    SHOW_PRETRAINED_ONLY = True

    results_path = "2020/05/9_10_compression_baselines_grid_lr"


    src_results_path = root_source_dir / results_path / "results.csv"

    df = pd.read_csv(src_results_path, header=0)
    df = df.fillna("None")
    df = df.drop(columns=["Unnamed: 0", "idx-expe"]).drop_duplicates()


    root_output_dir = pathlib.Path("/home/luc/PycharmProjects/palmnet/parameters/2020/05")
    output_dir = root_output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    # sparsity_factors = sorted(set(df_palminized["--sparsity-factor"]))

    hue_by_sparsity= {
        2: 10,
        3: 60,
        4: 110,
        5: 180
    }

    saturation_by_perm = {
        1: 50,
        0: 75
    }

    saturation_by_hier = {
        1: 50,
        0: 75
    }

    lum_by_clr = {
        1: 20,
        0: 30
    }
    lum_by_keep = {
        1: 40,
        0: 50
    }

    dct_symbol = {
        "FAUST Q=2": "square",
        "FAUST Q=3": "diamond",
        "FAUST Q=None": "square-x",
        "FAUST Q=None H": "star-square",
        "PYQALM Q=2": "square-open",
        "PYQALM Q=3": "diamond-open",
        "PYQALM Q=None": "hash-open",
        "PYQALM Q=None H": "star-square-open",
        "PYQALM Q=2 -1": "square-open-dot",
        "PYQALM Q=3 -1": "diamond-open-dot",
        "PYQALM Q=None -1": "hash-open-dot",
        "PYQALM Q=None H -1": "star-square-open-dot",
        "PYQALM Q=2 -1 M": "square",
        "PYQALM Q=3 -1 M": "diamond",
        "PYQALM Q=None -1 M": "hash",
        "PYQALM Q=None H -1 M": "star-square",
        "Base": "x",
        "Tucker": "circle",
        "Tucker -1": "circle-dot",
        "TT": "triangle-up",
        "TT -1": "triangle-up-dot",
        "TT -1 pretrained": "triangle-up-open-dot",
    }

    dct_colors = {
        "PALM K=2": "dodgerblue",
        "PALM K=3": "darkorchid",
        "PALM K=4": "green",
        "PALM K=6": "aqua",
        "PALM K=8": "cadetblue",
        "TT R=2": "orange",
        "TT R=6": "gold",
        "TT R=10": "red",
        "TT R=12": "darkred",
        "TT R=14": "indianred",
        "Base": "grey",
        "Tucker": "pink",
        "Tucker + Low Rank 10%": "orange",
        "Tucker + Low Rank 20%": "gold",
        "Tucker + Low Rank 30%": "red"
    }

    SIZE_MARKERS = 15
    WIDTH_MARKER_LINES = 2

    dct_results = dict()

    datasets = set(df["dataset"].values)
    for dataname in datasets:
        df_data = df[df["dataset"] == dataname]
        df_model_values = set(df_data["model"].values)

        dct_results[dataname] = dict()

        for modelname in df_model_values:
            df_model = df_data[df_data["model"] == modelname]

            dct_results[dataname][modelname] = dict()


            # magnitude
            df_magnitude = df_model[df_model["compression"] == "magnitude"]
            dct_results[dataname][modelname]["magnitude"] = dict()
            sparsity_values = set(df_magnitude["final-sparsity"].values)
            for sparsity in sparsity_values:
                df_sparsity = df_magnitude[df_magnitude["final-sparsity"] == sparsity]
                lr_values = df_sparsity["learning-rate"].values
                score_values = df_sparsity["finetuned-score-val"].values
                best_score_indice = np.argmax(score_values)
                assert score_values[best_score_indice] != "0.1"
                dct_results[dataname][modelname]["magnitude"][float(sparsity)] = float(lr_values[best_score_indice])

            # random
            df_random = df_model[df_model["compression"] == "random"]
            dct_results[dataname][modelname]["random"] = dict()
            sparsity_values = set(df_random["sparsity-factor"].values)
            for sparsity in sparsity_values:
                df_sparsity = df_random[df_random["sparsity-factor"] == sparsity]
                dct_results[dataname][modelname]["random"][int(sparsity)] = dict()
                nb_fac_values = set(df_sparsity["nb-fac"].values)
                for nb_fac in nb_fac_values:
                    df_nb_fac = df_sparsity[df_sparsity["nb-fac"] == nb_fac]
                    lr_values = df_nb_fac["learning-rate"].values
                    score_values = df_nb_fac["finetuned-score-val"].values
                    best_score_indice = np.argmax(score_values)
                    assert score_values[best_score_indice] != "0.1"
                    dct_results[dataname][modelname]["random"][int(sparsity)][int(nb_fac)] = float(lr_values[best_score_indice])

            # tensortrain
            df_tensortrain = df_model[df_model["compression"] == "tensortrain"]
            dct_results[dataname][modelname]["tensortrain"] = dict()
            order_values = set(df_tensortrain["order"].values)
            for order in order_values:
                df_order = df_tensortrain[df_tensortrain["order"] == order]
                dct_results[dataname][modelname]["tensortrain"][int(order)] = dict()
                df_rank_values = set(df_order["rank-value"].values)
                for rank in df_rank_values:
                    df_rank = df_order[df_order["rank-value"] == rank]
                    lr_values = df_rank["learning-rate"].values
                    score_values = df_rank["finetuned-score-val"].values
                    best_score_indice = np.argmax(score_values)
                    assert score_values[best_score_indice] != "0.1"
                    dct_results[dataname][modelname]["tensortrain"][int(order)][int(rank)] = float(lr_values[best_score_indice])

            # tucker
            df_tucker = df_model[df_model["compression"] == "tucker"]
            dct_results[dataname][modelname]["tucker"] = dict()
            rank_percentage_values = set(df_tucker["rank-percentage-dense"].values)
            for rank_percentage in rank_percentage_values:
                df_rank = df_tucker[df_tucker["rank-percentage-dense"] == rank_percentage]
                lr_values = df_rank["learning-rate"].values
                score_values = df_rank["finetuned-score-val"].values
                best_score_indice = np.argmax(score_values)
                assert score_values[best_score_indice] != "0.1"
                dct_results[dataname][modelname]["tucker"][float(rank_percentage)] = float(lr_values[best_score_indice])

            # deepfried
            # df_deepfried = df_model[df_model["compression"] == "deepfried"]
            # dct_results[dataname][modelname]["deepfried"] = dict()
            # lr_values = df_deepfried["learning-rate"].values
            # score_values = df_deepfried["finetuned-score-val"].values
            # best_score_indice = np.argmax(score_values)
            # assert score_values[best_score_indice] != "0.1"
            # dct_results[dataname][modelname]["deepfried"] = float(lr_values[best_score_indice])


    with open(output_dir / "baselines_lr.yml", 'w') as outfile:
        yaml.dump(dct_results, outfile, default_flow_style=False)

    # with open(output_dir / "res.yml", 'r') as outfile:
    #     dct_res_new = yaml.load(outfile, Loader=yaml.FullLoader)
    # print(dct_res_new)

