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

    results_path = "2020/11/12_13_finetune_sparse_facto_palm_act_find_lr"


    src_results_path = root_source_dir / results_path / "results.csv"

    df = pd.read_csv(src_results_path, header=0)
    df = df.fillna("None")
    df = df.drop(columns=["Unnamed: 0", "idx-expe"]).drop_duplicates()


    root_output_dir = pathlib.Path("/home/luc/PycharmProjects/palmnet/parameters/2020/11")
    output_dir = root_output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    # sparsity_factors = sorted(set(df_palminized["--sparsity-factor"]))


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
            dct_results[dataname][modelname]["palm"] = dict()

            only_mask_values = set(df_model["only-mask"].values)
            for only_mask in only_mask_values:
                if only_mask:
                    str_only_mask = "only_mask"
                else:
                    str_only_mask = "weighted"
                df_only_mask = df_model[df_model["only-mask"] == only_mask]

                dct_results[dataname][modelname]["palm"][str_only_mask] = dict()

                sparsity_values = set(df_only_mask["sparsity-factor"].values)
                for sparsity in sparsity_values:
                    df_sparsity = df_only_mask[df_only_mask["sparsity-factor"] == sparsity]

                    dct_results[dataname][modelname]["palm"][str_only_mask][int(sparsity)] = dict()

                    nb_fac_values = set(df_sparsity["nb-factor"].values)
                    for nb_fac in nb_fac_values:
                        df_nb_fac = df_sparsity[df_sparsity["nb-factor"] == nb_fac]

                        lr_values = df_nb_fac["actual-lr"].values
                        score_values = df_nb_fac["finetuned-score-val"].values
                        best_score_indice = np.argmax(score_values)
                        assert score_values[best_score_indice] != "0.1"

                        dct_results[dataname][modelname]["palm"][str_only_mask][int(sparsity)][int(nb_fac)] = float(lr_values[best_score_indice])


    with open(output_dir / "finetune_sparse_facto_act.yml", 'w') as outfile:
        yaml.dump(dct_results, outfile, default_flow_style=False)

    with open(output_dir / "finetune_sparse_facto_act.yml", 'r') as outfile:
        dct_res_new = yaml.load(outfile, Loader=yaml.FullLoader)
    print(dct_res_new)

