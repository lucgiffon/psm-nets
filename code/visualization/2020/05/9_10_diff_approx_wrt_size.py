import pathlib
import random

import pandas as pd
from palmnet.visualization.utils import get_palminized_model_and_df, get_df
import matplotlib.pyplot as plt
import numpy as np
import logging
from collections import defaultdict
import plotly.graph_objects as go
import plotly.io as pio

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

tasks = {
    # ("nb-non-zero-compressed", "nb-non-zero-base"),
    # ("nb-non-zero-compression-rate", "nb-non-zero-compression-rate"),
    ("diff-approx", "diff-approx"),
    # ("nb-non-zero-compressed", "nb-non-zero-base"),
    # ("nb-non-zero-compressed", "nb-non-zero-base"),
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

def get_palm_results():
    # results_path = "2020/03/9_10_finetune_palminized_no_useless"
    # results_path_2 = "2020/04/9_10_finetune_palminized_no_useless"
    # results_path = "2020/05/9_10_finetune_sparse_facto_not_log_all_seeds"
    results_path = "2020/07/11_12_finetune_sparse_facto_fix_replicates"
    src_results_path = root_source_dir / results_path / "results_layers.csv"
    # src_results_path_2 = root_source_dir / results_path_2 / "results_layers.csv"

    df = pd.read_csv(src_results_path, header=0)
    # df_2 = pd.read_csv(src_results_path_2, header=0)
    # df = pd.concat([df, df_2])
    df = df.fillna("None")
    df = df.drop(columns=["Unnamed: 0", "idx-expe"]).drop_duplicates()

    df = df[df["keep-last-layer"] == 0]
    # df = df[df["use-clr"] == 1]
    df = df[df["hierarchical"] == False]

    df = df.loc[((df["sparsity-factor"] == 14) | (df["sparsity-factor"] == 2)) & (df["nb-factor-param"] == 3) ]

    df = df.assign(**{"only-dense": False, "keep-first-layer": False,
                      "method": "PYQALM",
                      "method_id": 0})

    df = df.fillna("None")
    return df

def get_tucker_results():
    results_path_tucker = "2020/04/0_1_compression_tucker_tensortrain"
    src_results_path_tucker = root_source_dir / results_path_tucker / "results_layers.csv"

    df_tucker_tt = pd.read_csv(src_results_path_tucker, header=0)
    df_tucker_tt = df_tucker_tt.fillna("None")

    df_tucker_tt = df_tucker_tt.assign(**{"only-dense": False, "use-pretrained": False})

    df_tucker_tt = df_tucker_tt[df_tucker_tt["compression"] == "tucker"]
    return df_tucker_tt

def get_tensortrain_results():
    results_path_tucker = "2020/05/2_3_compression_tensortrain"
    src_results_path_tucker = root_source_dir / results_path_tucker / "results_layers.csv"

    df_tucker_tt = pd.read_csv(src_results_path_tucker, header=0)
    df_tucker_tt = df_tucker_tt.fillna("None")

    # df_tucker_tt = df_tucker_tt[df_tucker_tt["use-pretrained"] == True]
    # df_tucker_tt = df_tucker_tt[df_tucker_tt["only-dense"] == False]

    return df_tucker_tt

if __name__ == "__main__":
    root_source_dir = pathlib.Path("/home/luc/PycharmProjects/palmnet/results/processed")
    results_path_pyqalm = "2020/04/9_10_finetune_palminized_no_useless"


    root_output_dir = pathlib.Path("/home/luc/PycharmProjects/palmnet/reports/figures/")
    output_dir = root_output_dir / results_path_pyqalm / "diff_vs_size"
    output_dir.mkdir(parents=True, exist_ok=True)


    df = get_palm_results()
    df_tucker = get_tucker_results()
    df_tt = get_tensortrain_results()
    df_tucker_tt = pd.concat([df_tucker, df_tt])

    dct_config_color = {
        "PYQALM K=2 Q=2": "green",
        "PYQALM K=2 Q=3": "blue",
        "PYQALM K=2 Q=None": "olive",
        "PYQALM K=3 Q=2": "navy",
        "PYQALM K=3 Q=3": "blue",
        "PYQALM K=3 Q=None": "teal",
        "PYQALM K=4 Q=2": "maroon",
        "PYQALM K=4 Q=3": "red",
        "PYQALM K=4 Q=None": "purple",
        'PYQALM K=14 Q=2': "navy",
        'PYQALM K=14 Q=3': "red",
        'PYQALM K=8 Q=3': "green",
        'PYQALM K=8 Q=2': "green",
        "TT K=4 R=10": "yellow",
        "TT K=4 R=14": "orange",
        "TT K=4 R=2": "chocolate",
        "TT K=4 R=6": "darkgoldenrod",

        "Tucker": "indigo",

    }

    datasets = set(df["data"].values)
    dct_table = dict()
    for tpl_task in tasks:
        task_compressed = tpl_task[0]
        task_base = tpl_task[1]
        for dataname in datasets:
            df_data = df[df["data"] == dataname]
            df_tucker_tt_data = df_tucker_tt[df_tucker_tt["data"] == dataname]
            df_model_values = set(df_data["model"].values)

            dct_table[dataname] = dict()
            for modelname in df_model_values:
                dct_table[dataname][modelname] = list()

                dct_config_dct_layer_tpl_comp_nfac = defaultdict(lambda: dict())
                dct_legend_hover = dict()

                df_model = df_data[df_data["model"] == modelname]

                fig = go.Figure()

                sparsity_values = set(df_model["sparsity-factor"].values)

                for sp_val in sparsity_values:
                    df_sp = df_model[df_model["sparsity-factor"] == sp_val]

                    x_vals = df_sp["nb-non-zero-compressed"].values / df_sp["nb-non-zero-base"].values
                    where = x_vals == 1
                    x_vals = x_vals[~where]
                    y_vals = df_sp["diff-approx"].values
                    y_vals = y_vals[~where]

                    name_trace = f"Sparse Facto. K={sp_val} Q=3"
                    dict_name_trace_color = {
                        "Sparse Facto. K=14 Q=3": "red",
                        "Sparse Facto. K=2 Q=3": "blue"
                    }
                    fig.add_trace(go.Scatter(name=name_trace,
                                             x=x_vals, y=y_vals,
                                             marker_color=dict_name_trace_color[name_trace],
                                             # hovertext=hover_text,
                                             showlegend=True,
                                             mode='markers'))




                title = "_".join(str(x) for x in [dataname, modelname, task_compressed])

                x_legend = 0
                y_legend = -0.8
                fig.update_layout(
                    # barmode='group',
                    # title=title,
                    xaxis_title="Taille de la couche de base",
                    yaxis_title="Erreur relative",
                    # yaxis_type="log",
                    # xaxis={'type': 'category',
                    #        'tickvals': tickvals,
                    #        'ticktext': ticktexts},
                    # xaxis_tickangle=-75,
                    # showlegend=True,
                    showlegend=False,
                    autosize=False,
                    margin=dict(l=20, r=20, t=20, b=20),
                    width=800,
                    height=400,
                    font=dict(
                        # family="Courier New, monospace",
                        size=12,
                        color="black"
                    ),
                    legend_orientation="h",
                    legend=dict(
                        x=x_legend, y=y_legend,
                        traceorder="normal",
                        font=dict(
                            family="sans-serif",
                            size=18,
                            color="black"
                        ),
                        # bgcolor="LightSteelBlue",
                        # bordercolor="Black",
                        borderwidth=1,
                    )
                )
                # fig.show()
                fig.write_image(str((output_dir / title).absolute()) + ".png")
