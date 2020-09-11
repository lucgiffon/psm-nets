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
    ("diff-only-layer-processing", "diff-only-layer-processing"),
    ("diff-total-processing", "diff-total-processing"),
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
    src_results_path = root_source_dir / results_path / "results_layers.csv"

    df = pd.read_csv(src_results_path, header=0)
    df = df.fillna("None")
    df = df.drop(columns=["Unnamed: 0", "idx-expe"]).drop_duplicates()

    df = df.fillna("None")
    return df

if __name__ == "__main__":
    root_source_dir = pathlib.Path("/home/luc/PycharmProjects/palmnet/results/processed")
    results_path = "2020/09/6_7_compression_palm_act"

    root_output_dir = pathlib.Path("/home/luc/PycharmProjects/palmnet/reports/figures/")
    output_dir = root_output_dir / results_path / "histo_compressions"
    output_dir.mkdir(parents=True, exist_ok=True)

    df = get_palm_results()

    dct_config_color = {
        "ACT K=3 Q=3": "navy",
        "K=3 Q=3": "blue",

    }

    datasets = set(df["data"].values)
    dct_table = dict()
    for tpl_task in tasks:
        task_compressed = tpl_task[0]
        task_base = tpl_task[1]
        for dataname in datasets:
            df_data = df[df["data"] == dataname]
            df_model_values = set(df_data["model"].values)

            dct_table[dataname] = dict()
            for modelname in df_model_values:
                dct_table[dataname][modelname] = list()

                dct_config_dct_layer_tpl_comp_nfac = defaultdict(lambda: dict())
                dct_legend_hover = dict()

                df_model = df_data[df_data["model"] == modelname]

                dct_idx_layer = dict()

                for idx_row, (_, row) in enumerate(df_model.iterrows()):
                    # config attributes
                    sp_fac_palm = int(row["sparsity-level"])
                    nb_fac_param = row["nb-fac"]
                    activations_str = "ACT " if row["activations"] else ""
                    # config string
                    str_config_legend = f"{activations_str}K={sp_fac_palm} Q={nb_fac_param}"
                    str_config_hover = f"{activations_str}K={sp_fac_palm} Q={nb_fac_param}"
                    dct_legend_hover[str_config_legend] = str_config_hover

                    # values
                    perf_compressed = row[task_compressed]
                    if task_base == "nb-non-zero-compression-rate":
                        perf_base = 1
                    else:
                        perf_base = row[task_base]

                    layer_name = row["layer-name-base"]
                    idx_layer = row["idx-layer"]
                    dct_idx_layer[idx_layer] = layer_name

                    dct_config_dct_layer_tpl_comp_nfac[str_config_legend][layer_name] = perf_compressed
                    if layer_name in dct_config_dct_layer_tpl_comp_nfac["Base"] and "diff" not in task_compressed:
                        assert perf_base == dct_config_dct_layer_tpl_comp_nfac["Base"][layer_name]
                    dct_config_dct_layer_tpl_comp_nfac["Base"][layer_name] = perf_base
                    # dct_config_dct_layer_tpl_comp_nfac[str_config_recons_legend][layer_name] = nnz_reconstructed

                fig = go.Figure()

                # organise les titres des configurations pour qu'elles apparaissent dans l'ordre
                if "diff" not in task_compressed:
                    sorted_configs = sorted(dct_legend_hover.keys()) + ["Base"]
                else:
                    sorted_configs = sorted(dct_legend_hover.keys())

                # attribue des noms "jolis" aux traces
                dct_name_trace = {
                    "K=3 Q=3": "Sparse Facto. K=3 Q=3",
                    "ACT K=3 Q=3": "ACT Sparse Facto. K=3 Q=3"
                }

                dct_config_color["Base"] = "black"
                dct_legend_hover["Base"] = "Base"
                for s_conf in sorted_configs:
                    if "Reconstructed" in s_conf:
                        continue
                    hover_text = dct_legend_hover[s_conf]
                    lst_layers = []
                    lst_configs = []

                    for idx_layer in sorted(dct_idx_layer.keys()):
                        layer_name = dct_idx_layer[idx_layer]
                        if layer_name in dct_config_dct_layer_tpl_comp_nfac[s_conf]:
                            nnz = dct_config_dct_layer_tpl_comp_nfac[s_conf][layer_name]
                            lst_layers.append(layer_name)
                            lst_configs.append(nnz)
                        else:
                            lst_layers.append(layer_name)
                            lst_configs.append(0)

                    fig.add_trace(go.Bar(name=dct_name_trace[s_conf],
                                         x=lst_layers, y=lst_configs,
                                         marker_color=dct_config_color[hover_text],
                                         hovertext=hover_text,
                                         showlegend=True
                                         ))


                title = "_".join(str(x) for x in [dataname, modelname, task_compressed])

                replace_name_layer = lambda str_name: (str_name.replace("conv2d_", "Conv2D ").replace("dense_", "Dense ").replace("fc", "Dense ").replace("conv", "Conv2D"))
                # replace_name_layer = lambda str_name: str_name
                tickvals = [dct_idx_layer[idx] for idx in sorted(dct_idx_layer.keys())]
                ticktexts = [replace_name_layer(elm) for elm in tickvals]
                ticktexts[-1] = "Softmax"

                count_conv = 1
                count_dense = 1
                for i_elm, elm in enumerate(ticktexts):
                    if "Conv2D" in elm:
                        ticktexts[i_elm] = f"Conv2D {count_conv}"
                        count_conv += 1
                    elif "Dense" in elm:
                        ticktexts[i_elm] = f"Dense {count_dense}"
                        count_dense += 1
                    else:
                        pass

                x_legend = 0
                y_legend = -0.8
                fig.update_layout(
                    barmode='group',
                    title=title,
                    xaxis_title="Nom de la couche",
                    yaxis_title="Erreur relative",
                    # yaxis_type="log",
                    xaxis={'type': 'category',
                           'tickvals': tickvals,
                           'ticktext': ticktexts},
                    xaxis_tickangle=-75,
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
                fig.update_xaxes(showline=True, ticks="outside", linewidth=2, linecolor='black', mirror=True)
                fig.update_yaxes(showline=True, ticks="outside", linewidth=2, linecolor='black', mirror=True)
                fig.show()
                # fig.write_image(str((output_dir / title).absolute()) + ".png")
