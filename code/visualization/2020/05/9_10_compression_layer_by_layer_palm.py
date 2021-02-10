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
    output_dir = root_output_dir / results_path_pyqalm / "histo_compressions"
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
            if "cifar100" not in dataname:
                continue
            df_data = df[df["data"] == dataname]
            df_tucker_tt_data = df_tucker_tt[df_tucker_tt["data"] == dataname]
            df_model_values = set(df_data["model"].values)

            dct_table[dataname] = dict()
            for modelname in df_model_values:
                if "vgg" not in modelname:
                    continue
                dct_table[dataname][modelname] = list()

                dct_config_dct_layer_tpl_comp_nfac = defaultdict(lambda: dict())
                dct_legend_hover = dict()

                df_model = df_data[df_data["model"] == modelname]
                df_tucker_tt_model = df_tucker_tt_data[df_tucker_tt_data["model"] == modelname]

                dct_idx_layer = dict()

                for idx_row, (_, row) in enumerate(df_model.iterrows()):
                    # config attributes
                    sp_fac_palm = int(row["sparsity-factor"])
                    hierarchical_value = row["hierarchical"]
                    try:
                        nb_fac_param = int(row["nb-factor-param"])
                    except:
                        nb_fac_param = row["nb-factor-param"]

                    # nb_fac = int(row["nb-factor-actual"])

                    # config string
                    method_str = row["method"]
                    hierarchical_str = " H" if hierarchical_value == 1 else ""
                    str_config_legend = f"{method_str} K={sp_fac_palm} Q={nb_fac_param}{hierarchical_str}"
                    str_config_hover = f"{method_str} K={sp_fac_palm} Q={nb_fac_param}{hierarchical_str}"
                    dct_legend_hover[str_config_legend] = str_config_hover

                    # values
                    perf_compressed = row[task_compressed]
                    if task_base == "nb-non-zero-compression-rate":
                        perf_base = 1
                    else:
                        perf_base = row[task_base]


                    dct_results_table = {
                        "layer_name": row["layer-name-base"],
                        "compression_rate": row["nb-non-zero-compression-rate"]

                    }

                    layer_name = row["layer-name-base"]
                    idx_layer = row["idx-layer"]
                    dct_idx_layer[idx_layer] = layer_name

                    dct_config_dct_layer_tpl_comp_nfac[str_config_legend][layer_name] = perf_compressed
                    if layer_name in dct_config_dct_layer_tpl_comp_nfac["Base"] and task_compressed != "diff-approx":
                        assert perf_base == dct_config_dct_layer_tpl_comp_nfac["Base"][layer_name]
                    dct_config_dct_layer_tpl_comp_nfac["Base"][layer_name] = perf_base
                    # dct_config_dct_layer_tpl_comp_nfac[str_config_recons_legend][layer_name] = nnz_reconstructed

                if task_compressed != "diff-approx":
                    for idx_row, (_, row) in enumerate(df_tucker_tt_model.iterrows()):
                        # config attributes
                        idx_xp = str(row["idx-expe"])
                        rank_value = str(row["rank-value"])
                        order = str(row["order"])
                        rank_percent = str(row["rank-percentage-dense"])
                        compression = str(row["compression"])

                        # config string
                        if compression == "tucker" and rank_percent == "None":
                            if rank_percent != "None":
                                str_config_legend = f"Tucker + Low Rank sv={rank_percent}%%"
                            else:
                                str_config_legend = f"Tucker"
                            str_config_hover = str_config_legend
                        elif compression == "tucker":
                            continue
                        else:
                            str_config_legend = f"TT K={order} R={rank_value}"
                            str_config_hover = str_config_legend
                        dct_legend_hover[str_config_legend] = str_config_hover


                        # comrpession-rate
                        perf_compressed = row[task_compressed]
                        if task_base == "nb-non-zero-compression-rate":
                            perf_base = 1
                        else:
                            perf_base = row[task_base]
                        # compression_rate = row["nb-non-zero-compression-rate"]

                        if "layer-name-base" in row:
                            layer_name = row["layer-name-base"].split("_-_")[0]
                        else:
                            layer_name = row["layer-name-compressed"].split("_-_")[0]
                        idx_layer = row["idx-layer"]
                        if idx_layer in dct_idx_layer:
                            assert dct_idx_layer[idx_layer] == layer_name
                        dct_idx_layer[idx_layer] = layer_name

                        dct_config_dct_layer_tpl_comp_nfac[str_config_legend][layer_name] = perf_compressed
                        if layer_name in dct_config_dct_layer_tpl_comp_nfac["Base"]:
                            assert perf_base == dct_config_dct_layer_tpl_comp_nfac["Base"][layer_name]

                        row_color_code = "hsl({}, {}%, 50%)".format(random.randint(0, 100), random.randint(20, 80))

                        # dct_config_color[str_config_hover] = row_color_code

                fig = go.Figure()

                if task_compressed != "diff-approx":
                    sorted_configs = sorted(dct_legend_hover.keys()) + ["Base"]
                else:
                    sorted_configs = sorted(dct_legend_hover.keys())

                dct_name_trace = {
                    "PYQALM K=14 Q=3": "Sparse Facto. K=14 Q=3",
                    "PYQALM K=2 Q=3": "Sparse Facto. K=2 Q=3"
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
                        ticktexts[i_elm] = f"C{count_conv}"
                        count_conv += 1
                    elif "Dense" in elm:
                        ticktexts[i_elm] = f"D{count_dense}"
                        count_dense += 1
                    else:
                        ticktexts[i_elm] = f"S"


                x_legend = 0
                y_legend = -0.8
                fig.update_layout(
                    barmode='group',
                    # title=title,
                    # xaxis_title="Layer name",
                    # yaxis_title="Error",
                    # yaxis_type="log",
                    xaxis={'type': 'category',
                           'tickvals': tickvals,
                           'ticktext': ticktexts},
                    xaxis_tickangle=-85,
                    # showlegend=True,
                    showlegend=False,
                    autosize=False,
                    margin=dict(l=5, r=5, t=5, b=5),
                    width=350,
                    height=200,
                    font=dict(
                        # family="Courier New, monospace",
                        size=12,
                        color="black"
                    ),
                    legend_orientation="h",
                    # legend=dict(
                    #     x=x_legend, y=y_legend,
                    #     traceorder="normal",
                    #     font=dict(
                    #         family="sans-serif",
                    #         size=18,
                    #         color="black"
                    #     ),
                    #     # bgcolor="LightSteelBlue",
                    #     # bordercolor="Black",
                    #     borderwidth=1,
                    # )
                )
                # fig.show()
                fig.update_xaxes(showline=True, ticks="outside", linewidth=2, linecolor='black', mirror=True)
                fig.update_yaxes(showline=True, ticks="outside", linewidth=2, linecolor='black', mirror=True)
                fig.write_image(str((output_dir / title).absolute()) + ".png")
