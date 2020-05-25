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

color_bars_sparsity = {
    2: "g",
    3: "c",
    4: "b",
    5: "y"
}

tasks = {
    ("nb-non-zero-compressed", "nb-non-zero-base"),
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

if __name__ == "__main__":
    root_source_dir = pathlib.Path("/home/luc/PycharmProjects/palmnet/results/processed")

    # results_path = "2020/03/9_10_finetune_palminized_no_useless"
    results_path_faust = "2020/05/7_8_finetune_sparse_facto_faust_only_mnist_cifar10"
    results_path_pyqalm = "2020/04/9_10_finetune_palminized_no_useless"
    results_path_tucker_tt = "2020/04/0_0_compression_tucker_tensortrain"

    # src_results_path = root_source_dir / results_path / "results_layers.csv"
    src_results_path_faust = root_source_dir / results_path_faust / "results_layers.csv"
    src_results_pyqalm = root_source_dir / results_path_pyqalm / "results_layers.csv"
    src_results_path_tucker_tt = root_source_dir / results_path_tucker_tt / "results_layers.csv"

    root_output_dir = pathlib.Path("/home/luc/PycharmProjects/palmnet/reports/figures/")
    output_dir = root_output_dir / results_path_pyqalm / "histo_compressions"
    output_dir.mkdir(parents=True, exist_ok=True)

    df_tucker_tt = pd.read_csv(src_results_path_tucker_tt, header=0)
    df_tucker_tt = df_tucker_tt.fillna("None")

    df = pd.read_csv(src_results_pyqalm, header=0)
    df = df.fillna("None")
    df = df[df["hierarchical"] == False]
    df = df[df["keep-last-layer"] == 0]
    df = df.assign(method=["PYQALM"]  * len(df))
    df = df.assign(method_id=[0] * len(df))

    df_faust = pd.read_csv(src_results_path_faust, header=0)
    df_faust = df_faust.fillna("None")
    df_faust = df_faust[df_faust["hierarchical"] == False]
    df_faust = df_faust[df_faust["keep-last-layer"] == 0]
    df_faust = df_faust.assign(method=["FAUST"] * len(df_faust))
    df_faust = df_faust.assign(method_id=[1] * len(df_faust))

    df = pd.concat([df, df_faust])

    hue_by_sparsity= {
        2: 10,
        3: 20,
        4: 30,
        5: 40
    }
    hue_by_factor= {
        2: 50,
        3: 60,
        "None": 70
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
        0: 80
    }


    from sklearn import manifold
    from sklearn.metrics import euclidean_distances

    sp_fac = df["sparsity-factor"].values
    # nb_fac = df["nb-factor-actual"].values
    hierar = df["hierarchical"].values.astype(float)
    # config_vectors = np.vstack([sp_fac, nb_fac, hierar]).T
    method_id = df["method_id"].values
    config_vectors = np.vstack([sp_fac, hierar, method_id]).T

    config_vectors -= config_vectors.mean()
    similarities = euclidean_distances(config_vectors)
    mds = manifold.MDS(n_components=2, metric=True, max_iter=100, eps=1e-9, random_state=0,
                       dissimilarity="precomputed", n_jobs=1)
    pos = mds.fit(similarities).embedding_
    min_pos = np.min(pos, axis=0)
    max_pos = np.max(pos, axis=0)
    pos = (((pos - min_pos) / (max_pos - min_pos)))
    lst_color_codes = ["{}".format(tuple(lin)) for lin in pos]
    df = df.assign(color_code=lst_color_codes)


    del lst_color_codes
    datasets = set(df["data"].values)
    for tpl_task in tasks:
        task_compressed = tpl_task[0]
        task_base = tpl_task[1]
        for dataname in datasets:
            df_data = df[df["data"] == dataname]
            df_tucker_tt_data = df_tucker_tt[df_tucker_tt["data"] == dataname]
            df_model_values = set(df_data["model"].values)
            for modelname in df_model_values:
                dct_config_color = dict()
                dct_config_dct_layer_tpl_comp_nfac = defaultdict(lambda: dict())
                dct_legend_hover = dict()

                df_model = df_data[df_data["model"] == modelname]
                df_tucker_tt_model = df_tucker_tt_data[df_tucker_tt_data["model"] == modelname]

                dct_idx_layer = dict()

                for idx_row, (_, row) in enumerate(df_model.iterrows()):
                    # config attributes
                    idx_xp = str(row["idx-expe"])
                    sp_fac_palm = int(row["sparsity-factor"])
                    hierarchical_value = row["hierarchical"]
                    nb_fac_param = row["nb-factor-param"]
                    # nb_fac = int(row["nb-factor-actual"])

                    # config string
                    method_str = row["method"]
                    hierarchical_str = "-H" if hierarchical_value == 1 else ""
                    str_config_legend = f"{method_str} sp{sp_fac_palm}-nb{nb_fac_param}-{hierarchical_str}"
                    # str_config_recons_legend = "Reconstructed " + str_config_legend
                    str_config_hover = f"{method_str} sp{sp_fac_palm}-nb{nb_fac_param}-{hierarchical_str}"
                    # str_config_recons_hover = "Reconstructed " + str_config_hover
                    dct_legend_hover[str_config_legend] = str_config_hover
                    # dct_legend_hover[str_config_recons_legend] = str_config_recons_hover

                    # comrpession-rate

                    perf_compressed = row[task_compressed]
                    perf_base = row[task_base]
                    # nnz_reconstructed = row["nb-non-zero-reconstructed"]
                    # compression_rate = row["nb-non-zero-compression-rate"]

                    layer_name = row["layer-name-base"]
                    idx_layer = row["idx-layer"]
                    dct_idx_layer[idx_layer] = layer_name

                    dct_config_dct_layer_tpl_comp_nfac[str_config_legend][layer_name] = perf_compressed
                    if layer_name in dct_config_dct_layer_tpl_comp_nfac["Base"] and task_compressed != "diff-approx":
                        assert perf_base == dct_config_dct_layer_tpl_comp_nfac["Base"][layer_name]
                    dct_config_dct_layer_tpl_comp_nfac["Base"][layer_name] = perf_base
                    # dct_config_dct_layer_tpl_comp_nfac[str_config_recons_legend][layer_name] = nnz_reconstructed

                    # hls_str = "hsl({}, {}%, {}%)".format(hue_by_sparsity[sp_fac_palm] + hue_by_factor[nb_fac], saturation_by_hier[hierarchical_value], lum_by_clr[clr_value])

                    tpl_row_color_code = tuple(eval(row["color_code"]))
                    # row_color_code = row["color_code"]
                    hue = int(tpl_row_color_code[0] * 255)
                    sat = int(tpl_row_color_code[1] * 100)
                    row_color_code = "hsl({}, {}%, 50%)".format(hue, sat)
                    # row_color_code_recons = "hsl({}, {}%, 70%)".format(hue, sat)

                    dct_config_color[str_config_hover] = row_color_code
                    # dct_config_color[str_config_recons_hover] = row_color_code_recons

                    # fig.add_trace(go.Bar(name=str_config,
                    #            x=[idx_layer], y=[nnz_compressed],
                    #            marker_color=color_code,
                    #                      hovertext=str_config
                    # ))
                    #
                    # fig.add_trace(go.Bar(name=str_config_recons,
                    #            x=[idx_layer], y=[nnz_reconstructed],
                    #            marker_color=color_code,
                    #                      hovertext=str_config_recons
                    # ))

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
                            str_config_legend = f"Tucker %% sv: {rank_percent}"
                            str_config_hover = str_config_legend
                        elif compression == "tucker":
                            continue
                        else:
                            str_config_legend = f"TT order: {order}; rank: {rank_value}"
                            str_config_hover = str_config_legend
                        dct_legend_hover[str_config_legend] = str_config_hover


                        # comrpession-rate
                        perf_compressed = row[task_compressed]
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

                        dct_config_color[str_config_hover] = row_color_code

                fig = go.Figure()

                if task_compressed != "diff-approx":
                    sorted_configs = sorted(dct_legend_hover.keys()) + ["Base"]
                else:
                    sorted_configs = sorted(dct_legend_hover.keys())

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

                    fig.add_trace(go.Bar(name=s_conf,
                                         x=lst_layers, y=lst_configs,
                                         marker_color=dct_config_color[hover_text],
                                         hovertext=hover_text,
                                         showlegend=True
                                         ))

                # add_legend = True
                # for idx_layer, layer_name in dct_idx_layer.items():
                #
                #     nnz_base = dct_config_dct_layer_tpl_comp_nfac["Base"][layer_name]
                #     fig.add_trace(go.Bar(name="Base",
                #                          x=[layer_name], y=[nnz_base],
                #                          marker_color="red",
                #                          showlegend=add_legend
                #     ))
                #     add_legend = False


                title = "_".join(str(x) for x in [dataname, modelname, task_compressed])

                tickvals = [dct_idx_layer[idx] for idx in sorted(dct_idx_layer.keys())]
                fig.update_layout(
                    barmode='group',
                                  title=title,
                                  xaxis_title="Layer index",
                                  yaxis_title="NNZ",
                                  yaxis_type="log",
                    # bargap=0.1,
                                  xaxis={'type': 'category',
                                         'tickvals': tickvals},
                                  xaxis_tickangle=-45
                )
                fig.show()
                fig.write_image(str((output_dir / title).absolute()) + ".png")
                # exit()
