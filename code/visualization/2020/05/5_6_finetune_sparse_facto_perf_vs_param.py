import pathlib
import pandas as pd
from palmnet.visualization.utils import get_palminized_model_and_df, get_df
import matplotlib.pyplot as plt
import numpy as np
import logging
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


def get_palm_results():
    results_path = "2020/03/9_10_finetune_palminized_no_useless"
    results_path_2 = "2020/04/9_10_finetune_palminized_no_useless"

    src_results_path = root_source_dir / results_path / "results.csv"
    src_results_path_2 = root_source_dir / results_path_2 / "results.csv"

    df = pd.read_csv(src_results_path, header=0)
    df_2 = pd.read_csv(src_results_path_2, header=0)
    df = pd.concat([df, df_2])
    df = df.fillna("None")
    df = df.drop(columns=["Unnamed: 0", "idx-expe"]).drop_duplicates()

    df = df[df["keep-last-layer"] == 0]
    df = df[df["use-clr"] == 1]

    df = df.assign(**{"only-dense": False, "keep-first-layer": False})

    return df

def get_faust_results():
    results_path = "2020/05/3_4_finetune_faust_no_hierarchical_only_cifar_mnist"

    src_results_path = root_source_dir / results_path / "results.csv"

    df = pd.read_csv(src_results_path, header=0)
    df = df.fillna("None")
    df = df[df["hierarchical"] == False]
    df = df.drop(columns=["Unnamed: 0", "idx-expe"]).drop_duplicates()

    df = df[df["keep-last-layer"] == 0]
    df = df.assign(**{"only-dense": False, "keep-first-layer": False})
    return df

def get_tucker_results():
    results_path_tucker = "2020/04/0_1_compression_tucker_tensortrain"
    src_results_path_tucker = root_source_dir / results_path_tucker / "results.csv"

    df_tucker_tt = pd.read_csv(src_results_path_tucker, header=0)
    df_tucker_tt = df_tucker_tt.fillna("None")

    df_tucker_tt = df_tucker_tt.assign(**{"only-dense": False, "use-pretrained": False})

    df_tucker_tt = df_tucker_tt[df_tucker_tt["compression"] == "tucker"]
    return df_tucker_tt

def get_tensortrain_results():
    results_path_tucker = "2020/05/2_3_compression_tensortrain"
    src_results_path_tucker = root_source_dir / results_path_tucker / "results.csv"

    df_tucker_tt = pd.read_csv(src_results_path_tucker, header=0)
    df_tucker_tt = df_tucker_tt.fillna("None")

    # df_tucker_tt = df_tucker_tt[df_tucker_tt["use-pretrained"] == True]
    # df_tucker_tt = df_tucker_tt[df_tucker_tt["only-dense"] == False]

    return df_tucker_tt

def get_tucker_tensortrain_only_denseresults():
    results_path_tucker = "2020/05/2_3_compression_tucker_tensortrain_only_dense"
    src_results_path_tucker = root_source_dir / results_path_tucker / "results.csv"

    df_tucker_tt = pd.read_csv(src_results_path_tucker, header=0)
    df_tucker_tt = df_tucker_tt.fillna("None")

    # df_tucker_tt = df_tucker_tt[df_tucker_tt["use-pretrained"] == True]
    # df_tucker_tt = df_tucker_tt[df_tucker_tt["only-dense"] == True]

    return df_tucker_tt

def get_palm_results_only_dense_keep_first():
    results_path = "2020/05/5_6_finetune_sparse_facto_no_hierarchical_keep_first_layer_only_dense"

    src_results_path = root_source_dir / results_path / "results.csv"

    df = pd.read_csv(src_results_path, header=0)
    df = df.fillna("None")
    df = df.drop(columns=["Unnamed: 0", "idx-expe"]).drop_duplicates()

    # df = df[df["only-dense"] == False]

    return df

if __name__ == "__main__":
    root_source_dir = pathlib.Path("/home/luc/PycharmProjects/palmnet/results/processed")


    SHOW_FAUST = False
    SHOW_KEEP_FIRST_ONLY = True
    SHOW_PRETRAINED_ONLY = True

    results_path = "2020/05/5_6_finetune_sparse_facto_perf_vs_param"

    df_faust = get_faust_results()

    df_tucker = get_tucker_results()
    df_tt = get_tensortrain_results()
    df_tucker_tt_only_dense = get_tucker_tensortrain_only_denseresults()
    df_tucker_tt = pd.concat([df_tucker, df_tt, df_tucker_tt_only_dense])

    df_palm = get_palm_results()
    df_palm_bis = get_palm_results_only_dense_keep_first()
    df_palm = pd.concat([df_palm, df_palm_bis])

    ONLY_DENSE = False
    df_tucker_tt = df_tucker_tt[df_tucker_tt["only-dense"] == ONLY_DENSE]
    df_palm = df_palm[df_palm["only-dense"] == ONLY_DENSE]


    root_output_dir = pathlib.Path("/home/luc/PycharmProjects/palmnet/reports/figures/")
    output_dir = root_output_dir / results_path / "histogrammes"
    output_dir.mkdir(parents=True, exist_ok=True)

    # sparsity_factors = sorted(set(df_palminized["--sparsity-factor"]))
    nb_factors = set(df_faust["nb-factor"].values)

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

    datasets = set(df_palm["dataset"].values)
    for dataname in datasets:
        df_data_faust = df_faust[df_faust["dataset"] == dataname]
        df_data_palm = df_palm[df_palm["dataset"] == dataname]
        df_tucker_tt_data =  df_tucker_tt[df_tucker_tt["dataset"] == dataname]
        df_model_values = set(df_data_palm["model"].values)

        for modelname in df_model_values:
            df_model_faust = df_data_faust[df_data_faust["model"] == modelname]
            df_model_palm = df_data_palm[df_data_palm["model"] == modelname]
            df_tucker_tt_model = df_tucker_tt_data[df_tucker_tt_data["model"] == modelname]

            fig = go.Figure()

            base_score = None
            base_nb_param = None

            for idx_pal_algo, df_model in enumerate([df_model_faust, df_model_palm]):
                if idx_pal_algo == 0:
                    palm_algo = "FAUST"
                    if not SHOW_FAUST:
                        continue
                else:
                    palm_algo = "PYQALM"

                for idx_row, row in df_model.iterrows():
                    hierarchical_value = row["hierarchical"]
                    str_hierarchical = ' H' if hierarchical_value is True else ''
                    try:
                        nb_factor = int(row["nb-factor"])
                    except:
                        nb_factor = None
                    sparsity_factor = int(row["sparsity-factor"])

                    keep_first = row["keep-first-layer"]
                    str_keep_first = ' -1' if keep_first is True else ''
                    if SHOW_KEEP_FIRST_ONLY and not keep_first:
                        continue

                    only_mask = row["only-mask"]
                    str_only_mask = " M" if only_mask is True else ""

                    name_trace = f"{palm_algo} Q={nb_factor} K={sparsity_factor}{str_hierarchical}{str_keep_first}{str_only_mask}"

                    finetuned_score = row["finetuned-score"]
                    nb_param = row["nb-param-compressed-total"]

                    base_score_tmp = row["base-model-score"]
                    assert base_score == base_score_tmp or base_score is None
                    base_nb_param_tmp = row["nb-param-base-total"]
                    assert base_nb_param == base_nb_param_tmp or base_nb_param is None
                    base_score = base_score_tmp
                    base_nb_param = base_nb_param_tmp

                    fig.add_trace(
                        go.Scatter(
                            x=[nb_param],
                            y=[finetuned_score],
                            mode='markers',
                            name=name_trace,
                            hovertext=name_trace,
                            legendgroup=f"{palm_algo} K={sparsity_factor}{str_only_mask}",
                            marker=dict(
                                        color=dct_colors[f"PALM K={sparsity_factor}"],
                                        symbol=dct_symbol[f"{palm_algo} Q={nb_factor}{str_hierarchical}{str_keep_first}{str_only_mask}"],
                                        size=SIZE_MARKERS,
                                        line=dict(
                                            color='Black',
                                            width=WIDTH_MARKER_LINES
                                        )
                                    )

                        ))

            #############
            # base data #
            #############
            fig.add_trace(
                go.Scatter(
                    x=[base_nb_param],
                    y=[base_score],
                    mode='markers',
                    name="Base",
                    hovertext="Base",
                    legendgroup=f"Base",
                    marker=dict(
                        color=dct_colors[f"Base"],
                        symbol=dct_symbol[f"Base"],
                        size=SIZE_MARKERS,
                        line=dict(
                            color='Black',
                            width=WIDTH_MARKER_LINES,
                        )
                    )

                ))

            ###############
            # tucker data #
            ###############
            df_tucker = df_tucker_tt_model[df_tucker_tt_model["compression"] == "tucker"]
            for idx_row, row in df_tucker.iterrows():
                keep_first = row["keep-first-layer"]
                str_keep_first = ' -1' if keep_first is True else ''
                if SHOW_KEEP_FIRST_ONLY and not keep_first:
                    continue
                try:
                    rank_percentage = int(float(row["rank-percentage-dense"]) * 100)
                except:
                    rank_percentage = None

                str_percentage = f' + Low Rank {rank_percentage}%' if rank_percentage is not None else ''

                name_trace = f"Tucker{str_keep_first}{str_percentage}"

                finetuned_score = row["finetuned-score"]
                nb_param = row["nb-param-compressed-total"]

                base_score_tmp = row["base-model-score"]
                assert base_score == base_score_tmp or base_score is None
                base_nb_param_tmp = row["nb-param-base-total"]
                assert base_nb_param == base_nb_param_tmp or base_nb_param is None

                fig.add_trace(
                    go.Scatter(
                        x=[nb_param],
                        y=[finetuned_score],
                        mode='markers',
                        name=name_trace,
                        hovertext=name_trace,
                        legendgroup=f"Tucker{str_percentage}",
                        marker=dict(
                                    color=dct_colors[f"Tucker{str_percentage}"],
                                    symbol=dct_symbol[f"Tucker{str_keep_first}"],
                                    size=SIZE_MARKERS,
                                    line=dict(
                                        color='Black',
                                        width=WIDTH_MARKER_LINES
                                    )
                                )

                    ))

            ####################
            # tensortrain data #
            ####################
            df_tt = df_tucker_tt_model[df_tucker_tt_model["compression"] == "tensortrain"]
            for idx_row, row in df_tt.iterrows():
                keep_first = row["keep-first-layer"]
                str_keep_first = ' -1' if keep_first is True else ''
                if SHOW_KEEP_FIRST_ONLY and not keep_first:
                    continue
                order = int(row["order"])
                rank_value = int(row["rank-value"])

                if not np.isnan(row["use-pretrained"]):
                    use_petrained = bool(row["use-pretrained"])
                    str_pretrained = " pretrained" if use_petrained else ""
                else:
                    use_petrained = False
                    str_pretrained = ""

                if SHOW_PRETRAINED_ONLY and not use_petrained:
                    continue
                name_trace = f"Tensortrain{str_keep_first} K={order} R={rank_value}{str_pretrained}"

                finetuned_score = row["finetuned-score"]
                nb_param = row["nb-param-compressed-total"]

                base_score_tmp = row["base-model-score"]
                assert base_score == base_score_tmp or base_score is None
                base_nb_param_tmp = row["nb-param-base-total"]
                assert base_nb_param == base_nb_param_tmp or base_nb_param is None

                fig.add_trace(
                    go.Scatter(
                        x=[nb_param],
                        y=[finetuned_score],
                        mode='markers',
                        name=name_trace,
                        hovertext=name_trace,
                        legendgroup=f"TT R={rank_value}",
                        marker=dict(
                                    color=dct_colors[f"TT R={rank_value}"],
                                    symbol=dct_symbol[f"TT{str_keep_first}{str_pretrained}"],
                                    size=SIZE_MARKERS,
                                    line=dict(
                                        color='Black',
                                        width=WIDTH_MARKER_LINES
                                    )
                                )

                    ))

            title = "Performance = f(# Param); " + dataname + " " + modelname

            fig.update_layout(title=title,
                              xaxis_title="# Parameter in Dense and Conv Layers",
                              yaxis_title="Accuracy (%)",
                              xaxis_type="log",
                              )
            fig.show()
            fig.write_image(str((output_dir / title).absolute()) + ".png")
