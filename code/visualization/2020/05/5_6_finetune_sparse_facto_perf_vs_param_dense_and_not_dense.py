import pathlib
import pandas as pd
from palmnet.visualization.utils import get_palminized_model_and_df, get_df
import matplotlib.pyplot as plt
import numpy as np
import logging
import plotly.graph_objects as go
import plotly.io as pio
from pprint import pprint as pprint

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

def get_deepfried_results():
    results_path_tucker = "2020/05/5_6_compression_baselines"
    src_results_path_tucker = root_source_dir / results_path_tucker / "results.csv"

    df_tucker_tt = pd.read_csv(src_results_path_tucker, header=0)
    df_tucker_tt = df_tucker_tt.fillna("None")

    # df_tucker_tt = df_tucker_tt.assign(**{"only-dense": True, "use-pretrained": False})

    df_tucker_tt = df_tucker_tt[df_tucker_tt["compression"] == "deepfried"]
    return df_tucker_tt

def get_magnitude_results():
    results_path_tucker = "2020/05/5_6_compression_baselines"
    src_results_path_tucker = root_source_dir / results_path_tucker / "results.csv"

    df_tucker_tt = pd.read_csv(src_results_path_tucker, header=0)
    df_tucker_tt = df_tucker_tt.fillna("None")

    # df_tucker_tt = df_tucker_tt.assign(**{"only-dense": True, "use-pretrained": False})

    df_tucker_tt = df_tucker_tt[df_tucker_tt["compression"] == "magnitude"]
    return df_tucker_tt

if __name__ == "__main__":
    root_source_dir = pathlib.Path("/home/luc/PycharmProjects/palmnet/results/processed")


    SHOW_FAUST = False
    SHOW_KEEP_FIRST_ONLY = True
    SHOW_PRETRAINED_ONLY = True

    results_path = "2020/05/5_6_finetune_sparse_facto_perf_vs_param"

    df_tucker = get_tucker_results()
    df_tt = get_tensortrain_results()
    df_deepfried = get_deepfried_results()
    df_tucker_tt_only_dense = get_tucker_tensortrain_only_denseresults()
    df_magnitude = get_magnitude_results()
    df_tucker_tt_deepfried = pd.concat([df_tucker, df_tt, df_tucker_tt_only_dense, df_deepfried, df_magnitude])

    df_palm = get_palm_results()
    df_palm_bis = get_palm_results_only_dense_keep_first()
    df_palm = pd.concat([df_palm, df_palm_bis])

    # ONLY_DENSE = False
    # df_tucker_tt = df_tucker_tt[df_tucker_tt["only-dense"] == ONLY_DENSE]
    # df_palm = df_palm[df_palm["only-dense"] == ONLY_DENSE]

    root_output_dir = pathlib.Path("/home/luc/PycharmProjects/palmnet/reports/figures/")
    output_dir = root_output_dir / results_path / "histogrammes"
    output_dir.mkdir(parents=True, exist_ok=True)

    # sparsity_factors = sorted(set(df_palminized["--sparsity-factor"]))
    # nb_factors = set(df_palm["nb-factor"].values)

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
        "Deepfried": "hexagram",
        "Magnitude ": "square",
        "Magnitude  -1": "square",
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
        "Tucker + Low Rank 30%": "red",
        "Deepfried": "blueviolet",
        "Magnitude  50%": "red",
        "Magnitude  70%": "red",
        "Magnitude  90%": "red",
    }

    SIZE_MARKERS = 15
    WIDTH_MARKER_LINES = 2

    datasets = set(df_palm["dataset"].values)
    dct_table = dict()
    for dataname in datasets:
        dct_table[dataname] = dict()
        df_data_palm = df_palm[df_palm["dataset"] == dataname]
        df_tucker_tt_data = df_tucker_tt_deepfried[df_tucker_tt_deepfried["dataset"] == dataname]
        df_model_values = set(df_data_palm["model"].values)

        for modelname in df_model_values:
            dct_table[dataname][modelname] = dict()
            df_model_palm = df_data_palm[df_data_palm["model"] == modelname]
            df_tucker_tt_model = df_tucker_tt_data[df_tucker_tt_data["model"] == modelname]

            for ONLY_DENSE in [True, False]:
                df_tucker_tt_model_dense = df_tucker_tt_model[df_tucker_tt_model["only-dense"] == ONLY_DENSE]
                df_model_palm_dense = df_model_palm[df_model_palm["only-dense"] == ONLY_DENSE]

                if ONLY_DENSE:
                    str_nb_param_compressed = "nb-param-compressed-dense"
                    str_nb_param_base = "nb-param-base-dense"
                    str_only_dense = " only dense"
                else:
                    str_nb_param_compressed = "nb-param-compressed-total"
                    str_nb_param_base = "nb-param-base-total"
                    str_only_dense = ""

                dct_entry_only_dense = "Dense" if ONLY_DENSE else "Conv+Dense"
                dct_table[dataname][modelname][dct_entry_only_dense] = list()
                fig = go.Figure()

                base_score = None
                base_nb_param = None

                palm_algo = "PYQALM"
                for idx_row, row in df_model_palm_dense.iterrows():
                    hierarchical_value = row["hierarchical"]
                    str_hierarchical = ' H' if hierarchical_value is True else ''
                    try:
                        nb_factor = int(row["nb-factor"])
                    except:
                        nb_factor = None
                    sparsity_factor = int(row["sparsity-factor"])

                    keep_first = row["keep-first-layer"]
                    str_keep_first = ' -1' if keep_first is True else ''
                    if SHOW_KEEP_FIRST_ONLY and not keep_first and not ONLY_DENSE:
                        continue

                    only_mask = row["only-mask"]
                    str_only_mask = " M" if only_mask is True else ""

                    name_trace = f"{palm_algo} Q={nb_factor} K={sparsity_factor}{str_hierarchical}{str_keep_first}{str_only_mask}"
                    finetuned_score = row["finetuned-score"]
                    nb_param = row[str_nb_param_compressed]
                    dct_row = dict()
                    dct_row["method"] = name_trace
                    dct_row["perf"] = finetuned_score
                    dct_row["nb_param"] = nb_param
                    dct_table[dataname][modelname][dct_entry_only_dense].append(dct_row)

                    base_score_tmp = row["base-model-score"]
                    assert base_score == base_score_tmp or base_score is None
                    base_nb_param_tmp = row[str_nb_param_base]
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
                dct_row = dict()
                dct_row["method"] = "Base"
                dct_row["perf"] = base_score
                dct_row["nb_param"] = base_nb_param
                dct_table[dataname][modelname][dct_entry_only_dense].append(dct_row)
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
                df_tucker = df_tucker_tt_model_dense[df_tucker_tt_model_dense["compression"] == "tucker"]
                for idx_row, row in df_tucker.iterrows():
                    keep_first = row["keep-first-layer"]
                    str_keep_first = ' -1' if keep_first is True else ''
                    if SHOW_KEEP_FIRST_ONLY and not keep_first and not ONLY_DENSE:
                        continue
                    try:
                        rank_percentage = int(float(row["rank-percentage-dense"]) * 100)
                    except:
                        try:
                            rank_percentage = int(float(row["rank-percentage"]) * 100)
                        except:
                            rank_percentage = None

                    str_percentage = f' + Low Rank {rank_percentage}%' if rank_percentage is not None else ''

                    name_trace = f"Tucker{str_keep_first}{str_percentage}"

                    finetuned_score = row["finetuned-score"]
                    nb_param = row[str_nb_param_compressed]

                    dct_row = dict()
                    dct_row["method"] = name_trace
                    dct_row["perf"] = finetuned_score
                    dct_row["nb_param"] = nb_param
                    dct_table[dataname][modelname][dct_entry_only_dense].append(dct_row)

                    base_score_tmp = row["base-model-score"]
                    assert base_score == base_score_tmp or base_score is None
                    base_nb_param_tmp = row[str_nb_param_base]
                    assert base_nb_param == base_nb_param_tmp or base_nb_param is None  or base_nb_param_tmp == 0, f"{base_nb_param}!={base_nb_param_tmp}"

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

                ###############
                # magnitude data #
                ###############
                df_magnitude = df_tucker_tt_model_dense[df_tucker_tt_model_dense["compression"] == "magnitude"]
                for idx_row, row in df_magnitude.iterrows():
                    keep_first = row["keep-first-layer"]
                    str_keep_first = ' -1' if keep_first is True else ''
                    if SHOW_KEEP_FIRST_ONLY and not keep_first and not ONLY_DENSE:
                        continue
                    # try:
                    sparsity_percentage = int(float(row["final-sparsity"]) * 100)
                    # except:
                    #     try:
                    #         rank_percentage = int(float(row["rank-percentage"]) * 100)
                    #     except:
                    #         rank_percentage = None

                    str_percentage = f' {sparsity_percentage}%' #if sparsity_percentage is not None else ''

                    name_trace = f"Magnitude {str_keep_first}{str_percentage}"

                    finetuned_score = row["finetuned-score"]
                    nb_param = row[str_nb_param_compressed]

                    dct_row = dict()
                    dct_row["method"] = name_trace
                    dct_row["perf"] = finetuned_score
                    dct_row["nb_param"] = nb_param
                    dct_table[dataname][modelname][dct_entry_only_dense].append(dct_row)
                    print(finetuned_score)

                    base_score_tmp = row["base-model-score"]
                    assert np.isclose(base_score, base_score_tmp) or base_score is None, f"{base_score}!={base_score_tmp}"
                    base_nb_param_tmp = row[str_nb_param_base]
                    assert base_nb_param == base_nb_param_tmp or base_nb_param is None or base_nb_param_tmp == 0, f"{base_nb_param}!={base_nb_param_tmp}"

                    fig.add_trace(
                        go.Scatter(
                            x=[nb_param],
                            y=[finetuned_score],
                            mode='markers',
                            name=name_trace,
                            hovertext=name_trace,
                            legendgroup=f"Magnitude",
                            marker=dict(
                                color=dct_colors[f"Magnitude {str_percentage}"],
                                symbol=dct_symbol[f"Magnitude {str_keep_first}"],
                                size=SIZE_MARKERS,
                                line=dict(
                                    color='Black',
                                    width=WIDTH_MARKER_LINES
                                )
                            )

                        ))


                ###############
                # deepfried data #
                ###############
                df_deepfried = df_tucker_tt_model_dense[df_tucker_tt_model_dense["compression"] == "deepfried"]
                for idx_row, row in df_deepfried.iterrows():
                    keep_first = row["keep-first-layer"]
                    str_keep_first = ' -1' if keep_first is True else ''
                    if SHOW_KEEP_FIRST_ONLY and not keep_first and not ONLY_DENSE:
                        continue
                    # try:
                    # sparsity_percentage = int(float(row["final-sparsity"]) * 100)
                    # except:
                    #     try:
                    #         rank_percentage = int(float(row["rank-percentage"]) * 100)
                    #     except:
                    #         rank_percentage = None

                    # str_percentage = f' {sparsity_percentage}%' #if sparsity_percentage is not None else ''

                    name_trace = f"Deepfried {str_keep_first}"

                    finetuned_score = row["finetuned-score"]
                    nb_param = row[str_nb_param_compressed]
                    if nb_param == 0:
                        conv_nb_weights = row["nb-param-base-total"] - base_nb_param
                        nb_param = row["nb-param-compressed-total"] - conv_nb_weights

                    dct_row = dict()
                    dct_row["method"] = name_trace
                    dct_row["perf"] = finetuned_score
                    dct_row["nb_param"] = nb_param
                    dct_table[dataname][modelname][dct_entry_only_dense].append(dct_row)
                    print(finetuned_score)

                    base_score_tmp = row["base-model-score"]
                    assert np.isclose(base_score, base_score_tmp) or base_score is None, f"{base_score}!={base_score_tmp}"
                    base_nb_param_tmp = row[str_nb_param_base]
                    assert base_nb_param == base_nb_param_tmp or base_nb_param is None or base_nb_param_tmp == 0, f"{base_nb_param}!={base_nb_param_tmp}"

                    fig.add_trace(
                        go.Scatter(
                            x=[nb_param],
                            y=[finetuned_score],
                            mode='markers',
                            name=name_trace,
                            hovertext=name_trace,
                            legendgroup=f"Deepfried",
                            marker=dict(
                                color=dct_colors[f"Deepfried"],
                                symbol=dct_symbol[f"Deepfried"],
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
                df_tt = df_tucker_tt_model_dense[df_tucker_tt_model_dense["compression"] == "tensortrain"]
                for idx_row, row in df_tt.iterrows():
                    keep_first = row["keep-first-layer"]
                    str_keep_first = ' -1' if keep_first is True else ''
                    if SHOW_KEEP_FIRST_ONLY and not keep_first and not ONLY_DENSE:
                        continue
                    order = int(row["order"])
                    rank_value = int(row["rank-value"])

                    if not np.isnan(row["use-pretrained"]):
                        use_petrained = bool(row["use-pretrained"])
                        str_pretrained = " pretrained" if use_petrained else ""
                    else:
                        use_petrained = False
                        str_pretrained = ""

                    if SHOW_PRETRAINED_ONLY and not use_petrained and not ONLY_DENSE:
                        continue
                    name_trace = f"Tensortrain{str_keep_first} K={order} R={rank_value}{str_pretrained}"

                    finetuned_score = row["finetuned-score"]
                    nb_param = row[str_nb_param_compressed]

                    dct_row = dict()
                    dct_row["method"] = name_trace
                    dct_row["perf"] = finetuned_score
                    dct_row["nb_param"] = nb_param
                    dct_table[dataname][modelname][dct_entry_only_dense].append(dct_row)

                    base_score_tmp = row["base-model-score"]
                    assert base_score == base_score_tmp or base_score is None
                    base_nb_param_tmp = row[str_nb_param_base]
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

                title = "Performance = f(# Param); " + dataname + " " + modelname + str_only_dense

                fig.update_layout(title=title,
                                  xaxis_title="# Parameter in Dense and Conv Layers",
                                  yaxis_title="Accuracy (%)",
                                  xaxis_type="log",
                                  )
                fig.show()
                fig.write_image(str((output_dir / title).absolute()) + ".png")

    pprint(dct_table)
#     string_table = """
# \begin{tabular}{lcccccccccccccccccccccc}
#     \toprule
#
# {}                        &  \multicolumn{2}{c}{ \thead{ Ensemble } }  &  \multicolumn{2}{c}{ \thead{ Kmeans } }  &  \multicolumn{2}{c}{ \thead{ NN-OMP\\w/o weights } }  &  \multicolumn{2}{c}{ \thead{ NN-OMP } }  &  \multicolumn{2}{c}{ \thead{ OMP\\w/o weights } }  &  \multicolumn{2}{c}{ \thead{ OMP } }  &  \multicolumn{2}{c}{ \thead{ Random } }  &  \multicolumn{2}{c}{ \thead{ Zhang\\Predictions } }  &  \multicolumn{2}{c}{ \thead{ Zhang\\Similarities } }\\
# \midrule
# Diam.                     &  3.032E+05 & 86            &  \underline{3.024E+05} & \underline{143} &  \textbf{3.024E+05} & \textbf{86} &  3.033E+05 & 86            &  3.025E+05 & 143           &  \textit{3.087E+05} & \textit{29} &  3.025E+05 & 114           &  3.047E+05 & 143           &  3.032E+05 & 143\\
# Diab.                     &  3.431E+03 & 32            &  \underline{3.281E+03} & \underline{36} &  3.317E+03 & 36            &  3.549E+03 & 36            &  3.324E+03 & 36            &  \textit{3.607E+03} & \textit{25} &  3.303E+03 & 32            &  3.282E+03 & 36            &  \textbf{3.241E+03} & \textbf{32}\\
# Kin.                      &  1.892E-02 & 200           &  \textit{2.024E-02} & \textit{33} &  1.921E-02 & 133           &  \underline{1.809E-02} & \underline{133} &  1.931E-02 & 67            &  \textbf{1.776E-02} & \textbf{333} &  2.002E-02 & 333           &  2.089E-02 & 333           &  2.017E-02 & 333\\
# C. H.                     &  \underline{2.187E-01} & \underline{267} &  \textit{2.449E-01} & \textit{33} &  2.239E-01 & 100           &  \textbf{2.180E-01} & \textbf{133} &  \textit{2.267E-01} & \textit{33} &  2.197E-01 & 133           &  2.390E-01 & 333           &  2.536E-01 & 333           &  2.452E-01 & 333\\
# Bos.                      &  1.267E+01 & 30            &  \textit{1.278E+01} & \textit{13} &  \textbf{1.214E+01} & \textbf{33} &  1.253E+01 & 33            &  \underline{1.247E+01} & \underline{27} &  \textit{1.293E+01} & \textit{13} &  1.253E+01 & 33            &  1.430E+01 & 33            &  1.283E+01 & 33\\
# \midrule
# Sp. B.                    &  94.27\% & 133             &  95.52\% & 167             &  \textit{95.57\%} & \textit{100} &  \underline{\textit{95.59\%}} & \underline{\textit{100}} &  95.56\% & 167             &  95.39\% & 133             &  \textbf{95.59\%} & \textbf{167} &  95.45\% & 333             &  95.46\% & 167\\
# St. P.                    &  98.69\% & 233             &  99.05\% & 267             &  \underline{\textit{99.95\%}} & \underline{\textit{67}} &  \textbf{99.95\%} & \textbf{100} &  \textit{99.64\%} & \textit{67} &  99.90\% & 333             &  \textit{99.41\%} & \textit{67} &  99.43\% & 167             &  98.92\% & 300\\
# KR-KP                     &  \textit{98.22\%} & \textit{33} &  99.00\% & 333             &  \underline{99.42\%} & \underline{100} &  99.39\% & 100             &  99.22\% & 100             &  \textbf{99.48\%} & \textbf{100} &  99.14\% & 267             &  99.14\% & 133             &  98.94\% & 333\\
# B. C.                     &  95.09\% & 100             &  \textbf{\textit{96.58\%}} & \textbf{\textit{33}} &  \underline{96.49\%} & \underline{67} &  \textbf{96.58\%} & \textbf{67} &  95.79\% & 133             &  95.35\% & 67              &  95.88\% & 300             &  \textit{95.70\%} & \textit{33} &  95.61\% & 333\\
# LFW P.                    &  \textit{56.00\%} & \textit{67} &  65.25\% & 333             &  \textbf{66.02\%} & \textbf{333} &  65.73\% & 233             &  65.32\% & 133             &  65.55\% & 167             &  \underline{65.98\%} & \underline{267} &  65.43\% & 333             &  65.27\% & 333\\
# Gam.                      &  \textit{80.78\%} & \textit{3} &  87.68\% & 33              &  \underline{87.75\%} & \underline{33} &  \underline{87.75\%} & \underline{33} &  \underline{87.75\%} & \underline{33} &  \underline{87.75\%} & \underline{33} &  \textbf{87.76\%} & \textbf{33} &  87.72\% & 33              &  87.68\% & 33\\
#
# \bottomrule
# \end{tabular}
# """

    tab_headers = [
        "Dataset",
        "Architecture",
        "Compressed layers",
        "Method",
        "Performance",
        "# Parameters"
     ]

    str_table = """\\begin{{tabular}}{{cccccc}}
\\toprule
{}
\\bottomrule
\end{{tabular}}
"""

    lst_lines_tabular = ["&".join(tab_headers)]


    for dataname in dct_table:
        for model in dct_table[dataname]:
            for layers in dct_table[dataname][model]:
                if layers != "Conv+Dense":
                    continue
                for lin in dct_table[dataname][model][layers]:
                    if "PYQALM Q=None" in str(lin["method"]):
                        continue
                    lst_line = [dataname, model, layers]
                    lst_line.append(str(lin["method"]))
                    lst_line.append("{:.2f}".format(lin["perf"]))
                    lst_line.append(str(int(lin["nb_param"])))
                    str_line = "&".join(lst_line).replace("%", "\%").replace("#", "\#")
                    lst_lines_tabular.append(str_line)

    final_string = str_table.format("\\\\ \n".join(lst_lines_tabular) + "\\\\")
    with open(str((output_dir / "table.tex").absolute()), 'w') as wf:
        wf.write(final_string)
    print(final_string)

