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

if __name__ == "__main__":
    root_source_dir = pathlib.Path("/home/luc/PycharmProjects/palmnet/results/processed")

    results_path = "2020/03/9_10_finetune_palminized_no_useless"
    results_path_2 = "2020/04/9_10_finetune_palminized_no_useless"
    results_path_tucker = "2020/04/0_0_compression_tucker_tensortrain"

    src_results_path = root_source_dir / results_path / "results.csv"
    src_results_path_2 = root_source_dir / results_path_2 / "results.csv"
    src_results_path_tucker = root_source_dir / results_path_tucker / "results.csv"

    root_output_dir = pathlib.Path("/home/luc/PycharmProjects/palmnet/reports/figures/")
    output_dir = root_output_dir / results_path / "histogrammes"
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(src_results_path, header=0)
    df_2 = pd.read_csv(src_results_path_2, header=0)
    df = pd.concat([df, df_2])
    df = df.fillna("None")
    df = df.drop(columns=["Unnamed: 0", "idx-expe"]).drop_duplicates()

    df = df[df["keep-last-layer"] == 0]
    df = df[df["use-clr"] == 1]

    df_tucker_tt = pd.read_csv(src_results_path_tucker, header=0)
    df_tucker_tt = df_tucker_tt.fillna("None")
    # sparsity_factors = sorted(set(df_palminized["--sparsity-factor"]))
    nb_factors = set(df["nb-factor"].values)

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
        "PALM Q=2": "square",
        "PALM Q=3": "diamond",
        "PALM Q=None": "square-x",
        "PALM Q=None H": "star-square",
        "Base": "x",
        "Tucker": "circle",
        "Tucker -1": "circle-dot",
        "TT": "triangle-up",
        "TT -1": "triangle-up-dot",
    }

    dct_colors = {
        "PALM K=2": "dodgerblue",
        "PALM K=3": "darkorchid",
        "PALM K=4": "green",
        "TT R=2": "orange",
        "TT R=6": "gold",
        "TT R=10": "red",
        "Base": "grey",
        "Tucker": "pink",
        "Tucker 10%": "orange",
        "Tucker 20%": "gold",
        "Tucker 30%": "red"
    }

    SIZE_MARKERS = 15
    WIDTH_MARKER_LINES = 2

    datasets = set(df["dataset"].values)
    for dataname in datasets:
        df_data = df[df["dataset"] == dataname]
        df_tucker_tt_data =  df_tucker_tt[df_tucker_tt["dataset"] == dataname]
        df_model_values = set(df_data["model"].values)

        for modelname in df_model_values:
            df_model = df_data[df_data["model"] == modelname]
            df_tucker_tt_model = df_tucker_tt_data[df_tucker_tt_data["model"] == modelname]

            fig = go.Figure()

            base_score = None
            base_nb_param = None

            for idx_row, row in df_model.iterrows():
                hierarchical_value = row["hierarchical"]
                str_hierarchical = ' H' if hierarchical_value is True else ''
                try:
                    nb_factor = int(row["nb-factor"])
                except:
                    nb_factor = None
                sparsity_factor = int(row["sparsity-factor"])

                name_trace = f"PALM Q={nb_factor} K={sparsity_factor}{str_hierarchical}"

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
                        legendgroup=f"PALM K={sparsity_factor}",
                        marker=dict(
                                    color=dct_colors[f"PALM K={sparsity_factor}"],
                                    symbol=dct_symbol[f"PALM Q={nb_factor}{str_hierarchical}"],
                                    size=SIZE_MARKERS,
                                    line=dict(
                                        color='Black',
                                        width=WIDTH_MARKER_LINES
                                    )
                                )

                    ))

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

            df_tucker = df_tucker_tt_model[df_tucker_tt_model["compression"] == "tucker"]
            for idx_row, row in df_tucker.iterrows():
                keep_first = row["keep-first-layer"]
                str_keep_first = ' -1' if keep_first is True else ''

                try:
                    rank_percentage = int(float(row["rank-percentage-dense"]) * 100)
                except:
                    rank_percentage = None

                str_percentage = f' {rank_percentage}%' if rank_percentage is not None else ''

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


            df_tt = df_tucker_tt_model[df_tucker_tt_model["compression"] == "tensortrain"]
            for idx_row, row in df_tt.iterrows():
                keep_first = row["keep-first-layer"]
                str_keep_first = ' -1' if keep_first is True else ''

                order = int(row["order"])
                rank_value = int(row["rank-value"])

                name_trace = f"Tensortrain{str_keep_first} K={order} R={rank_value}"

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
                                    symbol=dct_symbol[f"TT{str_keep_first}"],
                                    size=SIZE_MARKERS,
                                    line=dict(
                                        color='Black',
                                        width=WIDTH_MARKER_LINES
                                    )
                                )

                    ))

            title = "Performance = f(# Param); " + dataname + " " + modelname

            fig.update_layout(title=title,
                              xaxis_title="# Parameter",
                              yaxis_title="Performance",
                              xaxis_type="log",
                              )
            fig.show()
            # fig.write_image(str((output_dir / title).absolute()) + ".png")
