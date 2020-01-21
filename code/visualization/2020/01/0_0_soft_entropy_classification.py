import pathlib
import pandas as pd
from palmnet.visualization.utils import get_palminized_model_and_df, get_df
import matplotlib.pyplot as plt
import numpy as np
import logging
import plotly.graph_objects as go

from skluc.utils import logger

mpl_logger = logging.getLogger('matplotlib')
mpl_logger.setLevel(logging.ERROR)

dataset = {
    "Cifar10": "--cifar10",
    "SVHN": "--svhn",
    "MNIST": "--mnist"
}

color_bars_sparsity = {
    2: "g",
    3: "c",
    4: "b",
    5: "y"
}

hatch_bars_permutation = {
    0: "/",
    1: None
}

hatch_hierarchical = {
    0: "X",
    1: "O"
}

tasks = {
    "finetuned_score"
}
ylabel_task = {
    "nb_flop": "log(# Flop)",
    "nb_param": "log(# non-zero value)",
    "finetuned_score": "Accuracy"
}

scale_tasks = {
    "finetuned_score": "linear"
}

if __name__ == "__main__":
    FORCE = False

    logger.setLevel(logging.ERROR)
    root_source_dir = pathlib.Path("/home/luc/PycharmProjects/palmnet/results/")
    expe_path = "2020/01/0_0_soft_entropy_classification"
    src_results_dir = root_source_dir / expe_path

    df_path = src_results_dir / "prepared_df.csv"

    if df_path.exists() and not FORCE:
        df = pd.read_csv(df_path, sep=";")
    else:
        df = get_df(src_results_dir)
        df[["failure", "finetuned_score", "--nb-factor"]] = df[["failure", "finetuned_score", "--nb-factor"]].apply(pd.to_numeric, errors='coerce')
        df = df.dropna(subset=["failure", "finetuned_score"]).drop(columns="oar_id").drop_duplicates()
        df.to_csv(df_path, sep=";")

    root_output_dir = pathlib.Path("/home/luc/PycharmProjects/palmnet/reports/figures/")
    output_dir = root_output_dir / expe_path / "histogrammes"
    output_dir.mkdir(parents=True, exist_ok=True)

    nb_units_dense_layer = set(df["--nb-units-dense-layer"])
    param_reg_softentropy = set(df["--param-reg-softmax-entropy"])
    param_reg_softentropy.remove("None")
    sparsity_factors = sorted(set(df["--sparsity-factor"]))
    sparsity_factors.remove("None")
    nb_factors = set(df["--nb-factor"])


    hue_by_sparsity= {
        '2': 10,
        '3': 60,
        '4': 110,
        '5': 180
    }

    saturation_by_param_softentropy = dict(zip(sorted(param_reg_softentropy), np.linspace(40, 80, len(param_reg_softentropy), dtype=int)))


    saturation_by_hier = {
        1: 50,
        0: 75
    }

    for dataname in dataset:
        df_data = df[df[dataset[dataname]] == 1]
        for nb_units in  nb_units_dense_layer:
            df_units = df_data[df_data["--nb-units-dense-layer"] == nb_units]
            for task in tasks:
                xticks = ["2", "3", "log(min(A, B))"]
                # xticks = ["A", "B", "log(min(A, B))"]
                # xticks = [1, 2, 3]

                fig = go.Figure()

                # dense model
                ############v
                df_dense = df_units[df_units["--dense-layers"] == 1]

                val = df_dense[task].values.mean()
                fig.add_trace(
                    go.Scatter(
                        x=[-1, "2", "3", "log(min(A, B))", 1],
                        y=[val, val, val, val, val],
                        mode='lines',
                        name="dense model"
                    ))


                # pbp
                #####
                for i, sp_fac in enumerate(sorted(sparsity_factors)):
                    df_sparsity = df_units[df_units["--sparsity-factor"] == sp_fac]
                    for param_reg in sorted(param_reg_softentropy):
                        df_reg = df_sparsity[df_sparsity["--param-reg-softmax-entropy"] == param_reg]
                        finetune_score_values = df_reg.sort_values("--nb-factor", na_position="last")[task].values

                        hls_str = "hsl({}, {}%, 40%)".format(hue_by_sparsity[sp_fac], saturation_by_param_softentropy[param_reg])
                        fig.add_trace(go.Bar(name='sparsity {} - reg {}'.format(sp_fac, param_reg), x=xticks, y=finetune_score_values, marker_color=hls_str))


                title = task + " " + dataname + " " + nb_units

                fig.update_layout(barmode='group',
                                  title=title,
                                  xaxis_title="# Factor",
                                  yaxis_title=ylabel_task[task],
                                  yaxis_type=scale_tasks[task],
                                  xaxis={'type': 'category'},
                                  )
                fig.show()
                fig.write_image(str((output_dir / title).absolute()) + ".png")
