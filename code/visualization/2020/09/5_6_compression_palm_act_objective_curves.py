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
    src_results_path = root_source_dir / results_path / "results.csv"

    df = pd.read_csv(src_results_path, header=0)
    df = df.fillna("None")
    df = df.drop(columns=["Unnamed: 0", "idx-expe"]).drop_duplicates()

    df = df.fillna("None")

    df = df[df["nb-epochs"] == 1]
    df = df[df["batch-size"] == 128]

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

    datasets = set(df["dataset"].values)
    dct_table = dict()
    for dataname in datasets:
        df_data = df[df["dataset"] == dataname]
        df_model_values = set(df_data["model"].values)

        for modelname in df_model_values:
            df_model = df_data[df_data["model"] == modelname]

            dct_fig = defaultdict(lambda: go.Figure())

            for idx_row, row in df_model.iterrows():
                nb_iter_palm = row["nb-iteration-palm"]
                objective_file_str = row["ouptut_file_objectives"]
                train_percentage_size = f"{1 - float(row['train-val-split']):.2f}"

                df_objectives = pd.read_csv(objective_file_str)
                try:
                    layers = set(df_objectives["0"].values)
                    dct_objectives = {"2": "Batch objective",
                                      # "3": "Val Objective"
                                      }
                except KeyError:
                    layers = set(df_objectives["layer_name"].values)
                    dct_objectives = {
                         "obj_batch": "Batch objective",
                                      # "obj_val": "Val Objective"
                                      }
                for key_objective in dct_objectives.keys():
                    for layer in layers:
                        try:
                            df_objectives_layer = df_objectives[df_objectives["0"] == layer]
                            num_batches = df_objectives_layer["1"].values
                        except KeyError:
                            df_objectives_layer = df_objectives[df_objectives["layer_name"] == layer]
                            num_batches = df_objectives_layer["id_batch"].values
                        obj_values = df_objectives_layer[key_objective].values

                        name = f"{nb_iter_palm} PALM iterations {train_percentage_size}% train"

                        fig_name = "_".join(str(x) for x in [dataname, modelname, layer, dct_objectives[key_objective]])
                        fig = dct_fig[fig_name]

                        fig.add_trace(go.Scatter(name=name,
                                             # x=num_batches, y=val_obj_values,
                                             x=num_batches, y=obj_values,
                                             # marker_color=dct_config_color[hover_text],
                                             hovertext=name,
                                             showlegend=True,
                                             mode="lines"
                                             ))

            replace_name_layer = lambda str_name: (str_name.replace("conv2d_", "Conv2D ").replace("dense_", "Dense ").replace("fc", "Dense ").replace("conv", "Conv2D"))

            for fig_name, fig in dct_fig.items():

                x_legend = 0
                y_legend = -0.8
                fig.update_layout(
                    barmode='group',
                    title=fig_name,
                    xaxis_title="Num batch",
                    yaxis_title="Valeur objectif",
                    # yaxis_type="log",
                    # xaxis_tickangle=-75,
                    # showlegend=True,
                    showlegend=True,
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
                # fig.show()
                fig.write_image(str((output_dir / fig_name).absolute()) + ".png")
