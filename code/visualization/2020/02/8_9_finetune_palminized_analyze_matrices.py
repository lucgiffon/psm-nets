import pathlib
import pandas as pd
from palmnet.visualization.utils import get_palminized_model_and_df, get_df
import matplotlib.pyplot as plt
import numpy as np
import logging
import plotly.graph_objects as go
import plotly.express as px


mpl_logger = logging.getLogger('matplotlib')
mpl_logger.setLevel(logging.ERROR)

if __name__ == "__main__":
    root_source_dir = pathlib.Path("/home/luc/PycharmProjects/palmnet/results/processed")

    results_path = "2020/02/8_9_finetune_palminized_resnet_new_lr/"

    src_results_path = root_source_dir / results_path / "results_layers.csv"

    root_output_dir = pathlib.Path("/home/luc/PycharmProjects/palmnet/reports/figures/")
    output_dir = root_output_dir / results_path / "histogrammes"
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(src_results_path, header=0)
    df = df.fillna("None")
    # df["compression-rate"] = df["nb-non-zero-base"] / df["nb-non-zero-compressed"]
    # df["non-zero-rate"] = df["nb-non-zero-base"] / df["nb-non-zero-reconstructed"]
    # df["non-zero-prop"] =  df["nb-non-zero-reconstructed"] / df["nb-non-zero-base"]

    # sparsity_factors = sorted(set(df_palminized["--sparsity-factor"]))

    # fig = go.Figure()
    # fig.add_trace(go.Scatter(x=df["entropy-base-sv-normalized"],
    #                         y=df["entropy-recons-sv-normalized"],
    #                         mode='markers',
    # ))

    models = set(df["model"].values)
    datasets = set(df["data"].values)
    for data in datasets:
        df_data = df[df["data"] == data]
        for model in models:
            df_model = df_data[df_data["model"] == model]

            fig = px.scatter(df_model, x="entropy-base-sv-normalized", y="entropy-recons-sv-normalized", color="nb-factor-param",
                             size='diff-approx', hover_data=['model', 'layer-name', 'data', 'nb-factor-param',
                                                              'sparsity-factor', 'nb-non-zero-compressed',
                                                              'nb-non-zero-base', 'nb-non-zero-reconstructed', 'compression-rate', 'non-zero-rate'])


            fig.update_layout(title="{} {} Entropie SV reconstruit en fonction de Entropie SV base".format(data, model),
                              xaxis_title="Entropie SV base",
                              yaxis_title="Entropie SV reconstruit",
                              yaxis_type="linear"
            )
            fig.show()


            fig = px.scatter(df_model, x="entropy-base-sv-normalized", y="non-zero-prop", color="nb-factor-param",
                             size='diff-approx', hover_data=['model', 'layer-name', 'data', 'nb-factor-param',
                                                              'sparsity-factor', 'nb-non-zero-compressed',
                                                              'nb-non-zero-base', 'nb-non-zero-reconstructed',
                                                             'compression-rate', 'non-zero-rate', "entropy-recons-sv-normalized"])


            fig.update_layout(title="{} {} Prop non zero reconstruit en fonction entropie base sv".format(data, model),
                              xaxis_title="Entropie SV base",
                              yaxis_title="Prop non zero reconstruit",
                              yaxis_type="linear"
            )
            fig.show()

            fig = px.scatter(df_model, x="entropy-base-sv-normalized", y="diff-approx", color="nb-factor-param",
                             hover_data=['model', 'layer-name', 'data', 'nb-factor-param',
                                                              'sparsity-factor', 'nb-non-zero-compressed',
                                                              'nb-non-zero-base', 'nb-non-zero-reconstructed',
                                                             'compression-rate', 'non-zero-rate', "entropy-recons-sv-normalized"])


            fig.update_layout(title="{} {} Erreur en fonction entropie base sv".format(data, model),
                              xaxis_title="Entropie SV base",
                              yaxis_title="Erreur reconstruction",
                              yaxis_type="linear"
            )
            fig.show()

            fig = px.scatter(df_model, x="nb-non-zero-base", y="diff-approx", color="nb-factor-param",
                             hover_data=['model', 'layer-name', 'data', 'nb-factor-param',
                                                              'sparsity-factor', 'nb-non-zero-compressed',
                                                              'nb-non-zero-base', 'nb-non-zero-reconstructed',
                                                             'compression-rate', 'non-zero-rate', "entropy-recons-sv-normalized"])


            fig.update_layout(title="{} {} Erreur en fonction taille".format(data, model),
                              xaxis_title="Nb non-zero base",
                              yaxis_title="Erreur reconstruction",
                              yaxis_type="linear"
            )
            fig.show()

            fig = px.scatter(df_model, x="idx-layer", y="diff-approx", color="nb-factor-param",
                             hover_data=['model', 'layer-name', 'data', 'nb-factor-param',
                                                              'sparsity-factor', 'nb-non-zero-compressed',
                                                              'nb-non-zero-base', 'nb-non-zero-reconstructed',
                                                             'compression-rate', 'non-zero-rate', "entropy-recons-sv-normalized"])


            fig.update_layout(title="{} {} Erreur en fonction profondeur".format(data, model),
                              xaxis_title="Idx layer (processing order)",
                              yaxis_title="Erreur reconstruction",
                              yaxis_type="linear"
            )
            fig.show()

            fig = px.scatter(df_model, x="idx-layer", y="entropy-base-sv-normalized", color="nb-factor-param",
                             hover_data=['model', 'layer-name', 'data', 'nb-factor-param',
                                                              'sparsity-factor', 'nb-non-zero-compressed',
                                                              'nb-non-zero-base', 'nb-non-zero-reconstructed',
                                                             'compression-rate', 'non-zero-rate', "entropy-recons-sv-normalized"])


            fig.update_layout(title="{} {} entropie SV en fonction profondeur".format(data, model),
                              xaxis_title="Idx layer (processing order)",
                              yaxis_title="Entropie SV",
                              yaxis_type="linear"
            )
            fig.show()

            fig = px.scatter(df_model, x="nb-non-zero-base", y="entropy-base-sv-normalized", color="nb-factor-param",
                             hover_data=['model', 'layer-name', 'data', 'nb-factor-param',
                                                              'sparsity-factor', 'nb-non-zero-compressed',
                                                              'nb-non-zero-base', 'nb-non-zero-reconstructed',
                                                             'compression-rate', 'non-zero-rate', "entropy-recons-sv-normalized"])


            fig.update_layout(title="{} {} entropie SV en fonction taille".format(data, model),
                              xaxis_title="Taille couche",
                              yaxis_title="Entropie SV",
                              yaxis_type="linear"
            )
            fig.show()

