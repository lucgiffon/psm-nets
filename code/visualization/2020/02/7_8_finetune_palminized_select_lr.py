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

dataset = {
    "Cifar10": "--cifar10",
    "Cifar100": "--cifar100",
    "SVHN": "--svhn",
    "MNIST": "--mnist"
}

basemodels = {
    "Cifar100": ["--cifar100-vgg19", "--cifar100-resnet20", "--cifar100-resnet50"],
    "Cifar10": ["--cifar10-vgg19"],
    "SVHN": ["--svhn-vgg19"],
    "MNIST": ["--mnist-lenet"]
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
    "nb_flop",
    "nb_param",
    "finetuned_score",
    "compression_rate"
}

if __name__ == "__main__":
    root_source_dir = pathlib.Path("/home/luc/PycharmProjects/palmnet/results/")

    expe_path = "2020/02/7_8_finetune_palminized_select_lr"

    src_results_dir = root_source_dir / expe_path

    df = get_df(src_results_dir)

    df = df.dropna(subset=["failure"])
    df = df.drop(columns="oar_id").drop_duplicates()


    root_output_dir = pathlib.Path("/home/luc/PycharmProjects/palmnet/reports/figures/")
    output_dir = root_output_dir / expe_path / "histogrammes"
    output_dir.mkdir(parents=True, exist_ok=True)

    # sparsity_factors = sorted(set(df_palminized["--sparsity-factor"]))
    nb_factors = set(df["--nb-factor"])

    sparsy_factors = sorted(set(df["--sparsity-factor"].values))



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

    # df = df.apply(pd.to_numeric, errors='coerce')
    for dataname in dataset:
        df_data = df[df[dataset[dataname]] == 1]
        for base_model_name in basemodels[dataname]:
            df_model = df_data[df_data[base_model_name] == 1]
            for nb_fac in nb_factors:
                df_fac = df_model[df_model["--nb-factor"] == nb_fac]
                fig = go.Figure()
                for index, row in df_fac.iterrows():
                    csv_file = src_results_dir / row["output_file_csvcbprinter"]
                    df_csv = pd.read_csv(csv_file)
                    win_size = 5
                    lr_values = df_csv["lr"].values
                    lr_values_log = np.log10(lr_values)
                    lr_rolling_mean = pd.Series(lr_values_log).rolling(window=win_size).mean().iloc[win_size-1:].values
                    lr_rolling_mean_exp = 10 ** lr_rolling_mean
                    loss_rolling_mean = df_csv["loss"].rolling(window=win_size).mean().iloc[win_size-1:].values
                    delta_loss = (np.hstack([loss_rolling_mean, [0]]) - np.hstack([[0], loss_rolling_mean]))[1:-1]

                    delta_loss_rolling_mean = pd.Series(delta_loss).rolling(window=win_size).mean().iloc[win_size-1:].values
                    lr_rolling_mean_2x = pd.Series(lr_rolling_mean).rolling(window=win_size).mean().iloc[win_size-1:].values
                    lr_rolling_mean_2x_exp = 10 ** lr_rolling_mean_2x

                    # fig.add_trace(go.Scatter(x=lr_rolling_mean_exp, y=loss_rolling_mean, name="sp_fac {} - hiearchical {}".format(row["--sparsity-factor"], row["--hierarchical"])))
                    fig.add_trace(go.Scatter(x=lr_rolling_mean_2x_exp[:-1], y=delta_loss_rolling_mean, name="sp_fac {} - hiearchical {}".format(row["--sparsity-factor"], row["--hierarchical"])))

                title_str = "{}:{} - nb_fac:{}".format(dataname, base_model_name, nb_fac)
                fig.update_layout(barmode='group',
                                    title=title_str,
                                    xaxis_title="lr",
                                    yaxis_title="loss",
                                    xaxis_type="log",
                                    xaxis={'type': 'category'},
                                    )
                fig.show()
            #
            # fig = go.Figure()
            #
            # # base model
            # ############v
            # fig.add_trace(
            #     go.Scatter(
            #         x=[-1, "2", "3", "log(min(A, B))", 1],
            #         y=[val, val, val, val, val],
            #         mode='lines',
            #         name="base model"
            #     ))
            # # fig, ax = plt.subplots()
            # # max_value_in_plot = -1
            # # bar_width = 0.9 / (len(sparsity_factors)*2 + 1)
            #
            # # palminized
            # ############
            # for i, sp_fac_palm in enumerate(sparsy_factors_palm):
            #     df_sparsity_palm = df_data_palminized[df_data_palminized["--sparsity-factor"] == sp_fac_palm]
            #     for hierarchical_value in [1, 0]:
            #         hierarchical_str = " H" if hierarchical_value == 1 else ""
            #         df_data_palminized_hierarchical = df_sparsity_palm[df_sparsity_palm["--hierarchical"] == hierarchical_value]
            #         if task == "compression_rate":
            #             val = df_data_palminized_hierarchical.sort_values("--nb-factor", na_position="last")[task_palminized].values
            #             val = nb_param_base / val
            #         else:
            #             val = df_data_palminized_hierarchical.sort_values("--nb-factor", na_position="last")[task_palminized].values
            #
            #         hls_str = "hsl({}, {}%, 60%)".format(hue_by_sparsity[sp_fac_palm], saturation_by_hier[hierarchical_value])
            #         fig.add_trace(go.Bar(name=('Palm {}' + hierarchical_str).format(sp_fac_palm), x=[xticks[-1]] if hierarchical_value == 1 else xticks, y=val, marker_color=hls_str))
            #
            # title = task + " " + dataname
            #
            # fig.update_layout(barmode='group',
            #                   title=title,
            #                   xaxis_title="# Factor",
            #                   yaxis_title=ylabel_task[task],
            #                   yaxis_type=scale_tasks[task],
            #                   xaxis={'type': 'category'},
            #                   )
            # fig.show()
            # fig.write_image(str((output_dir / title).absolute()) + ".png")
