import pathlib
import pandas as pd
from palmnet.visualization.utils import get_palminized_model_and_df, get_df
import matplotlib.pyplot as plt
import numpy as np
import logging
import plotly.graph_objects as go

mpl_logger = logging.getLogger('matplotlib')
mpl_logger.setLevel(logging.ERROR)

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

hatch_bars_permutation = {
    0: "/",
    1: None
}

hatch_hierarchical = {
    0: "X",
    1: "O"
}

tasks = {
    # "nb_flop",
    # "nb_param",
    "finetuned-score",
    "param-compression-rate"
}
ylabel_task = {
    "nb_flop": "log(# Flop)",
    "nb_param": "log(# non-zero value)",
    "finetuned-score": "Accuracy",
    "param-compression-rate": "Compression Rate"
}

scale_tasks = {
    "nb_flop": "log",
    "nb_param": "log",
    "finetuned-score": "linear",
    "param-compression-rate": "linear"
}

if __name__ == "__main__":
    root_source_dir = pathlib.Path("/home/luc/PycharmProjects/palmnet/results/processed")

    results_path = "2020/02/8_9_finetune_palminized_resnet_new_lr/"

    src_results_path = root_source_dir / results_path / "results.csv"

    root_output_dir = pathlib.Path("/home/luc/PycharmProjects/palmnet/reports/figures/")
    output_dir = root_output_dir / results_path / "histogrammes"
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(src_results_path, header=0)
    df = df.fillna("None")
    # sparsity_factors = sorted(set(df_palminized["--sparsity-factor"]))
    nb_factors = set(df["nb-factor"].values)

    sparsy_factors_palm = sorted(set(df["sparsity-factor"].values))

    datasets = set(df["dataset"].values)

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
        0: 80
    }


    for dataname in datasets:
        df_data = df[df["dataset"] == dataname]
        df_model_values = set(df_data["model"].values)
        for modelname in df_model_values:
            df_model = df_data[df_data["model"] == modelname]
            for task in tasks:
                xticks = ["2", "3", "log(min(A, B))"]
                # xticks = ["A", "B", "log(min(A, B))"]
                # xticks = [1, 2, 3]

                fig = go.Figure()

                # base model
                ############v
                if task == "finetuned-score":
                    task_base = "base-model-score"
                elif task == "nb-flop":
                    task_base = "nb-flop-base"
                elif task == "nb-param":
                    task_base = "nb-param-base"
                elif task == "param-compression-rate":
                    val_base = 1.
                else:
                    raise ValueError("Unknown task {}".format(task))

                if task != "param-compression-rate":
                    val_base = df_model[task_base].values.mean()

                fig.add_trace(
                    go.Scatter(
                        x=[-1] + xticks + [1],
                        y=[val_base, val_base, val_base, val_base, val_base],
                        mode='lines',
                        name="base model"
                    ))


                # palminized
                ############

                for i, sp_fac_palm in enumerate(sparsy_factors_palm):
                    df_sparsity_palm = df_model[df_model["sparsity-factor"] == sp_fac_palm]
                    for hierarchical_value in [1, 0]:
                        hierarchical_str = " H" if hierarchical_value == 1 else ""
                        df_data_palminized_hierarchical = df_sparsity_palm[df_sparsity_palm["hierarchical"] == hierarchical_value]
                        for clr_value in [1, 0]:
                            df_clr = df_data_palminized_hierarchical[df_data_palminized_hierarchical["use-clr"] == clr_value]
                            val = df_clr.sort_values("nb-factor", na_position="last")[task].values

                            hls_str = "hsl({}, {}%, {}%)".format(hue_by_sparsity[sp_fac_palm], saturation_by_hier[hierarchical_value], lum_by_clr[clr_value])
                            fig.add_trace(go.Bar(name=('Palm {} clr {}' + hierarchical_str).format(sp_fac_palm, clr_value), x=[xticks[-1]] if hierarchical_value == 1 else xticks, y=val, marker_color=hls_str))

                title = task + " " + dataname + " " + modelname

                fig.update_layout(barmode='group',
                                  title=title,
                                  xaxis_title="# Factor",
                                  yaxis_title=ylabel_task[task],
                                  yaxis_type=scale_tasks[task],
                                  xaxis={'type': 'category'},
                                  )
                fig.show()
                fig.write_image(str((output_dir / title).absolute()) + ".png")
