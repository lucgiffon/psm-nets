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

    results_path = "2020/05/12_13_finetune_palminized_explore_param_bis_bis"

    src_results_path = root_source_dir / results_path / "results.csv"

    root_output_dir = pathlib.Path("/home/luc/PycharmProjects/palmnet/reports/figures/")
    output_dir = root_output_dir / results_path / "losses"
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(src_results_path, header=0)
    df = df.fillna("None")

    df = df[df["actual-nb-epochs"] == 1000]


    hue_by_epoch_step_size= {
        2: 10,
        10: 40,
        100:30,
        200: 50,
        300: 60,
        400: 70,
        "None": 100
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
        "None": 20,
        "triangular": 50,
        "triangular2": 80
    }

    datasets = set(df["dataset"].values)
    for dataname in datasets:
        df_data = df[df["dataset"] == dataname]
        df_model_values = set(df_data["model"].values)
        for modelname in df_model_values:
            df_model = df_data[df_data["model"] == modelname]
            set_nb_facs = set(df_model["nb-factor"].values)
            for nb_fac in set_nb_facs:
                df_nb_fac = df_model[df_model["nb-factor"] == nb_fac]
                set_sparsity = set(df_model["sparsity-factor"].values)
                for spars in set_sparsity:
                    df_sparsity = df_nb_fac[df_nb_fac["sparsity-factor"] == spars]
                    fig = go.Figure()
                    for idx_row, row in df_sparsity.iterrows():
                        clr_value = row["use-clr"]
                        logrange_clr = row["logrange"]
                        idx_xp = str(row["idx-expe"])
                        path_history = row["path-learning-history-epoch"]
                        epoch_step_size = row["epoch-step-size"]
                        df_csv_history = pd.read_csv(path_history)
                        # loss_values = df_csv_history["loss"].values
                        # win_size = 50
                        loss_values = df_csv_history["loss"].values
                        # loss_values = df_csv_history["loss"].rolling(window=win_size).mean().iloc[win_size - 1:].values
                        epoch_values = df_csv_history["epoch"].values
                        try:
                            assert all(epoch_values[i] <= epoch_values[i + 1] for i in range(len(epoch_values) - 1))
                        except:
                            print(f"unordered epoch numbers {idx_xp}")

                        hls_str = "hsl({}, {}%, 50%)".format(hue_by_epoch_step_size[epoch_step_size], lum_by_clr[clr_value])

                        txt_name = f'Palm {clr_value} {epoch_step_size} log {logrange_clr}'
                        group = f"log {clr_value} {logrange_clr}"
                        fig.add_trace(go.Scatter(name=txt_name, hovertext=txt_name, legendgroup=group,
                                             x=np.arange(len(loss_values)), y=loss_values,
                                             marker_color=hls_str))

                    title = "_".join(str(x) for x in [dataname, modelname, f"nbfac {nb_fac}", f"sparsity {spars}",])

                    fig.update_layout(barmode='group',
                                      title=title,
                                      xaxis_title="Batch % 100",
                                      yaxis_title="Loss",
                                      yaxis_type="linear",
                                      )
                    fig.show()
                    fig.write_image(str((output_dir / title).absolute()) + ".png")
                    # exit()
