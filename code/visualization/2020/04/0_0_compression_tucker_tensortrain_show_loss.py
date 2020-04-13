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

    results_path = "2020/03/9_10_finetune_palminized_no_useless"

    src_results_path = root_source_dir / results_path / "results.csv"

    root_output_dir = pathlib.Path("/home/luc/PycharmProjects/palmnet/reports/figures/")
    output_dir = root_output_dir / results_path / "losses"
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(src_results_path, header=0)
    df = df.fillna("None")


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

    datasets = set(df["dataset"].values)
    for dataname in datasets:
        df_data = df[df["dataset"] == dataname]
        df_model_values = set(df_data["model"].values)
        for modelname in df_model_values:
            df_model = df_data[df_data["model"] == modelname]
            for clr_value in [1]:
                df_clr = df_model[df_model["use-clr"] == clr_value]
                for keep_last_layer in [1, 0]:
                    df_keep = df_clr[df_clr["keep-last-layer"] == keep_last_layer]
                    str_keep = " keep_last" if keep_last_layer else ""
                    for idx_row, row in df_keep.iterrows():
                        fig = go.Figure()
                        idx_xp = str(row["idx-expe"])
                        sp_fac_palm = row["sparsity-factor"]
                        hierarchical_value = row["hierarchical"]
                        hierarchical_str = " H" if hierarchical_value == 1 else ""
                        nb_fac = row["nb-factor"]
                        path_history = row["path-learning-history"]
                        df_csv_history = pd.read_csv(path_history)
                        # loss_values = df_csv_history["loss"].values
                        win_size = 50
                        loss_values = df_csv_history["loss"].rolling(window=win_size).mean().iloc[win_size - 1:].values
                        epoch_values = df_csv_history["epoch"].values
                        try:
                            assert all(epoch_values[i] <= epoch_values[i + 1] for i in range(len(epoch_values) - 1))
                        except:
                            print(f"unordered epoch numbers {idx_xp}")

                        hls_str = "hsl({}, {}%, {}%)".format(hue_by_sparsity[sp_fac_palm] + hue_by_factor[nb_fac], saturation_by_hier[hierarchical_value], lum_by_clr[clr_value])


                        fig.add_trace(go.Scatter(name=f'Palm {sp_fac_palm} {hierarchical_str}',
                                             x=np.arange(len(loss_values)), y=loss_values,
                                             marker_color=hls_str))

                        title = "_".join(str(x) for x in [dataname, modelname, hierarchical_str, str_keep, f"nbfac {nb_fac}", f"sparsity {sp_fac_palm}", idx_xp])

                        fig.update_layout(barmode='group',
                                          title=title,
                                          xaxis_title="Batch % 100",
                                          yaxis_title="Loss",
                                          yaxis_type="linear",
                                          )
                        # fig.show()
                        fig.write_image(str((output_dir / title).absolute()) + ".png")
                        # exit()
