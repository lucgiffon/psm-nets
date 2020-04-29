import pathlib
import pandas as pd
from palmnet.visualization.utils import get_palminized_model_and_df, get_df
import matplotlib.pyplot as plt
import numpy as np
import logging
import plotly.graph_objects as go
import plotly.express as px
from pprint import pprint as pprint


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


def show_for_tucker():
    # compression_method = ["tucker", "tensortrain"]
    # df = df.apply(pd.to_numeric, errors='coerce')
    dct_config_lr = dict()
    lst_name_trace_low = list()

    for dataname in dataset:
        df_data = df[df[dataset[dataname]] == 1]
        for base_model_name in basemodels[dataname]:
            df_model = df_data[df_data[base_model_name] == 1]

            for index, row in df_model.iterrows():
                fig = go.Figure()

                csv_file = pathlib.Path(row["results_dir"]) / row["output_file_csvcbprinter"]
                df_csv = pd.read_csv(csv_file)
                win_size = 5
                lr_values = df_csv["lr"].values
                lr_values_log = np.log10(lr_values)
                lr_rolling_mean = pd.Series(lr_values_log).rolling(window=win_size).mean().iloc[win_size - 1:].values
                loss_rolling_mean = df_csv["loss"].rolling(window=win_size).mean().iloc[win_size - 1:].values

                if all(np.isnan(loss_rolling_mean)):
                    continue

                delta_loss = (np.hstack([loss_rolling_mean, [0]]) - np.hstack([[0], loss_rolling_mean]))[1:-1]

                delta_loss_rolling_mean = pd.Series(delta_loss).rolling(window=win_size).mean().iloc[win_size - 1:].values
                lr_rolling_mean_2x = pd.Series(lr_rolling_mean).rolling(window=win_size).mean().iloc[win_size - 1:].values
                lr_rolling_mean_2x_exp = 10 ** lr_rolling_mean_2x


                # fig.add_trace(go.Scatter(x=lr_rolling_mean_exp, y=loss_rolling_mean, name="sp_fac {} - hiearchical {}".format(row["--sparsity-factor"], row["--hierarchical"])))
                fig.add_trace(go.Scatter(x=lr_rolling_mean_2x_exp[:-1], y=delta_loss_rolling_mean, name=""))

                argmin_loss = np.argmin(delta_loss_rolling_mean)
                val = lr_rolling_mean_2x_exp[:-1][argmin_loss]
                log_val = np.log10(val)
                approx = 10 ** np.around(log_val, decimals=0)

                sparsity = int(row["--sparsity-factor"])
                hierarchical = bool(row["--hierarchical"])
                str_hierarchical = " H" if hierarchical else ""
                try:
                    nb_fac = int(row["--nb-factor"])
                except ValueError:
                    nb_fac = None

                name_trace = f"tucker_sparse_facto-{dataset[dataname]}-{base_model_name}-Q={nb_fac}-K={sparsity}{str_hierarchical}"
                print(len(delta_loss_rolling_mean), name_trace)
                if len(delta_loss_rolling_mean) < 10:
                    lst_name_trace_low.append(name_trace)
                    continue


                dct_config_lr[name_trace] = approx

                # title_str = "{}:{} - {} - keep first :{}".format(dataname, base_model_name, "tucker", keep_first)
                fig.update_layout(barmode='group',
                                  title=name_trace,
                                  xaxis_title="lr",
                                  yaxis_title="loss",
                                  xaxis_type="log",
                                  xaxis={'type': 'category'},
                                  )
                # fig.show()
    pprint(dct_config_lr)
    pprint(lst_name_trace_low)

if __name__ == "__main__":
    root_source_dir = pathlib.Path("/home/luc/PycharmProjects/palmnet/results/")

    expe_path = "2020/04/0_0_compression_tucker_sparse_facto_select_lr"
    expe_path_errors = "2020/04/0_0_compression_tucker_sparse_facto_select_lr_errors"

    src_results_dir = root_source_dir / expe_path
    src_results_dir_errors = root_source_dir / expe_path_errors

    get_df_and_assign = lambda x: get_df(x).assign(results_dir=str(x))
    df = get_df_and_assign(src_results_dir)
    df_errors = get_df_and_assign(src_results_dir_errors)

    df = pd.concat([df, df_errors])

    df = df.dropna(subset=["failure"])
    df = df[df["failure"] == 0]
    df = df.drop(columns="oar_id").drop_duplicates()

    root_output_dir = pathlib.Path("/home/luc/PycharmProjects/palmnet/reports/figures/")
    output_dir = root_output_dir / expe_path / "line_plots"
    output_dir.mkdir(parents=True, exist_ok=True)


    show_for_tucker()