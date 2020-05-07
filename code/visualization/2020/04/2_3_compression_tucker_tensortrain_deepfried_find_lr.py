import pathlib
import pandas as pd
from palmnet.visualization.utils import get_palminized_model_and_df, get_df
import matplotlib.pyplot as plt
import numpy as np
import logging
import plotly.graph_objects as go
import plotly.express as px
from pprint import pprint as print


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

    for dataname in dataset:
        df_data = df[df[dataset[dataname]] == 1]
        for base_model_name in basemodels[dataname]:
            df_model = df_data[df_data[base_model_name] == 1]
            df_comp = df_model[df_model["tucker"] == 1]
            for keep_first in [True, False]:
                df_keep = df_comp[df_comp["--keep-first-layer"] == keep_first]

                fig = go.Figure()
                for index, row in df_keep.iterrows():
                    csv_file = row["results_dir"] / row["output_file_csvcbprinter"]
                    df_csv = pd.read_csv(csv_file)
                    win_size = 5
                    lr_values = df_csv["lr"].values
                    lr_values_log = np.log10(lr_values)
                    lr_rolling_mean = pd.Series(lr_values_log).rolling(window=win_size).mean().iloc[win_size - 1:].values
                    lr_rolling_mean_exp = 10 ** lr_rolling_mean
                    loss_rolling_mean = df_csv["loss"].rolling(window=win_size).mean().iloc[win_size - 1:].values
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
                    str_keep_first = "-keep_first" if keep_first else ""
                    dct_config_lr[f"{dataset[dataname]}-{base_model_name}-tucker"+str_keep_first] = approx
                title_str = "{}:{} - {} - keep first :{}".format(dataname, base_model_name, "tucker", keep_first)
                fig.update_layout(barmode='group',
                                  title=title_str,
                                  xaxis_title="lr",
                                  yaxis_title="loss",
                                  xaxis_type="log",
                                  xaxis={'type': 'category'},
                                  )
                # fig.show()
    print(dct_config_lr)

def show_for_tensortrain():
    # compression_method = ["tucker", "tensortrain"]
    # df = df.apply(pd.to_numeric, errors='coerce')
    df_comp = df[df["tensortrain"] == 1]
    dct_config_lr = dict()
    for dataname in dataset:
        df_data = df_comp[df_comp[dataset[dataname]] == 1]
        for base_model_name in basemodels[dataname]:
            df_model = df_data[df_data[base_model_name] == 1]
            for keep_first in [True, False]:
                df_keep = df_model[df_model["--keep-first-layer"] == keep_first]

                set_orders = set(df_keep["--order"].values)
                for order in set_orders:
                    df_order = df_keep[df_keep["--order"] == order]
                    set_values = set(df_order["--rank-value"].values)
                    for rank in set_values:
                        df_rank = df_order[df_order["--rank-value"] == rank]
                        fig = go.Figure()
                        for index, row in df_rank.iterrows():
                            csv_file = pathlib.Path(row["results_dir"]) / row["output_file_csvcbprinter"]
                            try:
                                df_csv = pd.read_csv(csv_file)
                            except:
                                continue
                            win_size = 5
                            lr_values = df_csv["lr"].values
                            lr_values_log = np.log10(lr_values)
                            lr_rolling_mean = pd.Series(lr_values_log).rolling(window=win_size).mean().iloc[win_size - 1:].values
                            lr_rolling_mean_exp = 10 ** lr_rolling_mean
                            loss_rolling_mean = df_csv["loss"].rolling(window=win_size).mean().iloc[win_size - 1:].values
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
                            str_keep_first = "-keep_first" if keep_first else ""
                            dct_config_lr[f"{dataset[dataname]}-{base_model_name}-tensortrain-{order}-{rank}"+str_keep_first] = approx

                        title_str = "{}:{} - tensortrain - keep first :{} - order: {} - rank {}".format(dataname, base_model_name,  keep_first, order, rank)
                        fig.update_layout(barmode='group',
                                          title=title_str,
                                          xaxis_title="lr",
                                          yaxis_title="loss",
                                          xaxis_type="log",
                                          xaxis={'type': 'category'},
                                          )
                        # if rank == 12 or rank == 14:
                        #     fig.show()

    print(dct_config_lr)


def show_for_deepfried():
    # compression_method = ["tucker", "tensortrain"]
    # df = df.apply(pd.to_numeric, errors='coerce')
    df_comp = df[df["deepfried"] == 1]
    dct_config_lr = dict()
    for dataname in dataset:
        df_data = df_comp[df_comp[dataset[dataname]] == 1]
        for base_model_name in basemodels[dataname]:
            df_model = df_data[df_data[base_model_name] == 1]
            fig = go.Figure()
            for index, row in df_model.iterrows():
                csv_file = pathlib.Path(row["results_dir"]) / row["output_file_csvcbprinter"]
                try:
                    df_csv = pd.read_csv(csv_file)
                except:
                    continue
                win_size = 5
                lr_values = df_csv["lr"].values
                lr_values_log = np.log10(lr_values)
                lr_rolling_mean = pd.Series(lr_values_log).rolling(window=win_size).mean().iloc[win_size - 1:].values
                lr_rolling_mean_exp = 10 ** lr_rolling_mean
                loss_rolling_mean = df_csv["loss"].rolling(window=win_size).mean().iloc[win_size - 1:].values
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

                dct_config_lr[f"{dataset[dataname]}-{base_model_name}-deepfried"] = approx

            title_str = "{}:{} - deepfried".format(dataname, base_model_name)
            fig.update_layout(barmode='group',
                              title=title_str,
                              xaxis_title="lr",
                              yaxis_title="loss",
                              xaxis_type="log",
                              xaxis={'type': 'category'},
                              )
                        # if rank == 12 or rank == 14:
            fig.show()

    print(dct_config_lr)


if __name__ == "__main__":
    root_source_dir = pathlib.Path("/home/luc/PycharmProjects/palmnet/results/")
    expe_path = "2020/05/2_3_compression_deepfried_find_lr"

    lst_expe_path = [
        # "2020/04/0_0_compression_tucker_tensortrain_find_lr",
        # "2020/04/0_0_compression_tucker_tensortrain_find_lr_only_mnist_tensortrain",
        # "2020/04/0_0_compression_tensortrain_find_lr_12_14",
        "2020/05/2_3_compression_deepfried_find_lr"
    ]
    # expe_path = "2020/04/0_0_compression_tucker_tensortrain_find_lr"
    # expe_path_only_mnist_tensortrain = "2020/04/0_0_compression_tucker_tensortrain_find_lr_only_mnist_tensortrain"
    # expe_path_tensortrain_12_14 = "2020/04/0_0_compression_tensortrain_find_lr_12_14"
    #
    # src_results_dir = root_source_dir / expe_path
    # src_results_dir_only_mnist_tensortrain = root_source_dir / expe_path_only_mnist_tensortrain
    # src_results_dir_tensortrain_12_14 = root_source_dir / expe_path_tensortrain_12_14

    get_df_and_assign = lambda x: get_df(root_source_dir / x).assign(results_dir=str(root_source_dir / x))

    df = pd.concat(list(map(get_df_and_assign, lst_expe_path)))
    # df = get_df_and_assign(src_results_dir)
    # df_only_mnist = get_df_and_assign(src_results_dir_only_mnist_tensortrain)
    # df_tensortrain_12_14 = get_df_and_assign(src_results_dir_tensortrain_12_14)
    #
    # df = pd.concat([df, df_only_mnist, df_tensortrain_12_14])

    df = df.dropna(subset=["failure"])
    df = df.drop(columns="oar_id").drop_duplicates()


    root_output_dir = pathlib.Path("/home/luc/PycharmProjects/palmnet/reports/figures/")
    output_dir = root_output_dir / expe_path / "line_plots"
    output_dir.mkdir(parents=True, exist_ok=True)


    show_for_deepfried()
    # show_for_tensortrain()
    # show_for_tucker()