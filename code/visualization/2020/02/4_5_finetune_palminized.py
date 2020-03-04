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
    # "Cifar10": "--cifar10",
    "Cifar100": "--cifar100",
    # "SVHN": "--svhn",
    # "MNIST": "--mnist"
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
ylabel_task = {
    "nb_flop": "log(# Flop)",
    "nb_param": "log(# non-zero value)",
    "finetuned_score": "Accuracy",
    "compression_rate": "Compression Rate"
}

scale_tasks = {
    "nb_flop": "log",
    "nb_param": "log",
    "finetuned_score": "linear",
    "compression_rate": "linear"
}

if __name__ == "__main__":
    root_source_dir = pathlib.Path("/home/luc/PycharmProjects/palmnet/results/")
    # expe_path_mnist = "2019/12/0_0_random_sparse_facto"
    # expe_path_others = "2019/12/0_0_random_sparse_facto/bis_no_mnist"

    expe_path_palminized_mnist = "2019/11/0_0_finetune_palminized"
    expe_path_palminized_cifar10 = "2019/11/0_0_finetune_palminized/bis_cifar10/bis_bis"
    expe_path_palminized_cifar100_palm = "2020/02/4_5_finetune_palminized_cifar100"
    expe_path_palminized_cifar100_palm_before_finetune = "2020/02/1_2_palminize_cifar_100"
    expe_path_palminized_svhn_cifar100 = "2019/11/0_0_finetune_palminized/bis_cifar100_svhn/bis_bis"
    expe_path_palminized_before_finetune = "2019/10/0_0_hierarchical_palminize"
    expe_path_palminized_not_hierarchical_log = "2020/01/0_1_finetune_palminized_not_hierarchical_log_bis"
    expe_path_palminized_not_hierarchical_log_before_finetune = "2020/01/0_1_palmnet_zero_not_hierarchical_log"
    expe_path_palminized_not_hierarchical_2_3_factors = "2020/01/2_3_finetune_palminized_with_2_3_factors_nobug"
    expe_path_palminized_not_hierarchical_2_3_factors_before_finetune = "2019/12/0_1_palmnet_patient_zero"

    src_results_dir_palminized_mnist = root_source_dir / expe_path_palminized_mnist
    src_results_dir_palminized_before_finetune = root_source_dir / expe_path_palminized_before_finetune
    src_results_dir_palminized_cifar10 = root_source_dir / expe_path_palminized_cifar10
    src_results_dir_palminized_cifar100_palm = root_source_dir / expe_path_palminized_cifar100_palm
    src_results_dir_palminized_cifar100_palm_before_finetune = root_source_dir / expe_path_palminized_cifar100_palm_before_finetune
    src_results_dir_palminized_svhn_cifar100 = root_source_dir / expe_path_palminized_svhn_cifar100
    # src_results_dir_mnist = root_source_dir / expe_path_mnist
    # src_results_dir_others = root_source_dir / expe_path_others
    src_results_dir_palminized_not_hierarchical_log = root_source_dir / expe_path_palminized_not_hierarchical_log
    src_results_dir_palminized_not_hierarchical_log_before_finetune = root_source_dir / expe_path_palminized_not_hierarchical_log_before_finetune
    src_results_dir_palminized_not_hierarchical_2_3_factors = root_source_dir / expe_path_palminized_not_hierarchical_2_3_factors
    src_results_dir_palminized_not_hierarchical_2_3_factors_before_finetune = root_source_dir / expe_path_palminized_not_hierarchical_2_3_factors_before_finetune


    df_palminized_mnist = get_df(src_results_dir_palminized_mnist)
    df_palminized_cifar10 = get_df(src_results_dir_palminized_cifar10)
    df_palminized_cifar100_palm = get_df(src_results_dir_palminized_cifar100_palm)
    df_palminized_cifar100_svhn = get_df(src_results_dir_palminized_svhn_cifar100)
    df_palmnized_not_hierarchical_log = get_df(src_results_dir_palminized_not_hierarchical_log)
    df_palminized_not_hierarchical_2_3_factors = get_df(src_results_dir_palminized_not_hierarchical_2_3_factors)

    df_palminized = pd.concat([df_palminized_cifar100_palm, df_palminized_mnist, df_palminized_cifar10, df_palminized_cifar100_svhn, df_palmnized_not_hierarchical_log, df_palminized_not_hierarchical_2_3_factors])
    df_palminized = df_palminized.dropna(subset=["failure"])
    df_palminized = df_palminized[df_palminized["finetuned_score"] != "None"]
    df_palminized = df_palminized.drop(columns="oar_id").drop_duplicates()

    df_palminized_before_finetune = get_df(src_results_dir_palminized_before_finetune)
    df_palminized_before_finetune_cifar100_palm_before_finetune = get_df(src_results_dir_palminized_cifar100_palm_before_finetune)
    df_palminized_not_hierarchical_log_before_finetune = get_df(src_results_dir_palminized_not_hierarchical_log_before_finetune)
    df_palminized_not_hierarchical_2_3_factors_before_finetune = get_df(src_results_dir_palminized_not_hierarchical_2_3_factors_before_finetune)

    df_palminized_before_finetune = pd.concat([df_palminized_before_finetune_cifar100_palm_before_finetune, df_palminized_before_finetune, df_palminized_not_hierarchical_log_before_finetune, df_palminized_not_hierarchical_2_3_factors_before_finetune])

    # df_others = get_df(src_results_dir_others)
    #
    # df_mnist = get_df(src_results_dir_mnist)
    # df_mnist = df_mnist[df_mnist["--mnist"] == 1]

    # df_random_sparse_facto = pd.concat([df_mnist, df_others])
    # df_random_sparse_facto = df_random_sparse_facto.drop(columns="oar_id").drop_duplicates()

    root_output_dir = pathlib.Path("/home/luc/PycharmProjects/palmnet/reports/figures/")
    output_dir = root_output_dir / expe_path_palminized_cifar100_palm / "histogrammes"
    output_dir.mkdir(parents=True, exist_ok=True)

    # sparsity_factors = sorted(set(df_palminized["--sparsity-factor"]))
    nb_factors = set(df_palminized["--nb-factor"])

    sparsy_factors_palm = sorted(set(df_palminized["--sparsity-factor"].values))

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

    df_palminized = df_palminized.apply(pd.to_numeric, errors='coerce')
    df_palminized_before_finetune = df_palminized_before_finetune.apply(pd.to_numeric, errors='coerce')
    for dataname in dataset:
        df_data_palm = df_palminized[df_palminized[dataset[dataname]] == 1]
        df_data_palminized_before_finetune = df_palminized_before_finetune[df_palminized_before_finetune[dataset[dataname]] == 1]
        for task in tasks:

            xticks = ["2", "3", "log(min(A, B))"]
            # xticks = ["A", "B", "log(min(A, B))"]
            # xticks = [1, 2, 3]

            fig = go.Figure()

            # base model
            ############v
            if task == "finetuned_score":
                task_base = "test_accuracy_base_model"
            elif task == "nb_flop":
                task_base = "nb_flops_base_layers_conv_dense"
            elif task == "nb_param":
                task_base = "nb_param_base_layers_conv_dense"
            elif task == "compression_rate":
                pass
            else:
                raise ValueError("Unknown task {}".format(task))

            if task == "compression_rate":
                nb_param_base = df_data_palminized_before_finetune["nb_param_base_layers_conv_dense"].values.mean()
                val = 1.
            else:
                val = df_data_palminized_before_finetune[task_base].values.mean()
            fig.add_trace(
                go.Scatter(
                    x=[-1, "2", "3", "log(min(A, B))", 1],
                    y=[val, val, val, val, val],
                    mode='lines',
                    name="base model"
                ))
            # fig, ax = plt.subplots()
            # max_value_in_plot = -1
            # bar_width = 0.9 / (len(sparsity_factors)*2 + 1)

            # palminized
            ############
            if task == "finetuned_score":
                df_data_palminized = df_data_palm
                task_palminized = "finetuned_score"
            elif task == "nb_flop":
                df_data_palminized = df_data_palminized_before_finetune
                task_palminized = "nb_flops_compressed_layers_conv_dense"
            elif task == "nb_param" or task == "compression_rate":
                df_data_palminized = df_data_palminized_before_finetune
                task_palminized = "nb_param_compressed_layers_conv_dense"
            else:
                raise ValueError("Unknown task {}".format(task))

            for i, sp_fac_palm in enumerate(sparsy_factors_palm):
                df_sparsity_palm = df_data_palminized[df_data_palminized["--sparsity-factor"] == sp_fac_palm]
                for hierarchical_value in [1, 0]:
                    hierarchical_str = " H" if hierarchical_value == 1 else ""
                    df_data_palminized_hierarchical = df_sparsity_palm[df_sparsity_palm["--hierarchical"] == hierarchical_value]
                    if task == "compression_rate":
                        val = df_data_palminized_hierarchical.sort_values("--nb-factor", na_position="last")[task_palminized].values
                        val = nb_param_base / val
                    else:
                        val = df_data_palminized_hierarchical.sort_values("--nb-factor", na_position="last")[task_palminized].values

                    hls_str = "hsl({}, {}%, 60%)".format(hue_by_sparsity[sp_fac_palm], saturation_by_hier[hierarchical_value])
                    fig.add_trace(go.Bar(name=('Palm {}' + hierarchical_str).format(sp_fac_palm), x=[xticks[-1]] if hierarchical_value == 1 else xticks, y=val, marker_color=hls_str))

            title = task + " " + dataname

            fig.update_layout(barmode='group',
                              title=title,
                              xaxis_title="# Factor",
                              yaxis_title=ylabel_task[task],
                              yaxis_type=scale_tasks[task],
                              xaxis={'type': 'category'},
                              )
            fig.show()
            fig.write_image(str((output_dir / title).absolute()) + ".png")
