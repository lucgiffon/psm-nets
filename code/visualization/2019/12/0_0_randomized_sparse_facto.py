import pathlib
import pandas as pd
from palmnet.visualization.utils import get_palminized_model_and_df, get_df
import matplotlib.pyplot as plt
import numpy as np
import logging
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

tasks = {
    "nb_flop",
    "nb_param",
    "finetuned_score"
}
ylabel_task = {
    "nb_flop": "log(# Flop)",
    "nb_param": "log(# non-zero value)",
    "finetuned_score": "Accuracy"
}

scale_tasks = {
    "nb_flop": "log",
    "nb_param": "log",
    "finetuned_score": "linear"
}

if __name__ == "__main__":
    root_source_dir = pathlib.Path("/home/luc/PycharmProjects/palmnet/results/")
    expe_path_mnist = "2019/12/0_0_random_sparse_facto"
    expe_path_others = "2019/12/0_0_random_sparse_facto/bis_no_mnist"

    expe_path_palminized_mnist = "2019/11/0_0_finetune_palminized"
    expe_path_palminized_cifar10 = "2019/11/0_0_finetune_palminized/bis_cifar10/bis_bis"
    expe_path_palminized_svhn_cifar100 = "2019/11/0_0_finetune_palminized/bis_cifar100_svhn/bis_bis"
    expe_path_palminized_before_finetune = "2019/10/0_0_hierarchical_palminize"

    src_results_dir_palminized_mnist = root_source_dir / expe_path_palminized_mnist
    src_results_dir_palminized_before_finetune = root_source_dir / expe_path_palminized_before_finetune
    src_results_dir_palminized_cifar10 = root_source_dir / expe_path_palminized_cifar10
    src_results_dir_palminized_svhn_cifar100 = root_source_dir / expe_path_palminized_svhn_cifar100
    src_results_dir_mnist = root_source_dir / expe_path_mnist
    src_results_dir_others = root_source_dir / expe_path_others

    df_palminized_mnist = get_df(src_results_dir_palminized_mnist)
    df_palminized_cifar10 = get_df(src_results_dir_palminized_cifar10)
    df_palminized_cifar100_svhn = get_df(src_results_dir_palminized_svhn_cifar100)
    df_palminized = pd.concat([df_palminized_mnist, df_palminized_cifar10, df_palminized_cifar100_svhn])
    df_palminized = df_palminized.dropna(subset=["failure"])
    df_palminized = df_palminized[df_palminized["finetuned_score"] != "None"]
    df_palminized = df_palminized.drop(columns="oar_id").drop_duplicates()

    df_palminized_before_finetune = get_df(src_results_dir_palminized_before_finetune)

    df_others = get_df(src_results_dir_others)

    df_mnist = get_df(src_results_dir_mnist)
    df_mnist = df_mnist[df_mnist["--mnist"] == 1]

    df = pd.concat([df_mnist, df_others])
    df = df.drop(columns="oar_id").drop_duplicates()

    sparsity_factors = sorted(set(df["--sparsity-factor"]))
    nb_factors = set(df["--nb-factor"])

    sparsy_factors_palm = sorted(set(df_palminized["--sparsity-factor"].values))

    df = df.apply(pd.to_numeric, errors='coerce')
    for dataname in dataset:
        df_data = df[df[dataset[dataname]] == 1]
        df_data_palm = df_palminized[df_palminized[dataset[dataname]] == 1]
        df_data_palminized_before_finetune = df_palminized_before_finetune[df_palminized_before_finetune[dataset[dataname]] == 1]
        for task in tasks:
            fig, ax = plt.subplots()
            max_value_in_plot = -1
            bar_width = 0.9 / (len(sparsity_factors)*2 + 1)
            # random sparse factorization
            #############################
            for i, sp_fac in enumerate(sparsity_factors):
                df_sparsity = df_data[df_data["--sparsity-factor"] == sp_fac]
                for j, no_perm in enumerate([1, 0]):
                    no_perm_str = "No P" if no_perm==1 else ""
                    df_perm = df_sparsity[df_sparsity["--no-permutation"] == no_perm]
                    finetune_score_values = df_perm.sort_values("--nb-factor", na_position="last")[task].values
                    ax.bar(np.arange(len(nb_factors)) + bar_width*2*i + bar_width*j, finetune_score_values,
                           width=bar_width, label='{} {}'.format(sp_fac, no_perm_str), zorder=10, color=color_bars_sparsity[sp_fac],
                           hatch=hatch_bars_permutation[no_perm])
                    max_value_in_plot = max(max_value_in_plot, max(finetune_score_values))

            # palminized
            ############
            start_x_pos = len(nb_factors) +  bar_width * 2 * len(sparsity_factors) - 1
            if task == "finetuned_score":
                df_data_palminized = df_data_palm
                task_palminized = "finetuned_score"
            elif task == "nb_flop":
                df_data_palminized = df_data_palminized_before_finetune
                task_palminized = "nb_flops_compressed_layers_conv_dense"
            elif task == "nb_param":
                df_data_palminized = df_data_palminized_before_finetune
                task_palminized = "nb_param_compressed_layers_conv_dense"
            else:
                raise ValueError("Unknown task {}".format(task))

            for i, sp_fac_palm in enumerate(sparsy_factors_palm):
                val = df_data_palminized[df_data_palminized["--sparsity-factor"] == sp_fac_palm][task_palminized].values.mean() if sp_fac_palm in df_data_palminized["--sparsity-factor"].values else 0
                ax.bar(start_x_pos + i*bar_width, val, width=bar_width, label='Palm {}'.format(sp_fac_palm), zorder=10, color=color_bars_sparsity[sp_fac_palm], hatch="X")
                max_value_in_plot = max(max_value_in_plot, val)

            # base model
            ############v
            if task == "finetuned_score":
                task_base = "test_accuracy_base_model"
            elif task == "nb_flop":
                task_base = "nb_flops_base_layers_conv_dense"
            elif task == "nb_param":
                task_base = "nb_param_base_layers_conv_dense"
            else:
                raise ValueError("Unknown task {}".format(task))

            val = df_data_palminized_before_finetune[task_base].values.mean()
            max_value_in_plot = max(max_value_in_plot, val)
            start_x_pos_base = 0
            end_x_pos_base = start_x_pos + len(sparsy_factors_palm) * bar_width
            ax.plot(np.linspace(start_x_pos_base, end_x_pos_base, num=2), np.ones(2)*val, color="k", label="base")


            plt.xlabel("# Factor")
            plt.ylabel(ylabel_task[task])
            plt.yscale(scale_tasks[task])
            ax.set_xticks([0, 1, 2])
            ax.set_xticklabels([2, 3, "log(min(A, B))"])
            plt.legend(ncol=4)
            ax.set_ylim(top=max_value_in_plot+0.2*max_value_in_plot)
            plt.title(task + " " + dataname)
            plt.show()

    root_output_dir = pathlib.Path("/home/luc/PycharmProjects/palmnet/reports/figures/")
    output_dir = root_output_dir / expe_path_mnist / "histogrammes"
    output_dir.mkdir(parents=True, exist_ok=True)
