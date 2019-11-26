import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pathlib
from palmnet.visualization.utils import get_dct_result_files_by_root, build_df


def get_palminized_model_and_df(path):
    src_result_dir = pathlib.Path(path)
    dct_output_files_by_root = get_dct_result_files_by_root(src_results_dir=src_result_dir, old_filename_objective=True)

    col_to_delete = []

    dct_oarid_palminized_model = {}
    for root_name, job_files in dct_output_files_by_root.items():
        objective_file_path = src_result_dir / job_files["palminized_model"]
        loaded_model = pickle.load(open(objective_file_path, 'rb'))
        dct_oarid_palminized_model[root_name] = loaded_model

    df_results = build_df(src_result_dir, dct_output_files_by_root, col_to_delete)
    return dct_oarid_palminized_model, df_results




datasets = {
    "Cifar10": "--cifar10",
    "Cifar100": "--cifar100",
    "Mnist": "--mnist",
    "SVHN": "--svhn"
}

tasks = {"Number of flops": ("nb_flops_base_layers_conv_dense", "nb_flops_compressed_layers_conv_dense"),
         "Number of parameters": ("nb_param_base_layers_conv_dense", "nb_param_compressed_layers_conv_dense"),
         "Test accuracy": ("test_accuracy_base_model", "test_accuracy_compressed_model"),
         "Test loss": ("test_loss_base_model", "test_loss_compressed_model")
}

scales = {
"Number of flops": "linear",
"Number of parameters": "linear",
"Test accuracy": "linear",
"Test loss": "linear"
}

if __name__ == "__main__":
    root_source_dir = pathlib.Path("/home/luc/PycharmProjects/palmnet/results/")
    expe_path = "2019/10/0_0_hierarchical_palminize"

    src_results_dir = root_source_dir / expe_path

    root_output_dir = pathlib.Path("/home/luc/PycharmProjects/qalm_qmeans/reports/figures/")
    output_dir = root_output_dir / expe_path / "histogrammes"

    palminized_model, df = get_palminized_model_and_df(src_results_dir)
    for dataname in datasets:
        df_dataset = df[df[datasets[dataname]]]
        for task_name, (task_base, task_compressed) in tasks.items():
            fig, ax = plt.subplots()
            values_base = df_dataset[task_base].values
            values_compressed = df_dataset[task_compressed].values
            sparsy_vals = df_dataset["--sparsity-factor"].values
            ax.bar(sparsy_vals, values_compressed, width=0.2,
                   label='Compressed', zorder=10, color="g")
            ax.bar(sparsy_vals + 0.2, values_base, width=0.2,
                   label='Base', zorder=10, color="r")

            plt.yscale(scales[task_name])
            title = "{} {}".format(dataname, task_name)
            plt.title(title)
            plt.legend()
            plt.savefig(output_dir / title.replace(" ", "_").replace(":", ""))
            plt.show()

