import pathlib
import pandas as pd
from pprint import pprint
import numpy as np

def get_results():
    results_path_baselines = "2020/07/0_1_image_size"
    src_results_path_tucker = root_source_dir / results_path_baselines / "results.csv"

    df = pd.read_csv(src_results_path_tucker, header=0)
    df = df.fillna("None")

    return df

if __name__ == "__main__":
    root_source_dir = pathlib.Path("/home/luc/PycharmProjects/palmnet/results/processed")

    df = get_results()

    dict_algo_data_perf = dict()
    datasets = set(df["dataset"].values)
    for dataname in datasets:
        df_data = df[df["dataset"] == dataname]
        df_model_values = set(df_data["model"].values)
        dict_algo_data_perf[dataname] = dict()

        for modelname in df_model_values:
            df_model_palm = df_data[df_data["model"] == modelname]
            dict_algo_data_perf[dataname][modelname] = dict()

            for i, row in df_model_palm.iterrows():
                if row["method"] == "tensortrain":
                    order = row["order"]
                    rank = row["rank"]

                    str_tt = f"Tensortrain R={int(rank)} K={int(order)}"
                    dict_algo_data_perf[dataname][modelname][str_tt] = row["max_image_size"]
                else:
                    dict_algo_data_perf[dataname][modelname]["Autres"] = row["max_image_size"]

    pprint(dict_algo_data_perf)

    array_str = np.empty((5, 7), dtype=object)
    lst_elm = [
        'Autres',
        'Tensortrain R=6 K=4',
        'Tensortrain R=10 K=4',
        'Tensortrain R=14 K=4',
    ]

    lst_base_models = [
        'mnist_lenet',
        "svhn_vgg19",
        "cifar10_vgg19",
        "cifar100_vgg19",
        "cifar100_resnet20",
        "cifar100_resnet50"
    ]
    dct_str_fancy = {
        'mnist_lenet': "MNIST Lenet",
        "svhn_vgg19": "SVHN Vgg19",
        "cifar10_vgg19": "Cifar10 Vgg19",
        "cifar100_vgg19": "Cifar100 Vgg19",
        "cifar100_resnet20": "Cifar100 Resnet20",
        "cifar100_resnet50": "Cifar100 Resnet50",
    }

    for i_d, dataname in enumerate(dict_algo_data_perf):
        for i_m, base_model in enumerate(dict_algo_data_perf[dataname]):
            for i_c, compression in enumerate(dict_algo_data_perf[dataname][base_model]):
                perf = dict_algo_data_perf[dataname][base_model][compression]
                idx_line = 1 + lst_elm.index(compression)
                str_fancy_compression = compression
                array_str[idx_line, 0] = str_fancy_compression

                str_base_model = dataname + "_" + base_model
                str_fancy_base_model = dct_str_fancy[str_base_model]
                idx_col = 1 + lst_base_models.index(str_base_model)
                array_str[idx_line, idx_col] = f"{int(perf)}"
                array_str[0, idx_col] = str_fancy_base_model

                print(f"{int(perf)}")
                print(array_str[idx_line, idx_col])


    array_str[0, 0] = "{}"
    print(array_str)

    print(r"\toprule")
    for i, line in enumerate(array_str):
        iter_line = line
        line_to_print = "&".join(iter_line) + r"\\"
        print(line_to_print)
        if i == 0 or i == 1 or (i - 1) % 3 == 0:
            print(r"\midrule")
    print(r"\bottomrule")