from palmnet.visualization.utils import get_df
import pathlib
import pandas as pd
import numpy as np
from collections import defaultdict


def get_df_from_expe_path(expe_path):
    src_dir = root_source_dir / expe_path
    df = get_df(src_dir)
    df = df.assign(results_dir=[str(src_dir.absolute())] * len(df))
    df = df.rename(columns={"--tol": "--delta-threshold"})

    return df

def cast_to_num(df):
    for col in df.columns.difference(columns_not_to_num):
        if col in df.columns.values:
            df.loc[:, col] = df.loc[:, col].apply(pd.to_numeric, errors='coerce')
    return df


if __name__ == "__main__":
    root_source_dir = pathlib.Path("/home/luc/PycharmProjects/palmnet/results/")
    expe_path = "2020/07/0_1_image_size"

    lst_path_finetune = [
        "2020/07/0_1_image_size"
    ]

    columns_not_to_num = ['hash', 'output_file_csvcbprinter', "--use-clr",
                          "--input-dir", "input_model_path", "output_file_csvcvprinter",
                          "output_file_finishedprinter", "output_file_layerbylayer",
                          "output_file_modelprinter", "output_file_notfinishedprinter",
                          "output_file_resprinter", "output_file_tensorboardprinter", "results_dir"]

    # df_finetune = pd.concat(list(map(get_df_from_expe_path, lst_path_finetune)))
    df = get_df_from_expe_path(lst_path_finetune[0])
    df = df.dropna(subset=["failure"])
    df = df[df["failure"] == False]
    df = df.drop(columns="oar_id").drop_duplicates()
    df = cast_to_num(df)

    root_output_dir = pathlib.Path("/home/luc/PycharmProjects/palmnet/results/processed/")
    output_dir = root_output_dir / expe_path
    output_dir.mkdir(parents=True, exist_ok=True)

    length_df = len(df)
    dct_attributes = defaultdict(lambda: [])

    for idx, (_, row) in enumerate(df.iterrows()):
        if row["--cifar10"]:
            dct_attributes["dataset"].append("cifar10")
        elif row["--cifar100"]:
            dct_attributes["dataset"].append("cifar100")
        elif row["--mnist"]:
            dct_attributes["dataset"].append("mnist")
        elif row["--svhn"]:
            dct_attributes["dataset"].append("svhn")
        else:
            raise ValueError("Unknown dataset")

        if row["--cifar100-vgg19"] or row["--cifar10-vgg19"] or row["--svhn-vgg19"]:
            dct_attributes["model"].append("vgg19")
        elif row["--mnist-lenet"]:
            dct_attributes["model"].append("lenet")
        elif row["--mnist-500"]:
            dct_attributes["model"].append("fc500")
        elif row["--cifar100-resnet20"]:
            dct_attributes["model"].append("resnet20")
        elif row["--cifar100-resnet50"]:
            dct_attributes["model"].append("resnet50")
        elif row["--cifar100-resnet20-new"]:
            dct_attributes["model"].append("resnet20")
        elif row["--cifar100-resnet50-new"]:
            dct_attributes["model"].append("resnet50")
        else:
            raise ValueError("Unknown model")

        if row["tensortrain"]:
            dct_attributes["method"].append("tensortrain")
        elif row["random"]:
            dct_attributes["method"].append("random")
        else:
            raise NotImplementedError

        # youtube
        dct_attributes["rank"].append(float(row["--rank-value"]) if not np.isnan(row["--rank-value"]) else None)
        dct_attributes["order"].append(float(row["--order"]) if not np.isnan(row["--order"]) else None)

        dct_attributes["max_image_size"].append(int(row["max_item_size"]))

    df_results = pd.DataFrame.from_dict(dct_attributes)
    df_results.to_csv(output_dir / "results.csv")