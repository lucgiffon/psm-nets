import pathlib
import pandas as pd
from palmnet.visualization.utils import get_palminized_model_and_df, get_df
import matplotlib.pyplot as plt
import numpy as np
import logging
import plotly.graph_objects as go
import plotly.io as pio

mpl_logger = logging.getLogger('matplotlib')
mpl_logger.setLevel(logging.ERROR)
pio.templates.default = "plotly_white"

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


def get_palm_results():
    results_path = "2020/07/11_12_finetune_sparse_facto_fix_replicates"
    # results_path_2 = "2020/04/9_10_finetune_palminized_no_useless"

    src_results_path = root_source_dir / results_path / "results.csv"
    # src_results_path_2 = root_source_dir / results_path_2 / "results.csv"

    df = pd.read_csv(src_results_path, header=0)
    # df_2 = pd.read_csv(src_results_path_2, header=0)
    # df = pd.concat([df, df_2])
    df = df.fillna("None")
    df = df.drop(columns=["Unnamed: 0", "idx-expe"]).drop_duplicates()

    # df = df[df["keep-last-layer"] == 0]
    # df = df[df["use-clr"] == 1]

    # df = df.assign(**{"only-dense": False, "keep-first-layer": False})

    return df


def get_baseline_results():
    results_path_baselines = "2020/05/10_11_compression_baselines_full"
    src_results_path_tucker = root_source_dir / results_path_baselines / "results.csv"

    df_baselines = pd.read_csv(src_results_path_tucker, header=0)
    df_baselines = df_baselines.fillna("None")

    # df_tucker_tt = df_tucker_tt.assign(**{"only-dense": False, "use-pretrained": False})

    # df_tucker_tt = df_tucker_tt[df_tucker_tt["compression"] == "tucker"]
    return df_baselines


if __name__ == "__main__":
    root_source_dir = pathlib.Path("/home/luc/PycharmProjects/palmnet/results/processed")


    SHOW_FAUST = False
    SHOW_KEEP_FIRST_ONLY = True
    SHOW_PRETRAINED_ONLY = True

    results_path = "2020/05/9_10_finetune_sparse_facto_not_log_all_seeds"

    # df_faust = get_faust_results()

    df_baselines = get_baseline_results()
    # df_tt = get_tensortrain_results()
    # df_tucker_tt_only_dense = get_tucker_tensortrain_only_denseresults()
    # df_tucker_tt = pd.concat([df_tucker, df_tt, df_tucker_tt_only_dense])

    df_palm = get_palm_results()
    # df_palm_bis = get_palm_results_only_dense_keep_first()
    # df_palm = pd.concat([df_palm, df_palm_bis])

    # ONLY_DENSE = False
    # df_tucker_tt = df_tucker_tt[df_tucker_tt["only-dense"] == ONLY_DENSE]
    # df_palm = df_palm[df_palm["only-dense"] == ONLY_DENSE]


    root_output_dir = pathlib.Path("/home/luc/PycharmProjects/palmnet/reports/figures/")
    output_dir = root_output_dir / results_path / "histogrammes"
    output_dir.mkdir(parents=True, exist_ok=True)

    # sparsity_factors = sorted(set(df_palminized["--sparsity-factor"]))
    # nb_factors = set(df_faust["nb-factor"].values)

    show_random = True

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
        0: 30
    }
    lum_by_keep = {
        1: 40,
        0: 50
    }

    dct_symbol = {
        "FAUST Q=2": "square",
        "FAUST Q=3": "diamond",
        "FAUST Q=None": "square-x",
        "FAUST Q=None H": "star-square",
        "PYQALM Q=2": "square-open",
        "PYQALM Q=3": "diamond-open",
        "PYQALM Q=None": "hash-open",
        "PYQALM Q=None H": "star-square-open",
        "PYQALM Q=2 -1": "square",
        "PYQALM Q=3 -1": "diamond",
        "PYQALM Q=None -1": "hash",
        "PYQALM Q=None H -1": "star-square",
        "Random Q=2 -1": "square",
        "Random Q=3 -1": "diamond",
        "Random Q=None -1": "hash",
        "Random Q=None H -1": "star-square",
        "PYQALM Q=2 -1 M": "square",
        "PYQALM Q=3 -1 M": "diamond",
        "PYQALM Q=None -1 M": "hash",
        "PYQALM Q=None H -1 M": "star-square",
        "Base": "x",
        "Tucker": "circle",
        "Tucker -1": "circle",
        "TT": "triangle-up",
        "TT -1": "triangle-up",
        "TT -1 pretrained": "triangle-up-open",
        "Magnitude": "hexagon"
    }

    dct_colors = {
        "PALM K=2": "dodgerblue",
        "PALM K=3": "darkorchid",
        "PALM K=4": "green",
        # "PALM K=6": "aqua",
        "PALM K=8": "cadetblue",
        "PALM K=14": "aqua",
        "Random K=2": "dodgerblue",
        "Random K=3": "darkorchid",
        "Random K=4": "green",
        # "PALM K=6": "aqua",
        "Random K=8": "cadetblue",
        "Random K=14": "aqua",
        # "PALM K=10": "aqua",
        "TT R=2": "orange",
        "TT R=6": "gold",
        "TT R=10": "red",
        "TT R=12": "darkred",
        "TT R=14": "indianred",
        "Magnitude 0.95": "red",
        "Magnitude 0.98": "gold",
        "Magnitude ": "red",
        "Base": "grey",
        "Tucker": "pink",
        "Tucker + Low Rank 10%": "orange",
        "Tucker + Low Rank 20%": "gold",
        "Tucker + Low Rank 30%": "red"
    }

    SIZE_MARKERS = 15
    WIDTH_MARKER_LINES = 2
    dict_algo_data_perf = dict()
    datasets = set(df_palm["dataset"].values)
    for dataname in datasets:
        df_data_palm = df_palm[df_palm["dataset"] == dataname]
        df_baselines_data = df_baselines[df_baselines["dataset"] == dataname]
        df_model_values = set(df_data_palm["model"].values)
        dict_algo_data_perf[dataname] = dict()
        for modelname in df_model_values:
            df_model_palm = df_data_palm[df_data_palm["model"] == modelname]
            df_baselines_model = df_baselines_data[df_baselines_data["model"] == modelname]
            dict_algo_data_perf[dataname][modelname] = dict()
            fig = go.Figure()

            base_score_mean = None
            base_nb_param_mean = None
            base_score_std = None
            base_nb_param_std = None

            # PALM results
            ##############
            palm_algo = "PYQALM"
            groupby_col = ['hierarchical', 'nb-factor', 'sparsity-factor', 'keep-first-layer', 'only-mask']
            gp_by = df_model_palm.groupby(groupby_col)
            for names, group in gp_by:
                dct_group_by_vals = dict(zip(groupby_col, names))
                hierarchical_value = dct_group_by_vals["hierarchical"]
                str_hierarchical = ' H' if hierarchical_value is True else ''
                try:
                    nb_factor = int(dct_group_by_vals["nb-factor"])
                except:
                    nb_factor = None

                sparsity_factor = int(dct_group_by_vals["sparsity-factor"])
                if sparsity_factor != 14 and sparsity_factor != 2: continue

                keep_first = dct_group_by_vals["keep-first-layer"]
                str_keep_first = ' -1' if keep_first is True else ''

                only_mask = dct_group_by_vals["only-mask"]
                # if only_mask: continue

                str_only_mask = " M" if only_mask is True else ""

                name_trace = f"{palm_algo} Q={nb_factor} K={sparsity_factor}{str_only_mask}"

                finetuned_score_mean = group["finetuned-score"].mean()
                finetuned_score_std = group["finetuned-score"].std()
                nb_param_mean = group["nb-param-compressed-total"].mean()
                nb_param_std = group["nb-param-compressed-total"].std()

                base_score_tmp_mean = group["base-model-score"].mean()
                base_score_tmp_std = group["base-model-score"].std()
                # assert base_score_mean == base_score_tmp_mean or base_score_mean is None
                # assert base_score_std == base_score_tmp_std or base_score_std is None
                base_nb_param_tmp_mean = group["nb-param-base-total"].mean()
                base_nb_param_tmp_std = group["nb-param-base-total"].std()
                # assert base_nb_param_mean == base_nb_param_tmp_mean or base_nb_param_mean is None
                # assert base_nb_param_std == base_nb_param_tmp_std or base_nb_param_std is None
                base_score_mean = base_score_tmp_mean
                base_nb_param_mean = base_nb_param_tmp_mean
                base_score_std = base_score_tmp_std
                base_nb_param_std = base_nb_param_tmp_std

                # nb_param_mean
                # finetuned_score_std
                dict_algo_data_perf[dataname][modelname][name_trace] = finetuned_score_mean

            #############
            # base data #
            #############
            # base_score_std
            # base_nb_param_mean
            dict_algo_data_perf[dataname][modelname]["Base"] = base_score_mean

            ###############
            # random data #
            ###############
            df_random = df_baselines_model[df_baselines_model["compression"] == "random"]

            groupby_col = ['nb-fac', 'sparsity-factor', 'keep-first-layer']
            gp_by = df_random.groupby(groupby_col)
            for names, group in gp_by:
                dct_group_by_vals = dict(zip(groupby_col, names))
                try:
                    nb_factor = int(dct_group_by_vals["nb-fac"])
                except:
                    nb_factor = None


                sparsity_factor = int(dct_group_by_vals["sparsity-factor"])
                if sparsity_factor == 10:
                    continue

                if sparsity_factor != 14 and sparsity_factor != 2: continue

                keep_first = dct_group_by_vals["keep-first-layer"]
                str_keep_first = ' -1' if keep_first is True else ''

                name_trace = f"Random Q={nb_factor} K={sparsity_factor}"

                finetuned_score_mean = group["finetuned-score"].mean()
                nb_param_mean = group["nb-param-compressed-total"].mean()
                finetuned_score_std = group["finetuned-score"].std()
                nb_param_std = group["nb-param-compressed-total"].std()

                # nb_param_mean
                # finetuned_score_std
                dict_algo_data_perf[dataname][modelname][name_trace] = finetuned_score_mean




            title = "Performance = f(# Param); " + dataname + " " + modelname

    from pprint import pprint
    pprint(dict_algo_data_perf)

    array_str = np.empty((14, 7), dtype=object)
    lst_elm = [
'Base',
'PYQALM Q=2 K=2',
'PYQALM Q=2 K=2 M',
'Random Q=2 K=2',
'PYQALM Q=2 K=14',
'PYQALM Q=2 K=14 M',
'Random Q=2 K=14',
'PYQALM Q=3 K=2',
'PYQALM Q=3 K=2 M',
'Random Q=3 K=2',
'PYQALM Q=3 K=14',
'PYQALM Q=3 K=14 M',
'Random Q=3 K=14',
]
    dict_str_fancy_compression = {
    'Base': "Base",
    'PYQALM Q=2 K=2': "Sparse Facto. Q=2 K=2",
    'PYQALM Q=2 K=2 M': "Sparse Facto. re-init. Q=2 K=2",
    'Random Q=2 K=2': "Sparse Facto. aléatoire Q=2 K=2",
    'PYQALM Q=2 K=14': "Sparse Facto. Q=2 K=14",
    'PYQALM Q=2 K=14 M': "Sparse Facto. re-init. Q=2 K=14",
    'Random Q=2 K=14': "Sparse Facto. aléatoire Q=2 K=14",
    'PYQALM Q=3 K=2': "Sparse Facto. Q=3 K=2",
    'PYQALM Q=3 K=2 M': "Sparse Facto. re-init. Q=3 K=2",
    'Random Q=3 K=2': "Sparse Facto. aléatoire Q=3 K=2",
    'PYQALM Q=3 K=14': "Sparse Facto. Q=3 K=14",
    'PYQALM Q=3 K=14 M': "Sparse Facto. re-init. Q=3 K=14",
    'Random Q=3 K=14': "Sparse Facto. aléatoire Q=3 K=14",
    }

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
                str_fancy_compression = dict_str_fancy_compression[compression]
                array_str[idx_line, 0] = str_fancy_compression

                str_base_model = dataname + "_" + base_model
                str_fancy_base_model = dct_str_fancy[str_base_model]
                idx_col = 1 + lst_base_models.index(str_base_model)
                array_str[idx_line, idx_col] = f"{perf:0.2f}"
                array_str[0, idx_col] = str_fancy_base_model

                print(f"{perf:0.2f}")
                print(array_str[idx_line, idx_col])

    array_str[0,0] = "{}"
    print(array_str)

    dct_idx_col_sorted = dict()

    for idx_col in range(array_str.shape[1]):
        if idx_col == 0: continue
        try:
            float_col = set(array_str[1:, idx_col].astype(float))
            float_col = {x for x in float_col if x == x}
            sorted_col = sorted(float_col)[::-1][:2]
            dct_idx_col_sorted[idx_col] = sorted_col

        except:
            continue



    print(r"\toprule")
    for i, line in enumerate(array_str):

        if i> 0:
            iter_line = []
            for idx_col, elm in enumerate(line):
                if idx_col in dct_idx_col_sorted:
                    if elm is None:
                        elm = "N/A"
                    elif np.float(elm) ==  dct_idx_col_sorted[idx_col][0]:
                        elm = "\\textbf{{{}}}".format(elm)
                    elif np.float(elm) ==  dct_idx_col_sorted[idx_col][1]:
                        elm = "\\underline{{{}}}".format(elm)

                    iter_line.append(elm)
                else:
                    iter_line.append(elm)
        else:
            iter_line = line
        line_to_print = "&".join(iter_line) + r"\\"
        print(line_to_print)
        if i == 0 or i == 1 or (i-1) %3 == 0:
            print(r"\midrule")
    print(r"\bottomrule")


