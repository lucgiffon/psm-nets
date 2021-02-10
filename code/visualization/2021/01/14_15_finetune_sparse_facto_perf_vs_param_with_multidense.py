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
    # results_path = "2020/05/9_10_finetune_sparse_facto_not_log_all_seeds"
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

def get_faust_results():
    results_path = "2020/05/3_4_finetune_faust_no_hierarchical_only_cifar_mnist"

    src_results_path = root_source_dir / results_path / "results.csv"

    df = pd.read_csv(src_results_path, header=0)
    df = df.fillna("None")
    df = df[df["hierarchical"] == False]
    df = df.drop(columns=["Unnamed: 0", "idx-expe"]).drop_duplicates()

    df = df[df["keep-last-layer"] == 0]
    df = df.assign(**{"only-dense": False, "keep-first-layer": False})
    return df

def get_baseline_results():
    results_path_baselines = "2021/01/14_15_compression_baselines_full"
    src_results_path_tucker = root_source_dir / results_path_baselines / "results.csv"

    df_baselines = pd.read_csv(src_results_path_tucker, header=0)
    df_baselines = df_baselines.fillna("None")

    # df_tucker_tt = df_tucker_tt.assign(**{"only-dense": False, "use-pretrained": False})

    # df_tucker_tt = df_tucker_tt[df_tucker_tt["compression"] == "tucker"]
    return df_baselines

def get_tensortrain_results():
    results_path_tucker = "2020/05/2_3_compression_tensortrain"
    src_results_path_tucker = root_source_dir / results_path_tucker / "results.csv"

    df_tucker_tt = pd.read_csv(src_results_path_tucker, header=0)
    df_tucker_tt = df_tucker_tt.fillna("None")

    # df_tucker_tt = df_tucker_tt[df_tucker_tt["use-pretrained"] == True]
    # df_tucker_tt = df_tucker_tt[df_tucker_tt["only-dense"] == False]

    return df_tucker_tt

def get_tucker_tensortrain_only_denseresults():
    results_path_tucker = "2020/05/2_3_compression_tucker_tensortrain_only_dense"
    src_results_path_tucker = root_source_dir / results_path_tucker / "results.csv"

    df_tucker_tt = pd.read_csv(src_results_path_tucker, header=0)
    df_tucker_tt = df_tucker_tt.fillna("None")

    # df_tucker_tt = df_tucker_tt[df_tucker_tt["use-pretrained"] == True]
    # df_tucker_tt = df_tucker_tt[df_tucker_tt["only-dense"] == True]

    return df_tucker_tt

def get_palm_results_only_dense_keep_first():
    results_path = "2020/05/5_6_finetune_sparse_facto_no_hierarchical_keep_first_layer_only_dense"

    src_results_path = root_source_dir / results_path / "results.csv"

    df = pd.read_csv(src_results_path, header=0)
    df = df.fillna("None")
    df = df.drop(columns=["Unnamed: 0", "idx-expe"]).drop_duplicates()

    # df = df[df["only-dense"] == False]

    return df

if __name__ == "__main__":
    root_source_dir = pathlib.Path("/home/luc/PycharmProjects/palmnet/results/processed")


    SHOW_FAUST = False
    SHOW_KEEP_FIRST_ONLY = True
    SHOW_PRETRAINED_ONLY = True


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
    results_path = "2021/01/12_13_compression_baselines_full"
    output_dir = root_output_dir / results_path / "histogrammes"
    output_dir.mkdir(parents=True, exist_ok=True)

    # sparsity_factors = sorted(set(df_palminized["--sparsity-factor"]))
    # nb_factors = set(df_faust["nb-factor"].values)

    show_random = False

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
        "IP": "hexagon",
        "HP": "star",
        "Multidense Doubleepochs:True": "asterisk",
        "Multidense Doubleepochs:False": "hash"
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
        "IP 0.95": "red",
        "IP 0.98": "gold",
        "HP 0.95": "red",
        "HP 0.98": "gold",
        "IP ": "red",
        "Base": "grey",
        "Tucker": "pink",
        "Tucker + Low Rank 10%": "orange",
        "Tucker + Low Rank 20%": "gold",
        "Tucker + Low Rank 30%": "red",
        "Multidense K=0.999": "pink",
        "Multidense K=0.99": "orange",
        "Multidense K=0.98": "gold",
        "Multidense K=0.0": "red",
    }

    def store_value_in_dict_algo_data_perf_2(current_delta_score, delta_score, current_compression, compression):
        if delta_score > 0.05:
            return False
        elif delta_score < current_delta_score:
            return False
        elif compression < current_compression:
            return False
        else:
            return True

    SIZE_MARKERS = 15
    WIDTH_MARKER_LINES = 2
    dict_algo_data_perf = dict()
    dict_algo_data_perf_2 = dict()
    datasets = set(df_palm["dataset"].values)
    for dataname in datasets:
        print(dataname)
        df_data_palm = df_palm[df_palm["dataset"] == dataname]
        df_baselines_data =  df_baselines[df_baselines["dataset"] == dataname]
        dataname = dataname.upper()
        df_model_values = set(df_data_palm["model"].values)
        dict_algo_data_perf[dataname] = dict()
        for modelname in df_model_values:
            print(modelname)
            df_model_palm = df_data_palm[df_data_palm["model"] == modelname]
            df_baselines_model = df_baselines_data[df_baselines_data["model"] == modelname]
            modelname = modelname.capitalize()
            dict_algo_data_perf_2[dataname+" "+modelname] = dict()

            # param-compression-rate-total
            fig = go.Figure()

            base_score_mean = None
            base_nb_param_mean = None
            base_score_std = None
            base_nb_param_std = None

            # PALM results
            ##############
            palm_algo = "PSM"
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


                keep_first = dct_group_by_vals["keep-first-layer"]
                str_keep_first = ' -1' if keep_first is True else ''

                only_mask = dct_group_by_vals["only-mask"]
                if only_mask: continue
                str_only_mask = " M" if only_mask is True else ""

                name_trace = f"{palm_algo} Q={nb_factor} K={sparsity_factor}"

                finetuned_score_mean = group["finetuned-score"].mean()
                finetuned_score_std = group["finetuned-score"].std()
                nb_param_mean = group["nb-param-compressed-total"].mean()
                nb_param_std = group["nb-param-compressed-total"].std()

                compression_rate_mean = group["param-compression-rate-total"].mean()
                base_score_tmp_mean = group["base-model-score"].mean()

                print(name_trace, end=" ")
                print("score, rate: ({:.2f}, {:.2f})".format(finetuned_score_mean, compression_rate_mean), end="; ")
                print("base score: {:.2f}".format(base_score_tmp_mean), end="; ")
                delta = base_score_tmp_mean - finetuned_score_mean
                print("delta: {:.2f}".format(delta))
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

                dict_algo_data_perf[dataname][name_trace] = finetuned_score_mean

                current_tuple = dict_algo_data_perf_2[dataname+" "+modelname].get("PSM network", (0, -1))
                if store_value_in_dict_algo_data_perf_2(current_tuple[0], delta, current_tuple[1], compression_rate_mean):
                    dict_algo_data_perf_2[dataname+" "+modelname]["PSM network"] = (delta, compression_rate_mean)
                else:
                    pass

                fig.add_trace(
                    go.Scatter(
                        x=[nb_param_mean],
                        y=[finetuned_score_mean],
                        mode='markers',
                        name=name_trace,
                        hovertext=name_trace,
                        legendgroup=f"{palm_algo} K={sparsity_factor}{str_only_mask}",
                        marker=dict(
                            color=dct_colors[f"PALM K={sparsity_factor}"],
                            symbol=dct_symbol[f"PYQALM Q={nb_factor}{str_hierarchical}{str_keep_first}{str_only_mask}"],
                            size=SIZE_MARKERS,
                            line=dict(
                                color='Black',
                                width=WIDTH_MARKER_LINES
                            )
                        ),
                        error_y=dict(
                            type='data',  # value of error bar given in data coordinates
                            array=[finetuned_score_std],
                            visible=True),
                        # error_x=dict(
                        #     type='data',  # value of error bar given in data coordinates
                        #     array=[nb_param_std],
                        #     visible=True)
                    ))

            #############
            # base data #
            #############
            fig.add_trace(
                go.Scatter(
                    x=[base_nb_param_mean],
                    y=[base_score_mean],
                    mode='markers',
                    name="Base",
                    hovertext="Base",
                    legendgroup=f"Base",
                    marker=dict(
                        color=dct_colors[f"Base"],
                        symbol=dct_symbol[f"Base"],
                        size=SIZE_MARKERS,
                        line=dict(
                            color='Black',
                            width=WIDTH_MARKER_LINES,
                        ))
                    ,
                    error_y=dict(
                        type='data',  # value of error bar given in data coordinates
                        array=[base_score_std],
                        visible=True),
                ))

            ###############
            # tucker data #
            ###############
            df_tucker = df_baselines_model[df_baselines_model["compression"] == "tucker"]

            groupby_col = ['rank-percentage-dense', 'keep-first-layer']
            gp_by = df_tucker.groupby(groupby_col)
            for names, group in gp_by:
                dct_group_by_vals = dict(zip(groupby_col, names))
                keep_first = dct_group_by_vals["keep-first-layer"]
                str_keep_first = ' -1' if keep_first is True else ''
                if SHOW_KEEP_FIRST_ONLY and not keep_first:
                    continue
                try:
                    rank_percentage = int(float(dct_group_by_vals["rank-percentage-dense"]) * 100)
                except:
                    rank_percentage = None

                str_percentage = f'-SVD {rank_percentage}%' if rank_percentage is not None else ''
                name_trace = f"Tucker{str_percentage}"
                str_percentage = f' + Low Rank {rank_percentage}%' if rank_percentage is not None else ''

                finetuned_score_mean = group["finetuned-score"].mean()
                nb_param_mean = group["nb-param-compressed-total"].mean()
                finetuned_score_std = group["finetuned-score"].std()
                nb_param_std = group["nb-param-compressed-total"].std()
                dict_algo_data_perf[dataname][name_trace] = finetuned_score_mean
                compression_rate_mean = group["param-compression-rate-total"].mean()
                base_score_tmp_mean = group["base-model-score"].mean()

                print(name_trace, end=" ")
                print("score, rate: ({:.2f}, {:.2f})".format(finetuned_score_mean, compression_rate_mean), end="; ")
                print("base score: {:.2f}".format(base_score_tmp_mean), end="; ")
                delta = base_score_tmp_mean - finetuned_score_mean
                print("delta: {:.2f}".format(delta))
                current_tuple = dict_algo_data_perf_2[dataname+" "+modelname].get("Tucker-SVD", (0, -1))
                if store_value_in_dict_algo_data_perf_2(current_tuple[0], delta, current_tuple[1], compression_rate_mean):
                    dict_algo_data_perf_2[dataname+" "+modelname]["Tucker-SVD"] = (delta, compression_rate_mean)
                else:
                    pass
                fig.add_trace(
                    go.Scatter(
                        x=[nb_param_mean],
                        y=[finetuned_score_mean],
                        mode='markers',
                        name=name_trace,
                        hovertext=name_trace,
                        legendgroup=f"Tucker{str_percentage}",
                        marker=dict(
                                    color=dct_colors[f"Tucker{str_percentage}"],
                                    symbol=dct_symbol[f"Tucker{str_keep_first}"],
                                    size=SIZE_MARKERS,
                                    line=dict(
                                        color='Black',
                                        width=WIDTH_MARKER_LINES
                                    )
                                ),
                        error_y=dict(
                        type='data',  # value of error bar given in data coordinates
                        array=[finetuned_score_std],
                        visible=True),

                    ))

            ####################
            # tensortrain data #
            ####################
            df_tensortrain = df_baselines_model[df_baselines_model["compression"] == "tensortrain"]

            groupby_col = ['order', "rank-value", 'keep-first-layer']
            gp_by = df_tensortrain.groupby(groupby_col)
            for names, group in gp_by:
                dct_group_by_vals = dict(zip(groupby_col, names))
                keep_first = dct_group_by_vals["keep-first-layer"]
                str_keep_first = ' -1' if keep_first is True else ''
                if SHOW_KEEP_FIRST_ONLY and not keep_first:
                    continue
                order = int(dct_group_by_vals["order"])
                rank_value = int(dct_group_by_vals["rank-value"])


                name_trace = f"TT R={rank_value}"

                finetuned_score_mean = group["finetuned-score"].mean()
                nb_param_mean = group["nb-param-compressed-total"].mean()
                finetuned_score_std = group["finetuned-score"].std()
                nb_param_std = group["nb-param-compressed-total"].std()
                dict_algo_data_perf[dataname][name_trace] = finetuned_score_mean
                compression_rate_mean = group["param-compression-rate-total"].mean()
                base_score_tmp_mean = group["base-model-score"].mean()

                print(name_trace, end=" ")
                print("score, rate: ({:.2f}, {:.2f})".format(finetuned_score_mean, compression_rate_mean), end="; ")
                print("base score: {:.2f}".format(base_score_tmp_mean), end="; ")
                delta = base_score_tmp_mean - finetuned_score_mean
                print("delta: {:.2f}".format(delta))
                current_tuple = dict_algo_data_perf_2[dataname+" "+modelname].get("Tensortrain", (0, -1))
                # if current_tuple == (0, -1):
                #     dict_algo_data_perf_2[dataname+" "+modelname]["Tensortrain"] = (, "N/A")
                if store_value_in_dict_algo_data_perf_2(current_tuple[0], delta, current_tuple[1], compression_rate_mean):
                    dict_algo_data_perf_2[dataname+" "+modelname]["Tensortrain"] = (delta, compression_rate_mean)
                else:
                    pass
                fig.add_trace(
                    go.Scatter(
                        x=[nb_param_mean],
                        y=[finetuned_score_mean],
                        mode='markers',
                        name=name_trace,
                        hovertext=name_trace,
                        legendgroup=f"TT R={rank_value}",
                        marker=dict(
                                    color=dct_colors[f"TT R={rank_value}"],
                                    symbol=dct_symbol[f"TT{str_keep_first}"],
                                    size=SIZE_MARKERS,
                                    line=dict(
                                        color='Black',
                                        width=WIDTH_MARKER_LINES
                                    )
                                ),
                        error_y=dict(
                        type='data',  # value of error bar given in data coordinates
                        array=[finetuned_score_std],
                        visible=True),

                    ))


            ###############
            # random data #
            ###############
            if show_random:
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
                    keep_first = dct_group_by_vals["keep-first-layer"]
                    str_keep_first = ' -1' if keep_first is True else ''

                    name_trace = f"Random Q={nb_factor} K={sparsity_factor}{str_keep_first}"

                    finetuned_score_mean = group["finetuned-score"].mean()
                    nb_param_mean = group["nb-param-compressed-total"].mean()
                    finetuned_score_std = group["finetuned-score"].std()
                    nb_param_std = group["nb-param-compressed-total"].std()

                    fig.add_trace(
                        go.Scatter(
                            x=[nb_param_mean],
                            y=[finetuned_score_mean],
                            mode='markers',
                            name=name_trace,
                            hovertext=name_trace,
                            legendgroup=f"Random K={sparsity_factor}",
                            marker=dict(
                                color=dct_colors[f"Random K={sparsity_factor}"],
                                symbol=dct_symbol[f"Random Q={nb_factor}{str_keep_first}"],
                                size=SIZE_MARKERS,
                                line=dict(
                                    color='Black',
                                    width=0
                                )
                            ),
                            error_y=dict(
                                type='data',  # value of error bar given in data coordinates
                                array=[finetuned_score_std],
                                visible=True),
                            # error_x=dict(
                            #     type='data',  # value of error bar given in data coordinates
                            #     array=[nb_param_std],
                            #     visible=True)
                        ))


            ###############
            # magnitude data #
            ###############
            dct_magnitude_names = {"magnitude": "IP",
                                   "magnitude_hard":"HP"}
            for magnitude_str, magnitude_name in dct_magnitude_names.items():
                df_magnitude = df_baselines_model[df_baselines_model["compression"] == magnitude_str]

                groupby_col = ['final-sparsity', 'keep-first-layer']
                gp_by = df_magnitude.groupby(groupby_col)
                for names, group in gp_by:
                    dct_group_by_vals = dict(zip(groupby_col, names))
                    keep_first = dct_group_by_vals["keep-first-layer"]
                    str_keep_first = ' -1' if keep_first is True else ''

                    final_sparsity = float(dct_group_by_vals["final-sparsity"])


                    name_trace = f"{magnitude_name} {int(final_sparsity*100)}%"

                    finetuned_score_mean = group["finetuned-score"].mean()
                    nb_param_mean = group["nb-param-compressed-total"].mean()
                    finetuned_score_std = group["finetuned-score"].std()
                    nb_param_std = group["nb-param-compressed-total"].std()
                    dict_algo_data_perf[dataname][name_trace] = finetuned_score_mean
                    compression_rate_mean = group["param-compression-rate-total"].mean()
                    base_score_tmp_mean = group["base-model-score"].mean()

                    print(name_trace, end=" ")
                    print("score, rate: ({:.2f}, {:.2f})".format(finetuned_score_mean, compression_rate_mean), end="; ")
                    print("base score: {:.2f}".format(base_score_tmp_mean), end="; ")
                    delta = base_score_tmp_mean - finetuned_score_mean
                    print("delta: {:.2f}".format(delta))
                    current_tuple = dict_algo_data_perf_2[dataname+" "+modelname].get("Iterative Pruning", (0, -1))
                    if store_value_in_dict_algo_data_perf_2(current_tuple[0], delta, current_tuple[1], compression_rate_mean):
                        dict_algo_data_perf_2[dataname+" "+modelname]["Iterative Pruning"] = (delta, compression_rate_mean)
                    else:
                        pass
                    fig.add_trace(
                        go.Scatter(
                            x=[nb_param_mean],
                            y=[finetuned_score_mean],
                            mode='markers',
                            name=name_trace,
                            hovertext=name_trace,
                            legendgroup=f"{magnitude_name}",
                            marker=dict(
                                        color=dct_colors[f"{magnitude_name} {final_sparsity}"],
                                        symbol=dct_symbol[f"{magnitude_name}"],
                                        size=SIZE_MARKERS,
                                        line=dict(
                                            color='Black',
                                            width=WIDTH_MARKER_LINES
                                        )
                                    ),
                            error_y=dict(
                            type='data',  # value of error bar given in data coordinates
                            array=[finetuned_score_std],
                            visible=True),

                        ))

            ###############
            # multidense data #
            ###############
            df_multi_dense = df_baselines_model[df_baselines_model["compression"] == "multidense"]

            groupby_col = ['nb-fac', "final-sparsity", 'double-epochs-pruning', "keep-first-layer"]
            gp_by = df_multi_dense.groupby(groupby_col)
            for names, group in gp_by:
                dct_group_by_vals = dict(zip(groupby_col, names))
                keep_first = dct_group_by_vals["keep-first-layer"]
                str_keep_first = ' -1' if keep_first is True else ''
                if SHOW_KEEP_FIRST_ONLY and not keep_first:
                    continue
                nb_fac = int(dct_group_by_vals["nb-fac"])
                final_sparsity = float(dct_group_by_vals["final-sparsity"])
                doublepochs = bool(dct_group_by_vals["double-epochs-pruning"])
                if doublepochs == True:
                    continue
                name_trace = f"MultiDense Q={nb_fac} K={final_sparsity} DoubleEpochs={doublepochs}"

                finetuned_score_mean = group["finetuned-score"].mean()
                nb_param_mean = group["nb-param-compressed-total"].mean()
                finetuned_score_std = group["finetuned-score"].std()
                nb_param_std = group["nb-param-compressed-total"].std()

                fig.add_trace(
                    go.Scatter(
                        x=[nb_param_mean],
                        y=[finetuned_score_mean],
                        mode='markers',
                        name=name_trace,
                        hovertext=name_trace,
                        legendgroup=f"MultiDense",
                        marker=dict(
                            line_color=dct_colors[f"Multidense K={final_sparsity}"],
                            symbol=dct_symbol[f"Multidense Doubleepochs:{doublepochs}"],
                            size=SIZE_MARKERS,
                            line=dict(
                                color='Black',
                                width=WIDTH_MARKER_LINES
                            )
                        ),
                        error_y=dict(
                            type='data',  # value of error bar given in data coordinates
                            array=[finetuned_score_std],
                            visible=True,
                        color=dct_colors[f"Multidense K={final_sparsity}"]),


                    ))




            title = "Perf_vs_param" + dataname + "_" + modelname
            showlegend = True; width, height = None, None
            # showlegend = True; width, height = 1000, 500
            # showlegend = False; width, height = 500, 250


            x_legend = 0
            y_legend = -0.3
            french = True
            fig.update_layout(
                title=title,
                xaxis_title="# Param. dans les couches Denses et Conv." if french else  "# Param. in Conv. and Dense layers",
                yaxis_title="Précision (%)" if  french else "Accuracy (%)",
                xaxis_type="log",
                # xaxis={"mirror": True,
                #        "ticks": 'outside',
                #        "showline": True,
                #        'zerolinewidth':3},
                # yaxis={"mirror": True,
                #        "ticks": 'outside',
                #        "showline": True, },
                # width=width,
                # height=height,
                # autosize=False,
                autosize=True,
                # margin=dict(l=20, r=20, t=20, b=20),
                # title=title,
                font=dict(
                    # family="Courier New, monospace",
                    size=18,
                    color="black"
                ),
                # legend_orientation="h",
                showlegend=showlegend,
                # legend=dict(
                #     x=x_legend, y=y_legend,
                #     traceorder="normal",
                #     font=dict(
                #         family="sans-serif",
                #         size=18,
                #         color="black"
                #     ),
                #     # bgcolor="LightSteelBlue",
                #     # bordercolor="Black",
                #     borderwidth=1,
                # ),

            )
            fig.update_xaxes(showline=True, ticks="outside", linewidth=2, linecolor='black', mirror=True)
            fig.update_yaxes(showline=True, ticks="outside", linewidth=2, linecolor='black', mirror=True)
            # fig.show()
            # fig.write_image(str((output_dir / title).absolute()) + ".png")

    from pprint import pprint
    pprint(dict_algo_data_perf_2)

    n_lin = 6
    n_col = 7
    array_final_values = np.empty((n_lin, n_col), dtype='a100')
    array_final_values[0, 0] = ""
    array_final_values[1, 0] = ""
    for i in range(1, n_col):
        array_final_values[1, i] = r"CR $\downarrow$ & $\Delta$ $\uparrow$"


    lst_methods = [
                   "PSM network",
                   "Tensortrain",
                   "Tucker-SVD",
                   "Iterative Pruning",
    ]
    lst_dataname = [
        "MNIST Lenet",
        "SVHN Vgg19",
        "CIFAR10 Vgg19",
        "CIFAR100 Vgg19",
        "CIFAR100 Resnet20",
        "CIFAR100 Resnet50",
    ]

    for idx_method, method in enumerate(lst_methods):
        for idx_data, dataname in enumerate(lst_dataname):
            dataname_dict = dict_algo_data_perf_2[dataname]
            array_final_values[idx_method+2, 0] = method
            if idx_data == len(lst_dataname)-1:
                array_final_values[0, idx_data+1] = "\multicolumn{2}{c}{"+r"\texttt{"+dataname.replace(" ", r"\nobreak\hspace{.16667em plus .08333em}")+"}}"
            else:
                array_final_values[0, idx_data+1] = "\multicolumn{2}{c|}{"+r"\texttt{"+dataname.replace(" ", r"\nobreak\hspace{.16667em plus .08333em}")+"}}"

            try:
                tpl_val = dataname_dict[method]
                val = "{:.2f} & {:.2f}".format(tpl_val[1], tpl_val[0])
            except:
                val = r"\textit{N/A} & \textit{N/A}"
                # if idx_data == len(lst_dataname) - 1:
                #     val = "\multicolumn{2}{c}{N/A}"
                # else:
                #     val = "\multicolumn{2}{c|}{N/A}"

            array_final_values[idx_method+2, idx_data + 1] = val


    print(array_final_values)

    print(r"\toprule")
    for idx_lin, lin in enumerate(array_final_values):
        for idx_col, col in enumerate(lin):
            if idx_col == idx_lin == 0 or (idx_col == 0 and idx_lin == 1):
                print("{}", end=" & ")
            else:
                print((array_final_values[idx_lin, idx_col]).decode("utf-8"), end=" & " if idx_col != (len(lin)-1) else " \\\\ \n")
        if idx_lin == 0 or idx_lin == 1:
            print(r"\midrule")
    print(r"\midrule")
    print(r"\bottomrule")

r"""
{}&MNIST Lenet&SVHN Vgg19&Cifar10 Vgg19&Cifar100 Vgg19&Cifar100 Resnet20&Cifar100 Resnet50\\
\midrule
Base&\textbf{0.99}&\textbf{0.96}&\textbf{0.93}&\textbf{0.67}&\textbf{0.73}&\textbf{0.76}\\
\midrule
PSM Q=2 K=2&\underline{0.99}&0.92&0.84&0.46&0.56&0.67\\
PSM re-init. Q=2 K=2&\underline{0.99}&0.82&0.81&0.42&0.53&0.57\\
PSM random Q=2 K=2&\underline{0.98}&0.91&0.81&0.44&0.48&0.41\\
\midrule
PSM Q=2 K=14&\underline{0.99}&\underline{0.95}&\underline{0.92}&\underline{0.64}&0.69&\underline{0.72}\\
PSM re-init. Q=2 K=14&\underline{0.99}&0.44&0.86&0.57&0.63&0.63\\
PSM random Q=2 K=14&\underline{0.99}&0.92&0.85&0.58&0.62&0.62\\
\midrule
PSM Q=3 K=2&\underline{0.99}&0.94&0.85&0.42&0.57&0.67\\
PSM re-init. Q=3 K=2&0.98&0.91&0.80&0.32&0.48&0.51\\
PSM random Q=3 K=2&0.98&0.90&0.79&0.39&0.29&0.47\\
\midrule
PSM Q=3 K=14&\underline{0.99}&\underline{0.95}&\underline{0.92}&0.62&\underline{0.70}&\underline{0.72}\\
PSM re-init. Q=3 K=14&\underline{0.99}&0.89&0.84&0.31&0.60&0.58\\
PSM random Q=3 K=14&\underline{0.99}&0.93&0.84&0.51&0.60&0.59\\
\midrule
\bottomrule"""
