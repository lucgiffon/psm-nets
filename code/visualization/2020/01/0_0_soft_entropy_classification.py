import pathlib
import pandas as pd

from palmnet.layers.pbp_layer import PBPDenseDensify
from palmnet.visualization.utils import get_palminized_model_and_df, get_df
import matplotlib.pyplot as plt
import numpy as np
import logging
import plotly.graph_objects as go
import keras
import scipy.special
from skluc.utils import logger

mpl_logger = logging.getLogger('matplotlib')
mpl_logger.setLevel(logging.ERROR)

dataset = {
    # "Cifar10": "--cifar10",
    # "SVHN": "--svhn",
    "MNIST": "--mnist"
}

color_bars_sparsity = {
    2: "g",
    3: "c",
    4: "b",
    5: "y"
}



tasks = {
    "finetuned_score"
}
ylabel_task = {
    "nb_flop": "log(# Flop)",
    "nb_param": "log(# non-zero value)",
    "finetuned_score": "Accuracy"
}

scale_tasks = {
    "finetuned_score": "linear"
}

def figure_perf():

    for dataname in dataset:
        df_data = df[df[dataset[dataname]] == 1]
        df_data_palminized = df_palminized[df_palminized[dataset[dataname]] == 1]
        df_data_glorot = df_glorot[df_glorot[dataset[dataname]] == 1]
        for nb_units in  nb_units_dense_layer:
            df_units = df_data[df_data["--nb-units-dense-layer"] == nb_units]
            df_units_glorot = df_data_glorot[df_data_glorot["--nb-units-dense-layer"] == nb_units]

            for task in tasks:
                # xticks = ["2", "3"]
                xticks = ["1", "2", "3", "log(min(A, B))"]
                # xticks = [1, 2, 3]

                fig = go.Figure()
                set_legend = set()
                # dense model
                ############v
                df_dense = df_units[df_units["--dense-layers"] == 1]
                x_ticks_dense = [-1] + xticks + [4]
                val = df_dense[task].values.mean()
                std_val = df_dense[task].values.std()
                y_val = [val] * len(x_ticks_dense)
                y_std_val = [std_val] * len(x_ticks_dense)
                fig.add_trace(
                    go.Scatter(
                        x=x_ticks_dense,
                        y=y_val,
                        mode='lines',
                        name="dense model",
                        error_y = dict(
                            type='data',  # value of error bar given in data coordinates
                            array=y_std_val,
                            visible=True)
                ))


                # pbp
                #####
                for i, sp_fac in enumerate(sorted(sparsity_factors)):
                    df_sparsity = df_units[df_units["--sparsity-factor"] == sp_fac]
                    df_sparsity_glorot = df_units_glorot[df_units_glorot["--sparsity-factor"] == int(sp_fac)]
                    for add_entro in [0]:
                    # for add_entro in [1, 0]:
                        df_add_entro = df_sparsity[df_sparsity["--add-entropies"] == add_entro]
                        df_add_entro_glorot = df_sparsity_glorot[df_sparsity_glorot["--add-entropies"] == add_entro]
                        str_add_entro = "+" if add_entro else "*"
                        for param_reg in sorted(param_reg_softentropy):
                            df_reg = df_add_entro[df_add_entro["--param-reg-softmax-entropy"] == param_reg]
                            df_reg_glorot = df_add_entro_glorot[df_add_entro_glorot["--param-reg-softmax-entropy"] == float(param_reg)]
                            for nb_fac in nb_factors:
                                df_nb_fac = df_reg[df_reg["--nb-factor"] == nb_fac]
                                df_nb_fac_glorot = df_reg_glorot[df_reg_glorot["--nb-factor"] == int(nb_fac)]
                                finetune_score_values = df_nb_fac["finetuned_score"].mean()
                                finetune_score_values_std = df_nb_fac["finetuned_score"].std()
                                finetune_score_values_glorot = df_nb_fac_glorot["finetuned_score"].mean()
                                finetune_score_values_std_glorot = df_nb_fac_glorot["finetuned_score"].std()

                                hls_str = "hsl(60, {}%, 40%)".format(saturation_by_param_softentropy[param_reg])
                                str_legend = 'sparsity {} - reg {} - {}'.format(sp_fac, param_reg, str_add_entro)
                                fig.add_trace(go.Bar(name=str_legend, x=[str(nb_fac)], y=[finetune_score_values], marker_color=hls_str
                                                     ,error_y = dict(
                                                        type='data',  # value of error bar given in data coordinates
                                                        array=[finetune_score_values_std],
                                                        visible=True),
                                                     showlegend=False if str_legend in set_legend else True
                                ))
                                set_legend.add(str_legend)

                                hls_str = "hsl(180, {}%, 40%)".format(saturation_by_param_softentropy[param_reg])
                                str_legend_2 = 'G - sparsity {} - reg {} - {}'.format(sp_fac, param_reg, str_add_entro)
                                fig.add_trace(go.Bar(name=str_legend_2, x=[str(nb_fac)], y=[finetune_score_values_glorot], marker_color=hls_str
                                                     ,error_y = dict(
                                                        type='data',  # value of error bar given in data coordinates
                                                        array=[finetune_score_values_std_glorot],
                                                        visible=True),
                                                     showlegend=False if str_legend_2 in set_legend else True
                                ))
                                set_legend.add(str_legend_2)


                # palminized
                for hierarchical_value in [1, 0]:
                    df_hier = df_data_palminized[df_data_palminized["--hierarchical"] == hierarchical_value]
                    str_hier = "- H" if hierarchical_value == 1 else ""
                    for i, sp_fac in enumerate(sorted(sparsity_factors)):
                        df_sparsity = df_hier[df_hier["--sparsity-factor"] == int(sp_fac)]
                        for nb_fac in df_sparsity["--nb-factor"].values:
                            df_nb_fac = df_sparsity[df_sparsity["--nb-factor"] == nb_fac]
                            if nb_fac == "None":
                                nb_fac_str = "log(min(A, B))"
                            else:
                                nb_fac_str = str(nb_fac)
                            finetune_score_values = df_nb_fac["finetuned_score"].mean()
                            finetune_score_values_std = df_nb_fac["finetuned_score"].std()

                            hls_str = "hsl({}, 50%, 40%)".format(hue_by_hierarchical[hierarchical_value])
                            fig.add_trace(go.Bar(name='PALM - sparsity {} {}'.format(sp_fac, str_hier), x=[str(nb_fac_str)], y=[finetune_score_values], marker_color=hls_str
                                                 ,
                                                 # error_y=dict(
                                                 #    type='data',  # value of error bar given in data coordinates
                                                 #    array=[finetune_score_values_std],
                                                 #    visible=True)
                                         ))


                title = task + " " + dataname + " " + str(nb_units)

                fig.update_layout(barmode='group',
                                  title=title,
                                  xaxis_title="# Factor",
                                  yaxis_title=ylabel_task[task],
                                  yaxis_type=scale_tasks[task],
                                  xaxis={'type': 'category'},
                                  )
                fig.show()
                fig.write_image(str((output_dir / title).absolute()) + ".png")

def figure_convergence():
    col_csv_file = "output_file_csvcbprinter"
    col_loss = 'loss'
    for dataname in dataset:
        df_data = df[df[dataset[dataname]] == 1]
        df_data_glorot = df_glorot[df_glorot[dataset[dataname]] == 1]
        for nb_units in nb_units_dense_layer:
            df_units = df_data[df_data["--nb-units-dense-layer"] == nb_units]
            df_units_glorot = df_data_glorot[df_data_glorot["--nb-units-dense-layer"] == nb_units]

            fig = go.Figure()
            set_legend = set()

            # dense model
            ############v
            df_dense = df_units[df_units["--dense-layers"] == 1]
            losses_dense = []
            for of in df_dense[col_csv_file]:
                with open(src_results_dir / of, 'r') as csvfile:
                    df_obj_vals = pd.read_csv(csvfile)
                losses_dense.append(df_obj_vals[col_loss].values)
            losses_dense_mean = np.mean(np.array(losses_dense), axis=0)
            losses_dense_std = np.std(np.array(losses_dense), axis=0)

            fig.add_trace(
                go.Scatter(
                    y=losses_dense_mean,
                    mode='lines',
                    name="dense model",
                    error_y=dict(
                        type='data',  # value of error bar given in data coordinates
                        array=losses_dense_std,
                        visible=True)
                ))

            # pbp
            #####
            for i, sp_fac in enumerate(sorted(sparsity_factors)):
                df_sparsity = df_units[df_units["--sparsity-factor"] == sp_fac]
                df_sparsity_glorot = df_units_glorot[df_units_glorot["--sparsity-factor"] == int(sp_fac)]
                for add_entro in [0]:
                    df_add_entro = df_sparsity[df_sparsity["--add-entropies"] == add_entro]
                    df_add_entro_glorot = df_sparsity_glorot[df_sparsity_glorot["--add-entropies"] == add_entro]
                    str_add_entro = "+" if add_entro else "*"
                    for param_reg in sorted(param_reg_softentropy):
                        df_reg = df_add_entro[df_add_entro["--param-reg-softmax-entropy"] == param_reg]
                        df_reg_glorot = df_add_entro_glorot[df_add_entro_glorot["--param-reg-softmax-entropy"] == float(param_reg)]
                        for nb_fac in ["2"]:
                        # for nb_fac in nb_factors:
                            df_nb_fac = df_reg[df_reg["--nb-factor"] == nb_fac]
                            df_nb_fac_glorot = df_reg_glorot[df_reg_glorot["--nb-factor"] == int(nb_fac)]

                            losses_pbp = []
                            for of in df_nb_fac[col_csv_file]:
                                with open(src_results_dir / of, 'r') as csvfile:
                                    df_obj_vals = pd.read_csv(csvfile)
                                losses_pbp.append(df_obj_vals[col_loss].values)

                            losses_pbp_mean = np.mean(np.array(losses_pbp), axis=0)
                            # losses_pbp_std = np.std(np.array(losses_pbp), axis=0)

                            hls_str = "hsl(60, {}%, 40%)".format(saturation_by_param_softentropy[param_reg])
                            try:
                                fig.add_trace(go.Scatter(name='sparsity {} - reg {} - {}'.format(sp_fac, param_reg, str_add_entro),
                                                         y=losses_pbp_mean,
                                                         marker_color=hls_str
                                                         #              , error_y=dict(
                                                         # type='data',  # value of error bar given in data coordinates
                                                         # array=losses_pbp_std,
                                                         # visible=True)
                                                         ))
                            except:
                                print("a")

                            try:
                                losses_pbp_glorot = []
                                for of in df_nb_fac_glorot[col_csv_file]:
                                    try:
                                        with open(src_results_dir_glorot / of, 'r') as csvfile:
                                            df_obj_vals = pd.read_csv(csvfile)
                                    except FileNotFoundError:
                                        print("file not found error")
                                        pass
                                    losses_pbp_glorot.append(df_obj_vals[col_loss].values)
                                losses_pbp_mean_glorot = np.mean(np.array(losses_pbp_glorot), axis=0)

                                hls_str = "hsl(180, {}%, 40%)".format(saturation_by_param_softentropy[param_reg])
                                fig.add_trace(go.Scatter(name='G - sparsity {} - reg {} - {}'.format(sp_fac, param_reg, str_add_entro),
                                                         y=losses_pbp_mean_glorot,
                                                         marker_color=hls_str
                                                         #              , error_y=dict(
                                                         # type='data',  # value of error bar given in data coordinates
                                                         # array=losses_pbp_std,
                                                         # visible=True)
                                                         ))

                            except:
                                print("nan error?")
                                pass
                            print("after nan")


            title = "loss" + " " + dataname + " " + str(nb_units)

            fig.update_layout(barmode='group',
                              title=title,
                              xaxis_title="Iteration",
                              yaxis_title="Loss value",
                              xaxis={'type': 'category'},
                              )
            fig.show()
            fig.write_image(str((output_dir / title).absolute()) + ".png")

def figure_permutations():
    col_model_file = "output_file_modelprinter"

    for dataname in dataset:
        df_data = df[df[dataset[dataname]] == 1]
        df_data_glorot = df_glorot[df_glorot[dataset[dataname]] == 1]
        for nb_units in nb_units_dense_layer:
            df_units = df_data[df_data["--nb-units-dense-layer"] == nb_units]
            df_units_glorot = df_data_glorot[df_data_glorot["--nb-units-dense-layer"] == nb_units]

            # pbp
            #####
            for i, sp_fac in enumerate(sorted(sparsity_factors)):
                df_sparsity = df_units[df_units["--sparsity-factor"] == sp_fac]
                df_sparsity_glorot = df_units_glorot[df_units_glorot["--sparsity-factor"] == int(sp_fac)]
                for add_entro in [0]:
                # for add_entro in [1, 0]:
                    df_add_entro = df_sparsity[df_sparsity["--add-entropies"] == add_entro]
                    df_add_entro_glorot = df_sparsity_glorot[df_sparsity_glorot["--add-entropies"] == add_entro]
                    str_add_entro = "+" if add_entro else "*"
                    for param_reg in ["0.0001"]:
                    # for param_reg in sorted(param_reg_softentropy):
                        df_reg = df_add_entro[df_add_entro["--param-reg-softmax-entropy"] == param_reg]
                        df_reg_glorot = df_add_entro_glorot[df_add_entro_glorot["--param-reg-softmax-entropy"] == float(param_reg)]
                        for nb_fac in ["2"]:
                        # for nb_fac in nb_factors:
                            df_nb_fac = df_reg[df_reg["--nb-factor"] == nb_fac]
                            df_nb_fac_glorot = df_reg_glorot[df_reg_glorot["--nb-factor"] == int(nb_fac)]

                            for of in df_nb_fac[col_model_file]:
                                break
                                pbp_model = keras.models.load_model(str((src_results_dir / of).absolute()),custom_objects={"PBPDenseDensify": PBPDenseDensify})
                                names = [weight.name for layer in pbp_model.layers for weight in layer.weights]
                                weights = pbp_model.get_weights()
                                for name, weight in zip(names, weights):
                                    if len(weight.shape) == 1:
                                        continue
                                    if "permutation" in name:
                                        weight = np.multiply(scipy.special.softmax(weight, axis=1), scipy.special.softmax(weight, axis=0))
                                        weight[weight < 0.5] = 0
                                        weight[weight >= 0.5] = 1
                                        sum_one = np.sum(weight)


                                        plt.imshow(weight)
                                        plt.title('{}{}-spar.{}-{}-{}-{}'.format(sum_one, name, sp_fac, param_reg, str_add_entro, nb_fac))
                                        plt.show()
                                break
                            for of in df_nb_fac_glorot[col_model_file]:
                                if df_nb_fac_glorot[df_nb_fac_glorot[col_model_file] == of] ["finetuned_score"].iloc[0] > 0.5:
                                    pbp_model = keras.models.load_model(str((src_results_dir_glorot / of).absolute()),custom_objects={"PBPDenseDensify": PBPDenseDensify})
                                    names = [weight.name for layer in pbp_model.layers for weight in layer.weights]
                                    weights = pbp_model.get_weights()
                                    for name, weight in zip(names, weights):
                                        if len(weight.shape) == 1:
                                            continue
                                        if "permutation" in name:
                                            weight = np.multiply(scipy.special.softmax(weight, axis=1), scipy.special.softmax(weight, axis=0))
                                            # weight[weight < 0.5] = 0
                                            # weight[weight >= 0.5] = 1
                                            # sum_one = np.sum(weight)
                                            sum_one=0
                                            plt.imshow(weight)
                                            plt.title('{}Glo-{}-spar.{}-{}-{}-{}'.format(sum_one, name, sp_fac, param_reg, str_add_entro, nb_fac))
                                            plt.show()
                                    # break
                            # losses_pbp_std = np.std(np.array(losses_pbp), axis=0)

if __name__ == "__main__":
    FORCE = False

    logger.setLevel(logging.ERROR)
    root_source_dir = pathlib.Path("/home/luc/PycharmProjects/palmnet/results/")
    expe_path = "2020/01/0_0_soft_entropy_classification"
    expe_path_palminized = "2020/02/3_4_finetune_palminized_mnist_500"
    expe_path_glorot = "2020/02/0_0_soft_entropy_classification_glorot_uniform_init_permutation"
    src_results_dir = root_source_dir / expe_path
    src_results_dir_palminized = root_source_dir / expe_path_palminized
    src_results_dir_glorot = root_source_dir /  expe_path_glorot

    df_path = src_results_dir / "prepared_df.csv"
    df_path_palminized = src_results_dir_palminized / "prepared_df.csv"
    df_path_glorot = src_results_dir_glorot / "prepared_df.csv"

    if df_path.exists() and df_path_palminized.exists() and df_path_glorot.exists() and not FORCE:
        df = pd.read_csv(df_path, sep=";")
        df_palminized = pd.read_csv(df_path_palminized, sep=";")
        df_glorot = pd.read_csv(df_path_glorot, sep=";")
    else:
        df = get_df(src_results_dir)
        df_palminized = get_df(src_results_dir_palminized)
        df_glorot = get_df(src_results_dir_glorot)

        df_glorot[["--sparsity-factor", "failure", "finetuned_score"]] = df_glorot[["--sparsity-factor", "failure", "finetuned_score"]].apply(pd.to_numeric, errors='coerce')

        df_palminized[["--sparsity-factor", "failure", "finetuned_score"]] = df_palminized[["--sparsity-factor", "failure", "finetuned_score"]].apply(pd.to_numeric, errors='coerce')

        df[["failure", "finetuned_score"]] = df[["failure", "finetuned_score"]].apply(pd.to_numeric, errors='coerce')
        df = df.dropna(subset=["failure", "finetuned_score"]).drop(columns="oar_id").drop_duplicates()
        df[["--sparsity-factor", "--param-reg-softmax-entropy", "--nb-factor"]] = df[["--sparsity-factor", "--param-reg-softmax-entropy", "--nb-factor"]].apply(pd.to_numeric, errors='ignore')

        df.to_csv(df_path, sep=";")
        df_palminized.to_csv(df_path_palminized, sep=";")
        df_glorot.to_csv(df_path_glorot, sep=";")


    root_output_dir = pathlib.Path("/home/luc/PycharmProjects/palmnet/reports/figures/")
    output_dir = root_output_dir / expe_path / "histogrammes"
    output_dir.mkdir(parents=True, exist_ok=True)

    nb_units_dense_layer = set(df["--nb-units-dense-layer"])
    param_reg_softentropy = set(df["--param-reg-softmax-entropy"]).union([str(elm) for elm in set(df_glorot["--param-reg-softmax-entropy"])])
    param_reg_softentropy.remove("None")
    param_reg_softentropy = [str(elm) for elm in sorted(map(float, param_reg_softentropy))]

    sparsity_factors = set(df["--sparsity-factor"])
    sparsity_factors.remove("None")
    sparsity_factors = sorted(sparsity_factors)
    nb_factors = set(df["--nb-factor"]).union([str(elm) for elm in set(df_glorot["--nb-factor"])])
    nb_factors.remove("None")


    hue_by_add_entropy= {

        1: 60,
        0: 180
    }
    hue_by_hierarchical= {

        1: 20,
        0: 140
    }

    saturation_by_param_softentropy = dict(zip(sorted(param_reg_softentropy), np.linspace(15, 100, len(param_reg_softentropy), dtype=int)))

    # figure_perf()
    # figure_convergence()
    figure_permutations()

