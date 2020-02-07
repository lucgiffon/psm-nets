import pathlib
import pandas as pd

from palmnet.layers.pbp_layer import PBPDenseDensify
from palmnet.visualization.utils import get_palminized_model_and_df, get_df
import matplotlib.pyplot as plt
import numpy as np
import logging
import plotly.graph_objects as go
import keras

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

hatch_bars_permutation = {
    0: "/",
    1: None
}

hatch_hierarchical = {
    0: "X",
    1: "O"
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
        for nb_units in  nb_units_dense_layer:
            df_units = df_data[df_data["--nb-units-dense-layer"] == nb_units]
            for task in tasks:
                xticks = ["2", "3"]
                # xticks = ["A", "B", "log(min(A, B))"]
                # xticks = [1, 2, 3]

                fig = go.Figure()

                # dense model
                ############v
                df_dense = df_units[df_units["--dense-layers"] == 1]
                x_ticks_dense = [-1] + xticks + [1]
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
                    for add_entro in [1, 0]:
                        df_add_entro = df_sparsity[df_sparsity["--add-entropies"] == add_entro]
                        str_add_entro = "+" if add_entro else "*"
                        for param_reg in sorted(param_reg_softentropy):
                            df_reg = df_add_entro[df_add_entro["--param-reg-softmax-entropy"] == param_reg]
                            for nb_fac in nb_factors:
                                df_nb_fac = df_reg[df_reg["--nb-factor"] == nb_fac]
                                finetune_score_values = df_nb_fac["finetuned_score"].mean()
                                finetune_score_values_std = df_nb_fac["finetuned_score"].std()

                                hls_str = "hsl({}, {}%, 40%)".format(hue_by_add_entropy[add_entro], saturation_by_param_softentropy[param_reg])
                                fig.add_trace(go.Bar(name='sparsity {} - reg {} - {}'.format(sp_fac, param_reg, str_add_entro), x=[str(nb_fac)], y=[finetune_score_values], marker_color=hls_str
                                                     ,error_y = dict(
                                                        type='data',  # value of error bar given in data coordinates
                                                        array=[finetune_score_values_std],
                                                        visible=True)
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
        for nb_units in nb_units_dense_layer:
            df_units = df_data[df_data["--nb-units-dense-layer"] == nb_units]

            fig = go.Figure()

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
                for add_entro in [1, 0]:
                    df_add_entro = df_sparsity[df_sparsity["--add-entropies"] == add_entro]
                    str_add_entro = "+" if add_entro else "*"
                    for param_reg in sorted(param_reg_softentropy):
                        df_reg = df_add_entro[df_add_entro["--param-reg-softmax-entropy"] == param_reg]
                        for nb_fac in nb_factors:
                            df_nb_fac = df_reg[df_reg["--nb-factor"] == nb_fac]

                            losses_pbp = []
                            for of in df_nb_fac[col_csv_file]:
                                with open(src_results_dir / of, 'r') as csvfile:
                                    df_obj_vals = pd.read_csv(csvfile)
                                losses_pbp.append(df_obj_vals[col_loss].values)
                            losses_pbp_mean = np.mean(np.array(losses_pbp), axis=0)
                            # losses_pbp_std = np.std(np.array(losses_pbp), axis=0)

                            hls_str = "hsl({}, {}%, 40%)".format(hue_by_add_entropy[add_entro], saturation_by_param_softentropy[param_reg])
                            fig.add_trace(go.Scatter(name='sparsity {} - reg {} - {}'.format(sp_fac, param_reg, str_add_entro),
                                                 y=losses_pbp_mean,
                                                 marker_color=hls_str
                                    #              , error_y=dict(
                                    # type='data',  # value of error bar given in data coordinates
                                    # array=losses_pbp_std,
                                    # visible=True)
                             ))

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
        for nb_units in nb_units_dense_layer:
            df_units = df_data[df_data["--nb-units-dense-layer"] == nb_units]

            # pbp
            #####
            for i, sp_fac in enumerate(sorted(sparsity_factors)):
                df_sparsity = df_units[df_units["--sparsity-factor"] == sp_fac]
                for add_entro in [1, 0]:
                    df_add_entro = df_sparsity[df_sparsity["--add-entropies"] == add_entro]
                    str_add_entro = "+" if add_entro else "*"
                    for param_reg in sorted(param_reg_softentropy):
                        df_reg = df_add_entro[df_add_entro["--param-reg-softmax-entropy"] == param_reg]
                        for nb_fac in nb_factors:
                            df_nb_fac = df_reg[df_reg["--nb-factor"] == nb_fac]

                            for of in df_nb_fac[col_model_file]:
                                pbp_model = keras.models.load_model(str((src_results_dir / of).absolute()),custom_objects={"PBPDenseDensify": PBPDenseDensify})
                                names = [weight.name for layer in pbp_model.layers for weight in layer.weights]
                                weights = pbp_model.get_weights()
                                for name, weight in zip(names, weights):
                                    if len(weight.shape) == 1:
                                        continue
                                    plt.imshow(weight)
                                    plt.title('{}- sparsity {} - reg {} - {} - {} fac'.format(name, sp_fac, param_reg, str_add_entro, nb_fac))
                                    plt.show()
                            # losses_pbp_std = np.std(np.array(losses_pbp), axis=0)



            title = "loss" + " " + dataname + " " + str(nb_units)


if __name__ == "__main__":
    FORCE = True

    logger.setLevel(logging.ERROR)
    root_source_dir = pathlib.Path("/home/luc/PycharmProjects/palmnet/results/")
    expe_path = "2020/01/0_0_soft_entropy_classification"
    src_results_dir = root_source_dir / expe_path

    df_path = src_results_dir / "prepared_df.csv"

    if df_path.exists() and not FORCE:
        df = pd.read_csv(df_path, sep=";")
    else:
        df = get_df(src_results_dir)
        df[["failure", "finetuned_score"]] = df[["failure", "finetuned_score"]].apply(pd.to_numeric, errors='coerce')
        df = df.dropna(subset=["failure", "finetuned_score"]).drop(columns="oar_id").drop_duplicates()
        df[["--sparsity-factor", "--param-reg-softmax-entropy", "--nb-factor"]] = df[["--sparsity-factor", "--param-reg-softmax-entropy", "--nb-factor"]].apply(pd.to_numeric, errors='ignore')
        df.to_csv(df_path, sep=";")

    root_output_dir = pathlib.Path("/home/luc/PycharmProjects/palmnet/reports/figures/")
    output_dir = root_output_dir / expe_path / "histogrammes"
    output_dir.mkdir(parents=True, exist_ok=True)

    nb_units_dense_layer = set(df["--nb-units-dense-layer"])
    param_reg_softentropy = set(df["--param-reg-softmax-entropy"])
    param_reg_softentropy.remove("None")
    sparsity_factors = set(df["--sparsity-factor"])
    sparsity_factors.remove("None")
    sparsity_factors = sorted(sparsity_factors)
    nb_factors = set(df["--nb-factor"])
    nb_factors.remove("None")


    hue_by_add_entropy= {

        1: 60,
        0: 180
    }

    saturation_by_param_softentropy = dict(zip(sorted(param_reg_softentropy), np.linspace(40, 80, len(param_reg_softentropy), dtype=int)))

    # figure_perf()
    figure_convergence()
    # figure_permutations()

