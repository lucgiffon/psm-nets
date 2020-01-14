import pathlib
import pickle
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    path_model = pathlib.Path("/home/luc/PycharmProjects/palmnet/code/scripts/2019/12/8439359374_model_layers.pckle")
    model = pickle.load(open(path_model, 'rb'))
    lst_factors_fc1 = [elm.toarray() for elm in model.sparsely_factorized_layers["fc1"][1].get_list_of_factors()]
    for elm in lst_factors_fc1:
        plt.imshow(elm)
        plt.show()
