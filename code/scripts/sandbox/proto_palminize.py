import logging
from copy import deepcopy

import keras
from keras.layers import Conv2D, Dense
from qkmeans.core.utils import build_constraint_set_smart
from qkmeans.palm.palm_fast import hierarchical_palm4msa

from skluc.utils import logger

from palmnet.utils import root_dir
from skluc.utils.osutils import download_file, check_file_md5
# import tensorflow as tf
import keras.backend as K
import numpy as np
from palmnet.data import Mnist

def apply_palm(matrix, sparsity_fac=2):
    logging.info("Applying palm function to matrix with shape {}".format(matrix.shape))
    transposed = False

    if matrix.shape[0] > matrix.shape[1]:
        # we want the bigger dimension to be on right due to the residual computation that should remain big
        matrix = matrix.T
        transposed = True

    left_dim, right_dim = matrix.shape
    A = min(left_dim, right_dim)
    B = max(left_dim, right_dim)
    assert A == left_dim and B == right_dim, "Dimensionality problem: left dim should be higher than right dim before palm"

    nb_factors = int(np.log(A))
    nb_iter = 300

    lst_factors = [np.eye(A) for _ in range(nb_factors + 1)]
    lst_factors[-1] = np.zeros((A, B))
    _lambda = 1.  # init the scaling factor at 1

    lst_proj_op_by_fac_step, lst_proj_op_by_fac_step_desc = build_constraint_set_smart(left_dim=left_dim,
                                                                                       right_dim=right_dim,
                                                                                       nb_factors=nb_factors + 1, # this is due to constant as first factor (so will be identity)
                                                                                       sparsity_factor=sparsity_fac,
                                                                                       residual_on_right=True,
                                                                                       fast_unstable_proj=False)

    final_lambda, final_factors, final_X, _, _ = hierarchical_palm4msa(
        arr_X_target=matrix,
        lst_S_init=lst_factors,
        lst_dct_projection_function=lst_proj_op_by_fac_step,
        f_lambda_init=_lambda,
        nb_iter=nb_iter,
        update_right_to_left=True,
        residual_on_right=True)

    if transposed:
        return final_X.T
    else:
        return final_X


def palminize_layer(layer_obj):
    """
    Takes a keras layer object as entry and modify its weights as reconstructed by the palm approximation.

    The layer is modifed in place but the inner weight tensor is returned modifed.

    :param layer_obj: The layer object to which modify weights
    :return: The new weights
    """
    if isinstance(layer_obj, Conv2D):
        logger.info("Find {}".format(layer_obj.__class__.__name__))
        layer_weights, layer_bias = layer_obj.get_weights()
        filter_height, filter_width, in_chan, out_chan = layer_weights.shape
        filter_matrix = layer_weights.reshape(filter_height*filter_width*in_chan, out_chan)
        reconstructed_filter_matrix = apply_palm(filter_matrix)
        new_layer_weights = reconstructed_filter_matrix.reshape(filter_height, filter_width, in_chan, out_chan)
        layer_obj.set_weights((new_layer_weights, layer_bias))
        return new_layer_weights
    elif isinstance(layer_obj, Dense):
        logger.info("Find {}".format(layer_obj.__class__.__name__))
        layer_weights, layer_bias = layer_obj.get_weights()
        reconstructed_dense_matrix = apply_palm(layer_weights)
        new_layer_weights = reconstructed_dense_matrix
        layer_obj.set_weights((new_layer_weights, layer_bias))
        return new_layer_weights
    else:
        logger.debug("Find {}. Can't Palminize this. Pass.".format(layer_obj.__class__.__name__))
        return

def palminize_model(model):
    """
    Takes a keras model object as entry and returns a version of it with all weights matrix palminized.

    Modifications are in-place but the model is still returned.

    :param model: Keras model
    :return: The same, model object with new weights.
    """
    for layer in model.layers:
        palminize_layer(layer)
    return model

if __name__ == "__main__":
    logger.setLevel(logging.INFO)
    url_mnist = "https://pageperso.lis-lab.fr/~luc.giffon/saved_models/mnist_lenet_1570207294.h5"
    md5sum = "26d44827c84d44a9fc8f4e021b7fe4d2"
    download_path = download_file(url_mnist, root_dir / "models")
    check_file_md5(download_path, md5sum)
    (x_train, y_train), (x_test, y_test) = Mnist.load_data()

    model = keras.models.load_model(download_path, compile=False)
    model_init = deepcopy(model)

    model = palminize_model(model)
    model.compile(loss='binary_crossentropy',
                  optimizer="adam",
                  metrics=['accuracy'])

    model_init.compile(loss='binary_crossentropy',
                  optimizer="adam",
                  metrics=['accuracy'])


    score, acc = model.evaluate(x_test, y_test)
    print("palminized", score, acc)
    score, acc = model_init.evaluate(x_test, y_test)
    print("init", score, acc)

    print(download_path)