"""
Simple module for computing number param and number of flop for model with conv2D and dense layers.

See example at the bottom of the document.
"""

from keras import Sequential
from keras.layers import Conv2D, Dense
import numpy as np

def count_nb_param_layer(layer):
    nb_param_layer = int(np.prod(layer.kernel.shape))
    nb_param_layer_bias = int(np.prod(layer.bias.shape))
    return nb_param_layer, nb_param_layer_bias

def count_nb_flop_conv_layer(layer, nb_param_layer, nb_param_layer_bias):
    if layer.padding == "valid":
        padding_horizontal = 0
        padding_vertical = 0
    elif layer.padding == "same":
        padding_horizontal = int(layer.kernel.shape[0]) // 2
        padding_vertical = int(layer.kernel.shape[1]) // 2
    else:
        raise ValueError("Unknown padding value for convolutional layer {}.".format(layer.name))

    nb_patch_horizontal = (layer.input_shape[1] + 2 * padding_horizontal - layer.kernel.shape[0]) // layer.strides[0]
    nb_patch_vertical = (layer.input_shape[2] + 2 * padding_vertical - layer.kernel.shape[1]) // layer.strides[1]

    imagette_matrix_size = int(nb_patch_horizontal * nb_patch_vertical)

    # *2 for the multiplcations and then sum
    nb_flop_layer_for_one_imagette = nb_param_layer * 2 + nb_param_layer_bias

    nb_flop_layer = imagette_matrix_size * nb_flop_layer_for_one_imagette

    return nb_flop_layer

def count_nb_flop_dense_layer(layer, nb_param_layer, nb_param_layer_bias):
    # *2 for the multiplcations and then sum
    nb_flop_layer = nb_param_layer * 2 + nb_param_layer_bias
    return nb_flop_layer

def count_model_param_and_flops(model):
    """
    Return the number of params and the number of flops of (only) 2DConvolutional Layers and Dense Layers for both the model.

    :return:
    """
    param_by_layer = dict()
    flop_by_layer = dict()

    nb_param_model, nb_flop_model = 0, 0

    for layer in model.layers:
        if isinstance(layer, Conv2D):
            nb_param_layer, nb_param_layer_bias = count_nb_param_layer(layer)
            nb_flop_layer = count_nb_flop_conv_layer(layer, nb_param_layer, nb_param_layer_bias)

        elif isinstance(layer, Dense):
            nb_param_layer, nb_param_layer_bias = count_nb_param_layer(layer)
            nb_flop_layer = count_nb_flop_dense_layer(layer, nb_param_layer, nb_param_layer_bias)

        else:
            # if you have over layers you want to compute flops in: put other conditions here and write the necessary functions
            nb_param_layer, nb_param_layer_bias, nb_flop_layer = 0, 0, 0

        param_by_layer[layer.name] = nb_param_layer + nb_param_layer_bias
        flop_by_layer[layer.name] = nb_flop_layer

        nb_param_model += nb_param_layer
        nb_flop_model += nb_flop_layer

    total_nb_param_model = nb_param_model
    total_nb_flop_model = nb_flop_model

    return total_nb_param_model, total_nb_flop_model

if __name__ == "__main__":
    my_model = Sequential()
    my_model.add(Dense(units=200, input_shape=(200,)))
    my_model.summary()
    print(count_model_param_and_flops(my_model))