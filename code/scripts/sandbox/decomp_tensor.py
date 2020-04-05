from keras.layers import Conv2D, DepthwiseConv2D


def cp_conv(inputs, cp_rank, num_activations_maps):
    x = inputs
    x = Conv2D(cp_rank, (1, 1), activation='relu', kernel_initializer='random_normal', padding='same', use_bias=False)(x)
    x = DepthwiseConv2D(kernel_size=(1, 3), kernel_initializer='random_normal', padding='same', use_bias=False)(x)
    x = DepthwiseConv2D(kernel_size=(3, 1), kernel_initializer='random_normal', padding='same', use_bias=False)(x)
    x = Conv2D(num_activations_maps, (1, 1), activation='relu', kernel_initializer='random_normal', padding='same')(x)
    return x


def tucker_conv(inputs, in_channels_rank, out_channels_rank, num_activations_maps):
    x = inputs
    x = Conv2D(in_channels_rank, (1, 1), activation='relu', kernel_initializer='glorot_normal', padding='same')(x)  #
    x = Conv2D(out_channels_rank, kernel_size=(3, 3), kernel_initializer='glorot_normal', padding='same')(x)  # core
    x = Conv2D(num_activations_maps, (1, 1), activation='relu', kernel_initializer='glorot_normal', padding='same')(x)
    return x