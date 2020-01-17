import keras
from keras import Sequential
from keras.initializers import he_normal
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Activation, Dropout

from palmnet.layers.pbp_layer import PBPDense
from palmnet.layers.sparse_tensor import RandomSparseFactorisationConv2D, RandomSparseFactorisationDense


def random_small_model(input_shape, num_classes):
    model = Sequential()
    model.add(Conv2D(5, tuple([v-7 for v in input_shape[:2]]), padding='valid', activation='relu', kernel_initializer=he_normal(), input_shape=input_shape))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(Flatten(name='flatten'))
    model.add(Dense(10, use_bias=True, kernel_initializer=he_normal(), name='fc1'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dense(num_classes, kernel_initializer=he_normal(), name='predictions'))
    model.add(BatchNormalization())
    model.add(Activation('softmax'))
    return model

def create_pbp_model(input_shape, num_classes, sparsity_factor, nb_sparse_factors, units, soft_entropy_regularisation):
    model = Sequential()
    model.add(PBPDense(input_shape=input_shape, units=units[0], activation='relu', nb_factor=nb_sparse_factors, sparsity_factor=sparsity_factor, entropy_regularization_parameter=soft_entropy_regularisation))
    for nb_unit_layer in units[1:]:
        model.add(PBPDense(units=nb_unit_layer, nb_factor=nb_sparse_factors, activation='relu', sparsity_factor=sparsity_factor, entropy_regularization_parameter=soft_entropy_regularisation))
    model.add(Dense(num_classes, activation='softmax'))
    return model

def create_dense_model(input_shape, num_classes,  units):
    model = Sequential()
    model.add(Dense(input_shape=input_shape, units=units[0]))
    for nb_unit_layer in units[1:]:
        model.add(Dense(units=nb_unit_layer, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    return model

def sparse_random_lenet_model(input_shape, num_classes, sparsity_factor, nb_sparse_factors, permutation=True, weight_decay=1e-4):
    model = Sequential()
    model.add(RandomSparseFactorisationConv2D(input_shape=input_shape, permutation=permutation, filters=6, kernel_size=(5, 5), activation='relu', kernel_initializer=he_normal(), padding="valid", sparsity_factor=sparsity_factor, nb_sparse_factors=nb_sparse_factors))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(RandomSparseFactorisationConv2D(input_shape=input_shape, permutation=permutation, filters=16, kernel_size=(5, 5), activation='relu', kernel_initializer=he_normal(), padding="valid", sparsity_factor=sparsity_factor, nb_sparse_factors=nb_sparse_factors))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(Flatten(name='flatten'))
    model.add(RandomSparseFactorisationDense(1024, permutation=permutation, sparsity_factor=sparsity_factor, nb_sparse_factors=nb_sparse_factors, use_bias=True, kernel_regularizer=keras.regularizers.l2(weight_decay), kernel_initializer=he_normal(), name='fc1'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(RandomSparseFactorisationDense(num_classes, permutation=permutation, sparsity_factor=sparsity_factor, nb_sparse_factors=nb_sparse_factors, kernel_regularizer=keras.regularizers.l2(weight_decay), kernel_initializer=he_normal(), name='predictions_mnist'))
    model.add(BatchNormalization())
    model.add(Activation('softmax'))
    return model

def pbp_lenet_model(input_shape, num_classes, sparsity_factor, nb_sparse_factors, permutation=True, weight_decay=1e-4):
    raise NotImplementedError
    model = Sequential()
    model.add(RandomSparseFactorisationConv2D(input_shape=input_shape, permutation=permutation, filters=6, kernel_size=(5, 5), activation='relu', kernel_initializer=he_normal(), padding="valid", sparsity_factor=sparsity_factor, nb_sparse_factors=nb_sparse_factors))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(RandomSparseFactorisationConv2D(input_shape=input_shape, permutation=permutation, filters=16, kernel_size=(5, 5), activation='relu', kernel_initializer=he_normal(), padding="valid", sparsity_factor=sparsity_factor, nb_sparse_factors=nb_sparse_factors))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(Flatten(name='flatten'))
    model.add(RandomSparseFactorisationDense(1024, permutation=permutation, sparsity_factor=sparsity_factor, nb_sparse_factors=nb_sparse_factors, use_bias=True, kernel_regularizer=keras.regularizers.l2(weight_decay), kernel_initializer=he_normal(), name='fc1'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(RandomSparseFactorisationDense(num_classes, permutation=permutation, sparsity_factor=sparsity_factor, nb_sparse_factors=nb_sparse_factors, kernel_regularizer=keras.regularizers.l2(weight_decay), kernel_initializer=he_normal(), name='predictions_mnist'))
    model.add(BatchNormalization())
    model.add(Activation('softmax'))
    return model

def sparse_random_vgg19_model(input_shape, num_classes, sparsity_factor, nb_sparse_factors, weight_decay=1e-4, dropout=0.5, size_denses=2048, permutation=True):
    model = Sequential()

    # Block 1
    model.add(RandomSparseFactorisationConv2D(filters=64, kernel_size=(3, 3), permutation=permutation, sparsity_factor=sparsity_factor, nb_sparse_factors=nb_sparse_factors, padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay), kernel_initializer=he_normal(), name='block1_conv1', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(RandomSparseFactorisationConv2D(filters=64, kernel_size=(3, 3), permutation=permutation, sparsity_factor=sparsity_factor, nb_sparse_factors=nb_sparse_factors, padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay), kernel_initializer=he_normal(), name='block1_conv2'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool'))

    # Block 2
    model.add(RandomSparseFactorisationConv2D(filters=128, kernel_size=(3, 3), permutation=permutation, sparsity_factor=sparsity_factor, nb_sparse_factors=nb_sparse_factors, padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay), kernel_initializer=he_normal(), name='block2_conv1'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(RandomSparseFactorisationConv2D(filters=128, kernel_size=(3, 3), permutation=permutation, sparsity_factor=sparsity_factor, nb_sparse_factors=nb_sparse_factors, padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay), kernel_initializer=he_normal(), name='block2_conv2'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool'))

    # Block 3
    model.add(RandomSparseFactorisationConv2D(filters=256, kernel_size=(3, 3), permutation=permutation, sparsity_factor=sparsity_factor, nb_sparse_factors=nb_sparse_factors, padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay), kernel_initializer=he_normal(), name='block3_conv1'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(RandomSparseFactorisationConv2D(filters=256, kernel_size=(3, 3), permutation=permutation, sparsity_factor=sparsity_factor, nb_sparse_factors=nb_sparse_factors, padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay), kernel_initializer=he_normal(), name='block3_conv2'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(RandomSparseFactorisationConv2D(filters=256, kernel_size=(3, 3), permutation=permutation, sparsity_factor=sparsity_factor, nb_sparse_factors=nb_sparse_factors, padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay), kernel_initializer=he_normal(), name='block3_conv3'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(RandomSparseFactorisationConv2D(filters=256, kernel_size=(3, 3), permutation=permutation, sparsity_factor=sparsity_factor, nb_sparse_factors=nb_sparse_factors, padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay), kernel_initializer=he_normal(), name='block3_conv4'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool'))

    # Block 4
    model.add(RandomSparseFactorisationConv2D(filters=512, kernel_size=(3, 3), permutation=permutation, sparsity_factor=sparsity_factor, nb_sparse_factors=nb_sparse_factors, padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay), kernel_initializer=he_normal(), name='block4_conv1'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(RandomSparseFactorisationConv2D(filters=512, kernel_size=(3, 3), permutation=permutation, sparsity_factor=sparsity_factor, nb_sparse_factors=nb_sparse_factors, padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay), kernel_initializer=he_normal(), name='block4_conv2'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(RandomSparseFactorisationConv2D(filters=512, kernel_size=(3, 3), permutation=permutation, sparsity_factor=sparsity_factor, nb_sparse_factors=nb_sparse_factors, padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay), kernel_initializer=he_normal(), name='block4_conv3'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(RandomSparseFactorisationConv2D(filters=512, kernel_size=(3, 3), permutation=permutation, sparsity_factor=sparsity_factor, nb_sparse_factors=nb_sparse_factors, padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay), kernel_initializer=he_normal(), name='block4_conv4'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool'))

    # Block 5
    model.add(RandomSparseFactorisationConv2D(filters=512, kernel_size=(3, 3), permutation=permutation, sparsity_factor=sparsity_factor, nb_sparse_factors=nb_sparse_factors, padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay), kernel_initializer=he_normal(), name='block5_conv1'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(RandomSparseFactorisationConv2D(filters=512, kernel_size=(3, 3), permutation=permutation, sparsity_factor=sparsity_factor, nb_sparse_factors=nb_sparse_factors, padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay), kernel_initializer=he_normal(), name='block5_conv2'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(RandomSparseFactorisationConv2D(filters=512, kernel_size=(3, 3), permutation=permutation, sparsity_factor=sparsity_factor, nb_sparse_factors=nb_sparse_factors, padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay), kernel_initializer=he_normal(), name='block5_conv3'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(RandomSparseFactorisationConv2D(filters=512, kernel_size=(3, 3), permutation=permutation, sparsity_factor=sparsity_factor, nb_sparse_factors=nb_sparse_factors, padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay), kernel_initializer=he_normal(), name='block5_conv4'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool'))

    model.add(Flatten(name='flatten'))
    model.add(RandomSparseFactorisationDense(size_denses, permutation=permutation, sparsity_factor=sparsity_factor, nb_sparse_factors=nb_sparse_factors, use_bias=True, kernel_regularizer=keras.regularizers.l2(weight_decay), kernel_initializer=he_normal(), name='fc1'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(dropout))
    model.add(RandomSparseFactorisationDense(size_denses, permutation=permutation, sparsity_factor=sparsity_factor, nb_sparse_factors=nb_sparse_factors, kernel_regularizer=keras.regularizers.l2(weight_decay), kernel_initializer=he_normal(), name='fc2'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(dropout))
    model.add(RandomSparseFactorisationDense(num_classes, permutation=permutation, sparsity_factor=sparsity_factor, nb_sparse_factors=nb_sparse_factors, kernel_regularizer=keras.regularizers.l2(weight_decay), kernel_initializer=he_normal(), name='predictions'))
    model.add(BatchNormalization())
    model.add(Activation('softmax'))

    return model