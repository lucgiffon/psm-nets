import unittest
import numpy as np
from keras.optimizers import Adam
from keras.datasets import cifar10

from keras.models import Sequential
from keras.layers import Conv2D, GlobalAveragePooling2D, Dense

from palmnet.layers.sparse_facto_conv2D_masked import SparseFactorisationConv2D
from palmnet.layers.sparse_facto_dense_masked import SparseFactorisationDense
from palmnet.utils import get_nb_learnable_weights, create_sparse_factorization_pattern, NAME_INIT_SPARSE_FACTO


class TestSparseFactoLayers(unittest.TestCase):
    def setUp(self) -> None:
        (self.X_train, self.y_train), (self.X_test, self.y_test) = cifar10.load_data()

    @staticmethod
    def build_model(sparse_pattern_conv, sparse_pattern_dense):
        input_shape = (32, 32, 3)
        kernel_size = 3
        filters = 512
        model = Sequential()

        model.add(Conv2D(512, (3, 3), padding='same', input_shape=input_shape, name="conv2D_1"))
        model.add(SparseFactorisationConv2D(kernel_size=[kernel_size, kernel_size], sparsity_patterns=sparse_pattern_conv, filters=filters, name="sparsefactoconv1", padding="same",
                                            kernel_initializer=NAME_INIT_SPARSE_FACTO))
        model.add(Conv2D(512, (3, 3), padding='same', input_shape=input_shape, name="conv2D_2"))
        model.add(GlobalAveragePooling2D())
        model.add(Dense(512, name="dense1", use_bias=False))
        model.add(SparseFactorisationDense(units=filters,sparsity_patterns=sparse_pattern_dense, use_scaling=False, use_bias=False, name="sparsefactodense1",
                                            kernel_initializer=NAME_INIT_SPARSE_FACTO))
        model.add(Dense(512, name="dense2"))
        return model


    def test_sparse_facto_masked_fine(self):

        kernel_size = 3
        filters = 512
        sparse_pattern_conv = create_sparse_factorization_pattern((filters*kernel_size*kernel_size, filters), block_size=2, nb_factors=2)
        sparse_pattern_dense = create_sparse_factorization_pattern((filters, filters), block_size=3, nb_factors=3)

        model = self.build_model(sparse_pattern_conv, sparse_pattern_dense)
        model.compile(Adam(), loss="mse")
        result = model.predict(self.X_train[:10])
        assert result.shape == (10, 512)

        nb_val_dense_sparse_facto = np.sum(np.sum(elm) for elm in sparse_pattern_dense)
        nb_val_conv_sparse_facto = np.sum(np.sum(elm) for elm in sparse_pattern_conv)

        dct_layer_name_expected_nb_weights = {
            "conv2D_1": kernel_size*kernel_size*kernel_size*filters + filters,
            "sparsefactoconv1": nb_val_conv_sparse_facto + filters + 1,
            "sparsefactodense1": nb_val_dense_sparse_facto,
            "conv2D_2": kernel_size * kernel_size * filters * filters + filters,
            "global_average_pooling2d_1": 0,
            "dense1": filters*filters,
            "dense2": filters*filters+filters
        }

        for layer in model.layers:
            found = get_nb_learnable_weights(layer)
            expected = dct_layer_name_expected_nb_weights[layer.name]
            print(found, expected)
            assert found  == expected, f"not good {found}!={expected} {layer.name} {found + filters}"

            if isinstance(layer, SparseFactorisationConv2D):
                assert (layer.get_weights()[1].astype(bool) == sparse_pattern_conv[0].astype(bool)).all()
                assert (layer.get_weights()[2].astype(bool) == sparse_pattern_conv[1].astype(bool)).all()
            if isinstance(layer, SparseFactorisationDense):
                assert (layer.get_weights()[0].astype(bool) == sparse_pattern_dense[0].astype(bool)).all()
                assert (layer.get_weights()[1].astype(bool) == sparse_pattern_dense[1].astype(bool)).all()
                assert (layer.get_weights()[2].astype(bool) == sparse_pattern_dense[2].astype(bool)).all()
