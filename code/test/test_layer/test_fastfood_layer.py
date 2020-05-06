import unittest

from keras.optimizers import Adam
from keras.datasets import cifar10

from palmnet.layers.fastfood_layer import FastFoodLayer
from palmnet.layers.low_rank_dense_layer import LowRankDense
from palmnet.layers.tt_layer_conv import TTLayerConv
from palmnet.layers.tt_layer_dense import TTLayerDense
from keras.models import Sequential
from keras.layers import Conv2D, GlobalAveragePooling2D, Dense, Flatten

from palmnet.utils import get_nb_learnable_weights
import numpy as np


class TestFastFoodLayers(unittest.TestCase):
    def setUp(self) -> None:
        # (self.X_train, self.y_train), (self.X_test, self.y_test) = cifar10.load_data()
        self.X_train = np.random.rand(100, 8, 8, 1)

    @staticmethod
    def build_model(rank=12):
        model = Sequential()
        input_shape = (8, 8, 1)
        model.add(Flatten(input_shape=input_shape))
        model.add(FastFoodLayer(nbr_stack=2))
        model.add(FastFoodLayer(nb_units=128))
        model.add(FastFoodLayer(nb_units=512))
        model.add(FastFoodLayer(nb_units=10))
        model.add(FastFoodLayer(nb_units=34))
        return model

    def test_low_rank_layers_fine(self):
        rank=24
        model = self.build_model(rank=rank)
        model.compile(Adam(), loss="mse")
        result = model.predict(self.X_train[:10])
        assert result.shape == (10, 34)

        weights_last_layer = model.layers[-1].get_weights()
        assert weights_last_layer[-1].shape == (34,)  # asked dimension
        assert weights_last_layer[-2].shape == (34,)  # asked dimension
        assert weights_last_layer[0].shape[-1] == 16  # first power of 2 above 10

