import unittest

from keras.optimizers import Adam
from keras.datasets import cifar10

from palmnet.layers.low_rank_dense_layer import LowRankDense
from palmnet.layers.tt_layer_conv import TTLayerConv
from palmnet.layers.tt_layer_dense import TTLayerDense
from keras.models import Sequential
from keras.layers import Conv2D, GlobalAveragePooling2D, Dense, Flatten

from palmnet.utils import get_nb_learnable_weights


class TestTTLayers(unittest.TestCase):
    def setUp(self) -> None:
        (self.X_train, self.y_train), (self.X_test, self.y_test) = cifar10.load_data()

    @staticmethod
    def build_model(rank=12):
        model = Sequential()
        input_shape = (32, 32, 3)
        model.add(Flatten(input_shape=input_shape))
        model.add(LowRankDense(512, rank))
        return model

    def test_low_rank_layers_fine(self):
        rank=24
        model = self.build_model(rank=rank)
        model.compile(Adam(), loss="mse")
        result = model.predict(self.X_train[:10])
        assert result.shape == (10, 512)

        low_rank_weights = model.layers[-1].get_weights()
        assert len(low_rank_weights) == 3
        assert low_rank_weights[-1].shape == (512,)
        assert low_rank_weights[0].shape[-1] == rank
        assert low_rank_weights[1].shape == (rank, 512)

