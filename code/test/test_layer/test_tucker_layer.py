import unittest

from keras.optimizers import Adam

from palmnet.layers.tt_layer_conv import TTLayerConv
from palmnet.layers.tt_layer_dense import TTLayerDense
from keras.models import Sequential
from keras.layers import Conv2D, GlobalAveragePooling2D, Dense

from palmnet.layers.tucker_layer import TuckerLayerConv
from keras.datasets import cifar10

from palmnet.utils import get_nb_learnable_weights


class TestTTLayers(unittest.TestCase):
    def setUp(self) -> None:
        (self.X_train, self.y_train), (self.X_test, self.y_test) = cifar10.load_data()

    @staticmethod
    def build_model():
        model = Sequential()
        input_shape = (32, 32, 3)

        model.add(Conv2D(512, (3, 3), padding='same', input_shape=input_shape, name="conv2D_1", use_bias=True))
        model.add(TuckerLayerConv(kernel_size=[3, 3], in_rank=128, out_rank=128, filters=512, name="tuckerlayerconv_1", padding="same", use_bias=True))
        model.add(TuckerLayerConv(kernel_size=[3, 3], in_rank=128, out_rank=128, filters=512, name="tuckerlayerconv_2", padding="same", use_bias=True))
        model.add(TuckerLayerConv(kernel_size=[3, 3], in_rank=128, out_rank=128, filters=512, name="tuckerlayerconv_3", padding="same", use_bias=False))
        model.add(Conv2D(512, (3, 3), padding='same', input_shape=input_shape, name="conv2D_2", use_bias=False))
        return model


    def test_tuckerconv_layer(self):
        model = self.build_model()
        model.compile(Adam(), loss="mse")
        result = model.predict(self.X_train[:10])
        assert result.shape == (10, 32, 32, 512)

        dct_layer_name_expected_nb_weights = {
            "conv2D_1": 3*3*3*512 + 512,
            "tuckerlayerconv_1": 3*3*128*128 + 512*128 + 128*512 + 512,
            "tuckerlayerconv_2": 3*3*128*128 + 512*128 + 128*512 + 512,
            "tuckerlayerconv_3": 3*3*128*128 + 512*128 + 128*512,
            "conv2D_2": 3 * 3 * 512 * 512,
        }

        for layer in model.layers:
            found = get_nb_learnable_weights(layer)
            expected = dct_layer_name_expected_nb_weights[layer.name]
            print(found, expected)
            assert found  == expected, f"not good {found}!={expected} {layer.name}"



