import unittest

from keras.optimizers import Adam

from palmnet.layers.tt_layer_conv import TTLayerConv
from palmnet.layers.tt_layer_dense import TTLayerDense
from keras.models import Sequential
from keras.layers import Conv2D, GlobalAveragePooling2D, Dense

from palmnet.layers.tucker_layer import TuckerLayerConv
from keras.datasets import cifar10

class TestTTLayers(unittest.TestCase):
    def setUp(self) -> None:
        (self.X_train, self.y_train), (self.X_test, self.y_test) = cifar10.load_data()

    @staticmethod
    def build_model():
        model = Sequential()
        input_shape = (32, 32, 3)

        model.add(Conv2D(512, (3, 3), padding='same', input_shape=input_shape))
        model.add(TuckerLayerConv(kernel_size=[3, 3], in_rank=128, out_rank=128, filters=512, name="tuckerlayerconv_1", padding="same"))
        model.add(TuckerLayerConv(kernel_size=[3, 3], in_rank=128, out_rank=128, filters=512, name="tuckerlayerconv_2", padding="same"))
        model.add(TuckerLayerConv(kernel_size=[3, 3], in_rank=128, out_rank=128, filters=512, name="tuckerlayerconv_3", padding="same"))
        model.add(Conv2D(512, (3, 3), padding='same', input_shape=input_shape))
        return model


    def test_tuckerconv_layer(self):
        model = self.build_model()
        model.compile(Adam(), loss="mse")
        result = model.predict(self.X_train[:10])
        assert result.shape == (10, 32, 32, 512)



