import unittest

from keras.optimizers import Adam
from keras.datasets import cifar10

from palmnet.layers.tt_layer_conv import TTLayerConv
from palmnet.layers.tt_layer_dense import TTLayerDense
from keras.models import Sequential
from keras.layers import Conv2D, GlobalAveragePooling2D, Dense

class TestTTLayers(unittest.TestCase):
    def setUp(self) -> None:
        (self.X_train, self.y_train), (self.X_test, self.y_test) = cifar10.load_data()

    @staticmethod
    def build_model(tt_ranks_conv = (2, 2, 2, 2, 1), tt_ranks_dense = (1, 2, 2, 2, 1)):
        model = Sequential()
        input_shape = (32, 32, 3)

        model.add(Conv2D(512, (3, 3), padding='same', input_shape=input_shape))
        model.add(TTLayerConv(kernel_size=[3, 3], filters=512, inp_modes=[4, 4, 4, 8], out_modes=[4, 4, 4, 8], mat_ranks=tt_ranks_conv, mode="manual", name="ttlayerconv_1", padding="same"))
        model.add(TTLayerConv(kernel_size=[3, 3], filters=512, inp_modes=[4, 4, 4, 8], out_modes=[4, 4, 4, 8], mat_ranks=tt_ranks_conv, mode="auto", name="ttlayerconv_2", padding="same"))
        model.add(TTLayerConv(kernel_size=[3, 3], filters=512, out_modes=[4, 4, 4, 8], mat_ranks=tt_ranks_conv, mode="auto", name="ttlayerconv_3", padding="same"))
        model.add(TTLayerConv(kernel_size=[3, 3], filters=512, inp_modes=[4, 4, 4, 8], mat_ranks=tt_ranks_conv, mode="auto", name="ttlayerconv_4", padding="same"))
        model.add(TTLayerConv(kernel_size=[3, 3], filters=512, mat_ranks=tt_ranks_conv, mode="auto", name="ttlayerconv_5"))
        model.add(Conv2D(512, (3, 3), padding='same', input_shape=input_shape))
        model.add(GlobalAveragePooling2D())
        model.add(Dense(512))
        model.add(TTLayerDense(nb_units=512, inp_modes=[4, 4, 4, 8], out_modes=[4, 4, 4, 8], mat_ranks=tt_ranks_dense, mode="manual", name="ttlayerdense_1"))
        model.add(TTLayerDense(nb_units=512, inp_modes=[4, 4, 4, 8], out_modes=[4, 4, 4, 8], mat_ranks=tt_ranks_dense, mode="auto", name="ttlayerdense_2"))
        model.add(TTLayerDense(nb_units=512, out_modes=[4, 4, 4, 8], mat_ranks=tt_ranks_dense, mode="auto", name="ttlayerdense_3"))
        model.add(TTLayerDense(nb_units=512, inp_modes=[4, 4, 4, 8], mat_ranks=tt_ranks_dense, mode="auto", name="ttlayerdense_4"))
        model.add(TTLayerDense(nb_units=512, mat_ranks=tt_ranks_dense, mode="auto", name="ttlayerdense_5"))
        model.add(Dense(512))
        return model

    @staticmethod
    def build_model_full_auto(nb_dim, tt_ranks_conv = (2, 2, 2, 2, 1), tt_ranks_dense = (1, 2, 2, 2, 1)):
        model = Sequential()
        input_shape = (32, 32, 3)

        model.add(Conv2D(nb_dim, (3, 3), padding='same', input_shape=input_shape))
        model.add(TTLayerConv(kernel_size=[3, 3], filters=nb_dim, mat_ranks=tt_ranks_conv, mode="auto", name="ttlayerconv", padding="same"))
        model.add(Conv2D(nb_dim, (3, 3), padding='same'))
        model.add(GlobalAveragePooling2D())
        model.add(Dense(nb_dim))
        model.add(TTLayerDense(nb_units=nb_dim, mat_ranks=tt_ranks_dense, mode="auto", name="ttlayerdense"))
        model.add(Dense(nb_dim))
        return model

    def test_tt_layers_fine(self):
        model = self.build_model()
        model.compile(Adam(), loss="mse")
        result = model.predict(self.X_train[:10])
        assert result.shape == (10, 512)

        layers = model.layers
        modes = tuple([4, 4, 4, 8])

        for idx_layer, layer in enumerate(layers[1: 6]):
            idx_layer += 1
            assert isinstance(layer, TTLayerConv), "Layers of index {} should be TTLayerConv. {}".format(idx_layer, layer.name)

            assert tuple(layer.inp_modes) == modes, "Bad inp modes layer {} name: {}".format(idx_layer, layer.name)
            assert tuple(layer.out_modes) == modes, "Bad out modes layer {} name: {}".format(idx_layer, layer.name)


        for idx_layer, layer in enumerate(layers[9: 14]):
            idx_layer += 1
            assert isinstance(layer, TTLayerDense), "Layers of index {} should be TTLayerConv. {}".format(idx_layer, layer.name)

            assert tuple(layer.inp_modes) == modes, "Bad inp modes layer {} name: {}".format(idx_layer, layer.name)
            assert tuple(layer.out_modes) == modes, "Bad out modes layer {} name: {}".format(idx_layer, layer.name)

    def test_tt_layers_full_auto(self):

        for tt_ranks in [(2, 2, 2, 1), (2, 2, 1)]:
            tt_rank_conv = (2, *tt_ranks)
            tt_rank_dense = (1, *tt_ranks)

            order = len(tt_rank_conv) - 1
            dim = 4 ** order
            expected_prod = tuple([4] * order)
            model = self.build_model_full_auto(dim, tt_rank_conv, tt_rank_dense)

            for layer in model.layers:
                if isinstance(layer, TTLayerConv) or isinstance(layer, TTLayerDense):
                    assert tuple(layer.inp_modes) == expected_prod, "Bad inp modes layer name: {}".format(layer.name)
                    assert tuple(layer.out_modes) == expected_prod, "Bad out modes layer name: {}".format(layer.name)

    def test_tt_layer_crash(self):
        # not power of two
        self.assertRaises(Exception, lambda: self.build_model_full_auto(29))

