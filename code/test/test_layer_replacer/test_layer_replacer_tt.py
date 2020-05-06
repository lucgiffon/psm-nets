import unittest
from copy import deepcopy

from tensorly.decomposition import partial_tucker

from palmnet.core.layer_replacer_TT import LayerReplacerTT
from palmnet.core.layer_replacer_tucker import LayerReplacerTucker
from palmnet.data import Mnist
import numpy as np
from tensorly.tenalg.n_mode_product import multi_mode_dot


class TestLayerReplacerTT(unittest.TestCase):

    def setUp(self) -> None:
        self.base_model = Mnist.load_model("mnist_lenet")

    def test_simple(self):
        model_transformer = LayerReplacerTT(rank_value=2, order=4)
        new_model = model_transformer.fit_transform(deepcopy(self.base_model))

    def test_simple_decompo(self):
        model_transformer = LayerReplacerTT(rank_value=2, order=4, use_pretrained=True)
        new_model = model_transformer.fit_transform(deepcopy(self.base_model))

        # model_transformer = LayerReplacerTucker(rank_percentage_dense=0.5, keep_last_layer=True)
        # new_model = model_transformer.fit_transform(deepcopy(self.base_model))

if __name__ == '__main__':
    unittest.main()
