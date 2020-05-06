import unittest
from copy import deepcopy

from tensorly.decomposition import partial_tucker

from palmnet.core.layer_replacer_deepfried import LayerReplacerDeepFried
from palmnet.core.layer_replacer_tucker import LayerReplacerTucker
from palmnet.data import Mnist
import numpy as np
from tensorly.tenalg.n_mode_product import multi_mode_dot


class TestLayerReplacerTucker(unittest.TestCase):

    def setUp(self) -> None:
        self.base_model = Mnist.load_model("mnist_lenet")

    def test_simple(self):
        model_transformer = LayerReplacerDeepFried(only_dense=True)
        new_model = model_transformer.fit_transform(deepcopy(self.base_model))
        print(new_model)

if __name__ == '__main__':
    unittest.main()
