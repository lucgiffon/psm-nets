import unittest

from palmnet.data import Mnist
import numpy as np
import keras.backend as K

from palmnet.utils import get_nb_learnable_weights_from_model


class TestCountParams(unittest.TestCase):
    def setUp(self) -> None:
        self.base_model = Mnist.load_model("mnist_lenet")

    def test_count_params(self):
        count_param = self.base_model.count_params()
        counted = get_nb_learnable_weights_from_model(self.base_model)
        assert count_param == counted


if __name__ == '__main__':
    unittest.main()
