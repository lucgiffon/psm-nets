import unittest
from copy import deepcopy

from palmnet.core.faustizer import Faustizer
from palmnet.core.layer_replacer_faust import LayerReplacerFaust
from palmnet.core.layer_replacer_palm import LayerReplacerPalm
from palmnet.core.palminizer import Palminizer
from palmnet.core.palminizable import Palminizable
from palmnet.data import Cifar100, Mnist
from pprint import pprint
import numpy as np
from keras.layers import Dense

from palmnet.utils import get_idx_last_layer_of_class


class TestLayerReplacerPalm(unittest.TestCase):

    def setUp(self) -> None:
        self.base_model = Mnist.load_model("mnist_lenet")


    def test_fit_transform(self) -> None:
        for hierarchical in [True, False]:
            faustizer = Faustizer(sparsity_fac=2,
                                  nb_factor=2,
                                  nb_iter=2,
                                  tol=1e-6,
                                  hierarchical=hierarchical)
            model_transformer = LayerReplacerFaust(sparse_factorizer=faustizer, keep_last_layer=True, only_mask=False, dct_name_compression=None)
            new_model = model_transformer.fit_transform(deepcopy(self.base_model))


if __name__ == '__main__':
    unittest.main()
