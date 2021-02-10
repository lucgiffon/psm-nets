import signal
import os
import os.path
import unittest
from copy import deepcopy
import pathlib

from palmnet.core.faustizer import Faustizer
from palmnet.core.layer_replacer_palm import LayerReplacerPalm
from palmnet.core.layer_replacer_sparsefacto2dense import LayerReplacerSparseFacto2Dense
from palmnet.core.palminizer import Palminizer
from palmnet.core.palminizable import Palminizable
from palmnet.data import Cifar100, Mnist
from pprint import pprint
import numpy as np
from keras.layers import Dense
import tempfile

from palmnet.utils import get_idx_last_layer_of_class, timeout_signal_handler
from qkmeans.utils import log_memory_usage


class TestLayerReplacerSparseFacto2Dense(unittest.TestCase):

    def setUp(self) -> None:
        self.base_model = Mnist.load_model("mnist_lenet")
        self.palminizer = Palminizer(sparsity_fac=2,
                                nb_factor=2,
                                nb_iter=2,
                                delta_threshold_palm=1e-6,
                                hierarchical=False,
                                fast_unstable_proj=False)


    def test_fit(self):
        base_model_transformer = LayerReplacerPalm(sparse_factorizer=self.palminizer, keep_last_layer=True, only_mask=False, dct_name_compression=None)
        new_base_model = base_model_transformer.fit_transform(deepcopy(self.base_model))

        model_transformer = LayerReplacerSparseFacto2Dense(keep_last_layer=True, dct_name_compression=None)
        # model_transformer.fit(new_base_model)
        # new_model = model_transformer.transform(new_base_model)
        new_model = model_transformer.fit_transform(new_base_model)
        print(new_model)

if __name__ == '__main__':
    unittest.main()
