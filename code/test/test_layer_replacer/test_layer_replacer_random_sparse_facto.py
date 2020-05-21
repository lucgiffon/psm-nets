import signal
import os
import os.path
import unittest
from copy import deepcopy
import pathlib

from palmnet.core.faustizer import Faustizer
from palmnet.core.layer_replacer_palm import LayerReplacerPalm
from palmnet.core.layer_replacer_random_sparse_facto import LayerReplacerRandomSparseFacto
from palmnet.core.palminizer import Palminizer
from palmnet.core.palminizable import Palminizable
from palmnet.core.randomizer import Randomizer
from palmnet.data import Cifar100, Mnist
from pprint import pprint
import numpy as np
from keras.layers import Dense, Conv2D
import tempfile

from palmnet.layers.sparse_facto_conv2D_masked import SparseFactorisationConv2D
from palmnet.layers.sparse_facto_dense_masked import SparseFactorisationDense
from palmnet.utils import get_idx_last_layer_of_class, timeout_signal_handler
from qkmeans.utils import log_memory_usage


class TestLayerReplacerRandomSparseFacto(unittest.TestCase):

    def setUp(self) -> None:
        self.base_model = Mnist.load_model("mnist_lenet")
        self.randomizer = Randomizer(sparsity_fac=2,
                                nb_factor=2,
                                fast_unstable_proj=False)

    def test_fit_transform(self) -> None:
        model_transformer = LayerReplacerRandomSparseFacto(sparse_factorizer=self.randomizer,
                                                           keep_last_layer=False,
                                                           only_mask=True, dct_name_compression=None)
        new_model = model_transformer.fit_transform(deepcopy(self.base_model))
        for l in new_model.layers:
            if isinstance(l, SparseFactorisationDense) or isinstance(l, SparseFactorisationConv2D):
                weights = l.get_weights()
                if l.use_bias:
                    weights = weights[:-1]
                assert len(weights) == self.randomizer.nb_factor
                for w in weights:
                    B = max(w.shape)
                    nnz = np.sum(w.astype(bool))
                    assert self.randomizer.sparsity_fac * B <= nnz <= 2 * self.randomizer.sparsity_fac * B
                prod_w = np.linalg.multi_dot(weights)
                X = np.random.randn(1000, prod_w.shape[0])
                res = X @ prod_w
                print(np.mean(np.std(res, axis=0)))
                print(np.mean(np.std(X, axis=0)))

                print(new_model)
            elif isinstance(l, Dense) or isinstance(l, Conv2D):
                assert False


if __name__ == '__main__':
    unittest.main()
