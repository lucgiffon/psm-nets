import unittest
from copy import deepcopy

from tensorly.decomposition import partial_tucker

from palmnet.core.layer_replacer_deepfried import LayerReplacerDeepFried
from palmnet.core.layer_replacer_magnitude_pruning import LayerReplacerMagnitudePruning
from palmnet.core.layer_replacer_tucker import LayerReplacerTucker
from palmnet.data import Mnist
import numpy as np
from tensorly.tenalg.n_mode_product import multi_mode_dot
import tensorflow as tf

from palmnet.utils import translate_keras_to_tf_model


class TestLayerReplacerPruneMagnitude(unittest.TestCase):

    def setUp(self) -> None:
        self.base_model = Mnist.load_model("mnist_lenet")

    def test_simple(self):
        model_transformer = LayerReplacerMagnitudePruning(final_sparsity=0.9, end_step=5000, keras_module=tf.keras)
        tf_model = translate_keras_to_tf_model(self.base_model)
        new_model = model_transformer.fit_transform(tf_model)
        print(new_model)

if __name__ == '__main__':
    unittest.main()
