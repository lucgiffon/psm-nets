import unittest
from copy import deepcopy

from tensorly.decomposition import partial_tucker

from palmnet.core.layer_replacer_TT import LayerReplacerTT
from palmnet.core.layer_replacer_multi import LayerReplacerMulti
from palmnet.core.layer_replacer_tucker import LayerReplacerTucker
from palmnet.data import Mnist
import numpy as np
from tensorly.tenalg.n_mode_product import multi_mode_dot
import tensorflow as tf

from palmnet.utils import translate_keras_to_tf_model


class TestLayerReplacerTT(unittest.TestCase):

    def setUp(self) -> None:
        self.base_model = translate_keras_to_tf_model(Mnist.load_model("mnist_lenet"))

    def test_simple(self):
        model_transformer = LayerReplacerMulti(nb_factors=2, keras_module=tf.keras)
        new_model = model_transformer.fit_transform(self.base_model)
        print(new_model.predict(np.random.rand(1, 28, 28, 1)))


if __name__ == '__main__':
    unittest.main()
