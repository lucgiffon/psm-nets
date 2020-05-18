import unittest
from copy import deepcopy

from keras.optimizers import Adam
from tensorly.decomposition import partial_tucker

from palmnet.core.layer_replacer_deepfried import LayerReplacerDeepFried
from palmnet.core.layer_replacer_magnitude_pruning import LayerReplacerMagnitudePruning
from palmnet.core.layer_replacer_tucker import LayerReplacerTucker
from palmnet.data import Mnist
import numpy as np
from tensorly.tenalg.n_mode_product import multi_mode_dot

from palmnet.utils import translate_keras_to_tf_model


class TestTranslateKerasToTf(unittest.TestCase):

    def setUp(self) -> None:
        self.base_model = Mnist.load_model("mnist_lenet")
        self.X_train = np.random.rand(100, 28, 28, 1)

    def test_simple(self):
        tf_model = translate_keras_to_tf_model(self.base_model)
        self.base_model.compile("adam", loss="mse")
        base_result = self.base_model.predict(self.X_train[:10])
        tf_model.compile("adam", loss="mse")
        tf_result = tf_model.predict(self.X_train[:10])
        assert (base_result == tf_result).all()

if __name__ == '__main__':
    unittest.main()
