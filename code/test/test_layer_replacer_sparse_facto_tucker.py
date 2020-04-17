from copy import deepcopy
import tempfile
import pathlib
import unittest
from keras.layers import Dense, Conv2D
from keras.models import Sequential
from keras.optimizers import Adam

from palmnet.core.faustizer import Faustizer
from palmnet.core.layer_replacer_sparse_facto_tucker import LayerReplacerSparseFactoTucker
from palmnet.core.layer_replacer_sparse_facto_tucker_faust import LayerReplacerSparseFactoTuckerFaust
from palmnet.data import Cifar100, Mnist


class TestLayerReplacerTucker(unittest.TestCase):

    def setUp(self) -> None:
        self.base_model = Cifar100.load_model("cifar100_vgg19_2048x2048")

        (self.X_train, self.y_train), (self.X_test, self.y_test) = Cifar100.load_data()

    def test_simple(self):
        faustizer = Faustizer(sparsity_fac=2,
                              nb_factor=2,
                              nb_iter=2,
                              tol=1e-6,
                              hierarchical=False)
        with tempfile.TemporaryDirectory() as tmpdirname:
            path_to_checkpoint = pathlib.Path(tmpdirname) / "checkpoint"
            path_to_checkpoint.mkdir(parents=True)
            model_transformer = LayerReplacerSparseFactoTuckerFaust(sparse_factorizer=faustizer,
                                                               path_checkpoint_file=path_to_checkpoint)
            model_transformer.fit(deepcopy(self.base_model))
            del model_transformer

            model_transformer_bis = LayerReplacerSparseFactoTuckerFaust(sparse_factorizer=faustizer,
                                                               path_checkpoint_file=path_to_checkpoint)
            model_transformer_bis.load_dct_name_compression()
            new_model = model_transformer_bis.fit_transform(deepcopy(self.base_model))
            print(new_model)
            new_model.compile(Adam(), loss="mse")
            result = new_model.predict(self.X_train[:10])




if __name__ == '__main__':
    unittest.main()
