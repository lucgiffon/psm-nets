import signal
import os
import os.path
import unittest
from copy import deepcopy
import pathlib

from palmnet.core.faustizer import Faustizer
from palmnet.core.layer_replacer_palm import LayerReplacerPalm
from palmnet.core.palminizer import Palminizer
from palmnet.core.palminizable import Palminizable
from palmnet.data import Cifar100, Mnist
from pprint import pprint
import numpy as np
from keras.layers import Dense
import tempfile

from palmnet.utils import get_idx_last_layer_of_class, timeout_signal_handler


class TestLayerReplacerPalm(unittest.TestCase):

    def setUp(self) -> None:
        self.base_model = Mnist.load_model("mnist_lenet")
        self.palminizer = Palminizer(sparsity_fac=2,
                                nb_factor=2,
                                nb_iter=2,
                                delta_threshold_palm=1e-6,
                                hierarchical=False,
                                fast_unstable_proj=False)

        self.palminizable = Palminizable(deepcopy(self.base_model), self.palminizer)
        self.palminizable.palminize()
        pprint(self.palminizable.sparsely_factorized_layers)
        self.dct_sparsely_factorized_layers = self.palminizable.sparsely_factorized_layers

    def test_transform(self) -> None:
        for keep_last_layer in [True, False]:
            for only_mask in [True, False]:
                dct_name_facto = self.dct_sparsely_factorized_layers

                model_transformer = LayerReplacerPalm(keep_last_layer=keep_last_layer, only_mask=only_mask, dct_name_compression=dct_name_facto)
                new_model = model_transformer.transform(self.base_model)

                idx_last_dense = get_idx_last_layer_of_class(new_model)
                if keep_last_layer:
                    # test pour verifier que new model a bien une couche Dense (sinon c'est -1)
                    assert idx_last_dense != -1, "When keep last layer, there"
                atleast_one = False
                for idx_layer, new_layer in enumerate(new_model.layers):
                    old_layer_name = model_transformer.get_replaced_layer_name(new_layer.name)
                    if model_transformer.have_been_replaced(old_layer_name):
                        atleast_one = True
                        new_layer_w = [w for w in new_layer.get_weights() if len(w.shape) > 1]
                        sparse_facto = dct_name_facto[old_layer_name]
                        sparse_facto_w = [fac.toarray() for fac in sparse_facto[1].get_list_of_factors()]

                        for i_weight, w_new in enumerate(new_layer_w):
                            w_sparse_facto = sparse_facto_w[i_weight]
                            # test pour vérifier que les shape sont bien les meme avant de les comaprer
                            assert w_new.shape == w_sparse_facto.shape, "Weights shape are not the same"
                            # arr_bool_comparison = (w_new == w_sparse_facto)
                            # bool_all_equal = arr_bool_comparison.all()
                            bool_all_equal = np.allclose(w_new, w_sparse_facto)
                            if only_mask:
                                assert not bool_all_equal, "Weights should be different when only_mask is True."
                            else:
                                assert bool_all_equal, "Weights should be the same when only_mask is False."

                    if (keep_last_layer and idx_layer == idx_last_dense):
                        assert isinstance(new_layer, Dense), "Last dense layer should be of class Dense when keep_las_layer is True"

                assert atleast_one, "No layer have been replaced in test."

    def test_fit_transform(self) -> None:
        model_transformer = LayerReplacerPalm(sparse_factorizer=self.palminizer, keep_last_layer=True, only_mask=False, dct_name_compression=None)
        new_model = model_transformer.fit_transform(deepcopy(self.base_model))

        model_transformer_already_fit = LayerReplacerPalm(keep_last_layer=True, only_mask=False, dct_name_compression=self.dct_sparsely_factorized_layers)
        new_model_2 = model_transformer_already_fit.transform(deepcopy(self.base_model))

        for idx_layer, layer in enumerate(new_model.layers):
            layer2 = new_model_2.layers[idx_layer]

            w_layer_1 = layer.get_weights()
            w_layer_2 = layer2.get_weights()

            assert len(w_layer_1) == len(w_layer_2), f"weights of {layer.name} are of different size"

            for index_w, w1 in enumerate(w_layer_1):
                w2 = w_layer_2[index_w]
                assert w1.shape == w2.shape, f"shape of weights must be the same in {layer.name}"
                # try:
                assert np.allclose(w1, w2), f"weights are different in {layer.name}"
                # except:
                #     print(w1, w2)

    def test_save(self):
        palminizer = Palminizer(sparsity_fac=2,
                                nb_factor=2,
                                nb_iter=200,
                                delta_threshold_palm=1e-6,
                                hierarchical=False,
                                fast_unstable_proj=False)

        with tempfile.TemporaryDirectory() as tmpdirname:
            path_to_checkpoint = pathlib.Path(tmpdirname) / "checkpoint.pickle"
            model_transformer = LayerReplacerPalm(sparse_factorizer=palminizer, keep_last_layer=True, only_mask=False, dct_name_compression=None, path_checkpoint_file=path_to_checkpoint)

            signal.signal(signal.SIGALRM, timeout_signal_handler)
            signal.alarm(2) # will interrupt execution
            try:
                model_transformer.fit(deepcopy(self.base_model))
                raise AssertionError("Should have been interrupted")
            except TimeoutError as to_err:
                print("TIMEOUT")
                new_model_transformer = LayerReplacerPalm(sparse_factorizer=palminizer, keep_last_layer=True, only_mask=False, dct_name_compression=None, path_checkpoint_file=path_to_checkpoint)
                assert len(new_model_transformer.dct_name_compression) == 0, "new model transformer should have a length zero dict transformation"
                new_model_transformer.load_dct_name_compression()
                length_just_after = len(new_model_transformer.dct_name_compression)
                assert length_just_after > 0, "new model transformer should have a length > 0 dict transformation after reloading"
                new_model_transformer.fit(deepcopy(self.base_model))
                length_after_refit = len(new_model_transformer.dct_name_compression)
                print(length_just_after, length_after_refit)
                new_model_transformer.transform(deepcopy(self.base_model))

                with self.assertRaises(KeyError):
                    model_transformer.transform(deepcopy(self.base_model))


    def test_faustizer(self):
        palminizer = Faustizer(sparsity_fac=2,
                                nb_factor=None,
                                nb_iter=2,
                                tol=1e-6,
                                hierarchical=True)

        model_transformer = LayerReplacerPalm(sparse_factorizer=palminizer, keep_last_layer=True, only_mask=False, dct_name_compression=None)
        model_transformer.fit(deepcopy(self.base_model))

if __name__ == '__main__':
    unittest.main()
