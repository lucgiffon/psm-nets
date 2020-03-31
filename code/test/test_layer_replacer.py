import unittest
from copy import deepcopy

from palmnet.core.layer_replacer_palm import LayerReplacerPalm
from palmnet.core.palminize import Palminizer, Palminizable
from palmnet.data import Cifar100, Mnist
from pprint import pprint
import numpy as np
from keras.layers import Dense
from palmnet.utils import get_idx_last_dense_layer


class TestLayerReplacerPalm(unittest.TestCase):

    def setUp(self) -> None:
        self.base_model = Mnist.load_model("mnist_lenet")
        palminizer = Palminizer(sparsity_fac=2,
                                nb_factor=2,
                                nb_iter=2,
                                delta_threshold_palm=1e-6,
                                hierarchical=False,
                                fast_unstable_proj=True)

        self.palminizable = Palminizable(self.base_model, palminizer)
        self.palminizable.palminize()
        pprint(self.palminizable.sparsely_factorized_layers)
        self.dct_sparsely_factorized_layers = self.palminizable.sparsely_factorized_layers

    def test_keep_last(self) -> None:
        for keep_last_layer in [True, False]:
            for only_mask in [True, False]:
                dct_name_facto = self.dct_sparsely_factorized_layers

                model_transformer = LayerReplacerPalm(keep_last_layer, only_mask, dct_name_facto)
                new_model = model_transformer.fit_transform(self.base_model)

                idx_last_dense = get_idx_last_dense_layer(new_model)
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



if __name__ == '__main__':
    unittest.main()
