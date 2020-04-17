import unittest
from copy import deepcopy

from tensorly.decomposition import partial_tucker

from palmnet.core.layer_replacer_tucker import LayerReplacerTucker
from palmnet.data import Mnist
import numpy as np
from tensorly.tenalg.n_mode_product import multi_mode_dot


class TestLayerReplacerTucker(unittest.TestCase):

    def setUp(self) -> None:
        self.base_model = Mnist.load_model("cifar100_vgg19_2048x2048")

    def test_simple(self):
        model_transformer = LayerReplacerTucker(keep_last_layer=True)
        new_model = model_transformer.fit_transform(deepcopy(self.base_model))

        model_transformer = LayerReplacerTucker(rank_percentage_dense=0.5, keep_last_layer=True)
        new_model = model_transformer.fit_transform(deepcopy(self.base_model))

    def test_tucker_decomposition(self):
        import tensorly

        h, w, c, f = 3, 3, 64, 128
        c_prim, f_prim = 16, 32
        base_tensor = np.random.rand(h, w, c, f)

        lst_fac = []
        for k in [2, 3]:
            mod_k_unfold = tensorly.base.unfold(base_tensor, k)
            U, _, _ = np.linalg.svd(mod_k_unfold)
            lst_fac.append(U)

        # real_in_fac, real_out_fac = lst_fac[0][:, :c_prim], lst_fac[1][:, :f_prim]
        real_in_fac, real_out_fac = lst_fac[0], lst_fac[1]
        real_core = multi_mode_dot(base_tensor, [real_in_fac.T, real_out_fac.T], modes=(2,3))
        del base_tensor  # no need of it anymore

        real_core = real_core[:,:,:c_prim,:f_prim]
        real_in_fac = real_in_fac[:, :c_prim]
        real_out_fac = real_out_fac[:, :f_prim]

        base_tensor_low_rank = multi_mode_dot(real_core, [real_in_fac, real_out_fac], modes=(2,3))


        in_rank, out_rank = LayerReplacerTucker.get_rank_layer(base_tensor_low_rank)
        assert in_rank == c_prim and out_rank == f_prim, f"{in_rank}!={c_prim} or {out_rank} != {f_prim}"  # in_rank=16, out_rank=32 -> it works!

        decomposition = LayerReplacerTucker.get_tucker_decomposition(base_tensor_low_rank, in_rank, out_rank)
        # core_tilde, (in_fac_tilde, out_fac_tilde) = partial_tucker(base_tensor, modes=(2, 3), ranks=(in_rank, out_rank), init='svd')
        in_fac_tilde, core_tilde, out_fac_tilde = decomposition
        base_tensor_tilde = multi_mode_dot(core_tilde, [in_fac_tilde, out_fac_tilde], modes=(2,3))
        assert np.allclose(base_tensor_tilde, base_tensor_low_rank)
        print(np.linalg.norm(in_fac_tilde - real_in_fac) / np.linalg.norm(real_in_fac))
        # assert np.allclose(in_fac_tilde, real_in_fac)
        # assert np.allclose(core_tilde, core)
        # assert np.allclose(out_fac_tilde, out_fac)


    def test_stack_overflow(self):
        import tensorly
        import numpy as np

        h, w, c, f = 3, 3, 64, 128
        c_prim, f_prim = 16, 32
        base_tensor = np.random.rand(h, w, c, f)

        # compute tucker decomposition by hand using higher order svd describred here: https://www.alexejgossmann.com/tensor_decomposition_tucker/.
        lst_fac = []
        for k in [2, 3]:
            mod_k_unfold = tensorly.base.unfold(base_tensor, k)
            U, _, _ = np.linalg.svd(mod_k_unfold)
            lst_fac.append(U)

        real_in_fac, real_out_fac = lst_fac[0], lst_fac[1]
        real_core = multi_mode_dot(base_tensor, [real_in_fac.T, real_out_fac.T], modes=(2, 3))
        del base_tensor  # no need of it anymore

        # what i call the "low rank tucker decomposition"
        real_core = real_core[:, :, :c_prim, :f_prim]
        real_in_fac = real_in_fac[:, :c_prim]
        real_out_fac = real_out_fac[:, :f_prim]

        # low rank approximation
        base_tensor_low_rank = multi_mode_dot(real_core, [real_in_fac, real_out_fac], modes=(2, 3))
        in_rank, out_rank = c_prim, f_prim
        core_tilde, (in_fac_tilde, out_fac_tilde) = partial_tucker(base_tensor_low_rank, modes=(2, 3), ranks=(in_rank, out_rank), init='svd')
        base_tensor_tilde = multi_mode_dot(core_tilde, [in_fac_tilde, out_fac_tilde], modes=(2, 3))
        assert np.allclose(base_tensor_tilde, base_tensor_low_rank)  # this is OK

        assert np.allclose(in_fac_tilde, real_in_fac)  # this fails


if __name__ == '__main__':
    unittest.main()
