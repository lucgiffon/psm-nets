import logging

from palmnet.core.sparse_factorizer import SparseFactorizer
from qkmeans.core.utils import build_constraint_set_smart
from qkmeans.palm.palm_fast import hierarchical_palm4msa, palm4msa

import numpy as np


class ActivationPalminizer(SparseFactorizer):
    def __init__(self, data, model, delta_threshold_palm=1e-6, fast_unstable_proj=False, *args, **kwargs):
        self.delta_threshold_palm = delta_threshold_palm
        self.fast_unstable_proj = fast_unstable_proj
        super().__init__(*args, **kwargs)

    @staticmethod
    def get_factors_from_op_sparsefacto(op_sparse_facto):
        factors = [fac.toarray() for fac in op_sparse_facto.get_list_of_factors()]
        return factors

    def apply_factorization(self, matrix):
        """
        Apply Hierarchical-PALM4MSA algorithm to the input matrix and return the reconstructed approximation from
        the sparse factorisation.

        :param matrix: The matrix to apply PALM to.
        :param sparsity_fac: The sparsity factor for PALM.
        :return:
        """
        logging.info("Applying palm function to matrix with shape {}".format(matrix.shape))
        transposed = False

        if matrix.shape[0] > matrix.shape[1]:
            # we want the bigger dimension to be on right due to the residual computation that should remain big
            matrix = matrix.T
            transposed = True

        left_dim, right_dim = matrix.shape
        A = min(left_dim, right_dim)
        B = max(left_dim, right_dim)
        assert A == left_dim and B == right_dim, "Dimensionality problem: left dim should be higher than right dim before palm"

        if self.nb_factor is None:
            nb_factors = int(np.log2(B))
        else:
            nb_factors = self.nb_factor

        lst_factors = [np.eye(A) for _ in range(nb_factors)]
        lst_factors[-1] = np.zeros((A, B))
        _lambda = 1.  # init the scaling factor at 1

        lst_proj_op_by_fac_step, lst_proj_op_by_fac_step_desc = build_constraint_set_smart(left_dim=left_dim,
                                                                                           right_dim=right_dim,
                                                                                           nb_factors=nb_factors,
                                                                                           sparsity_factor=self.sparsity_fac,
                                                                                           residual_on_right=True,
                                                                                           fast_unstable_proj=self.fast_unstable_proj,
                                                                                           constant_first=False,
                                                                                           hierarchical=self.hierarchical)

        if self.hierarchical:
            final_lambda, final_factors, final_X, _, _ = hierarchical_palm4msa(
                arr_X_target=matrix,
                lst_S_init=lst_factors,
                lst_dct_projection_function=lst_proj_op_by_fac_step,
                f_lambda_init=_lambda,
                nb_iter=self.nb_iter,
                update_right_to_left=True,
                residual_on_right=True,
                delta_objective_error_threshold_palm=self.delta_threshold_palm,)
        else:
            final_lambda, final_factors, final_X, _, _ = \
                palm4msa(arr_X_target=matrix,
                         lst_S_init=lst_factors,
                         nb_factors=len(lst_factors),
                         lst_projection_functions=lst_proj_op_by_fac_step,
                         f_lambda_init=1.,
                         nb_iter=self.nb_iter,
                         update_right_to_left=True,
                         delta_objective_error_threshold=self.delta_threshold_palm,
                         track_objective=False)
            final_X *= final_lambda  # added later because palm4msa actually doesn't return the final_X multiplied by lambda contrary to hierarchical

        if transposed:
            return final_lambda, final_factors.transpose(), final_X.T
        else:
            return final_lambda, final_factors, final_X


    def factorize_layer(self, layer_obj, apply_weights=True):
        pass