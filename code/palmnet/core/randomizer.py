import logging

from qkmeans.palm.projection_operators import prox_splincol

from palmnet.core.sparse_factorizer import SparseFactorizer
from qkmeans.core.utils import build_constraint_set_smart
from qkmeans.palm.palm_fast import hierarchical_palm4msa, palm4msa

import numpy as np


class Randomizer(SparseFactorizer):
    def __init__(self, fast_unstable_proj=True, *args, **kwargs):
        self.fast_unstable_proj = fast_unstable_proj
        super().__init__(*args, **kwargs)

    @staticmethod
    def get_factors_from_op_sparsefacto(op_sparse_facto):
        factors = [fac for fac in op_sparse_facto]
        return factors

    def apply_factorization(self, matrix):
        """
        Generate a random sparse factorization of matrix.

        :param matrix: The matrix to random factorize.
        :return:
        """
        left_dim, right_dim = matrix.shape
        A = min(left_dim, right_dim)
        B = max(left_dim, right_dim)

        if self.nb_factor is None:
            nb_factors = int(np.log2(B))
        else:
            nb_factors = self.nb_factor
        # self.sparsity_fac = sparsity_fac
        # self.nb_factor = nb_factor
        random_facto = lambda x, y: prox_splincol(np.random.rand(x, y), max(x,y)*self.sparsity_fac, fast_unstable=True)

        lst_factors = [random_facto(left_dim, A), *(random_facto(A, A) for _ in range(nb_factors-2)), random_facto(A, right_dim)]

        final_X = np.linalg.multi_dot(lst_factors)

        return 1., lst_factors, final_X


