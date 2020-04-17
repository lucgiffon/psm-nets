import logging
from math import ceil

from pyfaust.fact import hierarchical, palm4msa
from pyfaust.factparams import ConstraintList, StoppingCriterion, ParamsPalm4MSA, ConstraintInt, ParamsHierarchical
from pyfaust.proj import splincol

from palmnet.core.sparse_factorizer import SparseFactorizer

import numpy as np


class Faustizer(SparseFactorizer):
    def __init__(self, tol=1e-6, *args, **kwargs):
        self.tol = tol
        super().__init__(*args, **kwargs)

    @staticmethod
    def get_factors_from_op_sparsefacto(op_sparse_facto):
        faust = op_sparse_facto
        factors = [np.array(faust.factors(i).todense()) if not isinstance(faust.factors(i), np.ndarray) else faust.factors(i) for i in range(len(faust))]
        return factors

    @staticmethod
    def create_param_hierarchical_rect_control_stop_criter(m, n, j, sparsity, tol, max_iter):

        P = 1.4*m**2
        rho = 0.8

        S1_cons = ConstraintInt('spcol', m, n, sparsity)
        S_cons = [S1_cons]
        for i in range(j-2):
            S_cons += [ ConstraintInt('sp', m, m, sparsity*m) ]

        R_cons = []
        for i in range(j-1):
            R_cons += [ConstraintInt('sp', m, m, int(ceil(P*rho**i)))]

        stop_crit = StoppingCriterion(tol=tol, maxiter=max_iter)
        # stop_crit = StoppingCriterion(num_its=30)

        return ParamsHierarchical(S_cons, R_cons,
                                  stop_crit,
                                  stop_crit,
                                  is_update_way_R2L=True,
                                  is_fact_side_left=True)

    @staticmethod
    def build_constraints_faust(left_dim, right_dim, sparsity, N_fac, hierarchical, tol, nb_iter):
        """
        Only for left to right optimization + bigger dim on left

        :param left_dim:
        :param right_dim:
        :param sparsity:
        :param N_fac:
        :return:
        """
        if not hierarchical or N_fac==1: # if N_fac ==1 then hierarchical is equivalent to not hierarchical. (and hierarchical would crash)
            assert left_dim >= right_dim

            stop = StoppingCriterion(tol=tol, maxiter=nb_iter)

            lst_constraints = [splincol((left_dim, right_dim), sparsity).constraint] + [splincol((right_dim, right_dim), sparsity).constraint for _ in range(N_fac - 1)]
            cons = ConstraintList(*lst_constraints)
            param = ParamsPalm4MSA(cons, stop)
        else:
            assert right_dim >= left_dim
            param = Faustizer.create_param_hierarchical_rect_control_stop_criter(left_dim, right_dim, N_fac, sparsity, tol, nb_iter)

        return param

    def apply_factorization(self, matrix):
        """
        Apply Hierarchical-PALM4MSA algorithm to the input matrix and return the reconstructed approximation from
        the sparse factorisation.

        :param matrix: The matrix to apply PALM to.
        :param sparsity_fac: The sparsity factor for PALM.
        :return:
        """
        transposed = False

        not_H_bad_shape = (not self.hierarchical and matrix.shape[0] < matrix.shape[1]) # we want the bigger dimension to be on left because error is lower as R2L doesn't work
        H_bad_shape = (self.hierarchical and matrix.shape[0] > matrix.shape[1]) # R2L work in that case

        if not_H_bad_shape or H_bad_shape:
            matrix = matrix.T
            transposed = True

        matrix = matrix.astype(float)
        left_dim, right_dim = matrix.shape

        # dynamic choice of the number of factor
        if self.nb_factor is None:
            nb_factors = int(np.log2(max(left_dim, right_dim)))
        else:
            nb_factors = self.nb_factor

        if nb_factors == 1 and self.hierarchical:
            # because in that case, it is actually not the hierarchical palm that will be used
            # but the standard PALM instead because hierarchical is non-sense in that degenerate case
            # and would crash
            assert transposed is True
            matrix = matrix.T
            transposed = False
            left_dim, right_dim = matrix.shape

        constraints = self.build_constraints_faust(left_dim, right_dim, sparsity=self.sparsity_fac, N_fac=nb_factors,
                                                   hierarchical=self.hierarchical, tol=self.tol, nb_iter=self.nb_iter)

        if self.hierarchical and not nb_factors == 1:
            logging.info("Applying hierarchical faust palm4msa to matrix with shape {}".format(matrix.shape))
            # in the case nb_factos==1, parameters have been built for the standard palm4msa
            faust, final_lambda = hierarchical(matrix, constraints, ret_lambda=True)
            # faust, final_lambda = hierarchical(matrix, ["rectmat", nb_factors, self.sparsity_fac, self.sparsity_fac], ret_lambda=True)
        else:
            logging.info("Applying faust palm4msa to matrix with shape {}".format(matrix.shape))
            faust, final_lambda = palm4msa(matrix, constraints, ret_lambda=True)

        faust /= final_lambda

        final_X = np.array(faust.todense())

        if transposed:
            return final_lambda, faust.T, final_X.T
        else:
            return final_lambda, faust, final_X

