import numpy as np
from pyfaust.fact import palm4msa
from pyfaust.proj import splincol, ConstraintList, ParamsPalm4MSA, StoppingCriterion
from qkmeans.palm.palm_fast import palm4msa as palm4msa_own
# little_dims = (27, 64)
from qkmeans.core.utils import build_constraint_set_smart
from qkmeans.utils import logger
import logging
logger.setLevel(logging.ERROR)
little_dims = (64, 27)
# big_dims = (64, 576)
big_dims = (576, 64)

# a = np.random.rand(64, 576)
# b = np.random.rand(27, 64)
N_fac = 2
sparsity = 2

repet = 1

errors = np.empty((4, repet))

a = np.random.rand(64, 64)
U, S, V = np.linalg.svd(a)
target3 = U[:27, :]
target4 = np.random.rand(27, 64)
for i in range(repet):
    # for j, target in enumerate([target1, target1.T, target2, target2.T]):
    for j, target in enumerate([target3, target3.T, target4, target4.T]):
        target = target.astype(float)
        left_dim, right_dim = target.shape
        stop = StoppingCriterion(tol=1e-10, maxiter=300)
        lst_constraints = [splincol((left_dim, right_dim), sparsity).constraint] + [splincol((right_dim, right_dim), sparsity).constraint for _ in range(N_fac - 1)]
        cons = ConstraintList(*lst_constraints)
        param = ParamsPalm4MSA(cons, stop, is_update_way_R2L=True)
        param.init_facts = [np.eye(left_dim, left_dim) for _ in range(N_fac-1)] + [np.zeros((left_dim, right_dim))]

        faust, final_lambda = palm4msa(target, param, ret_lambda=True)

        final_X = np.array(faust.todense())

        error = np.linalg.norm(final_X - target) / np.linalg.norm(target)
        errors[j][i] = error
        print(f"Error Faust: {error}")

        lst_proj_op_by_fac_step, lst_proj_op_by_fac_step_desc = build_constraint_set_smart(left_dim=left_dim,
                                                                                           right_dim=right_dim,
                                                                                           nb_factors=N_fac,
                                                                                           sparsity_factor=sparsity,
                                                                                           residual_on_right=True,
                                                                                           fast_unstable_proj=True,
                                                                                           constant_first=False,
                                                                                           hierarchical=False)
        lst_factors = [np.eye(left_dim) for _ in range(N_fac)]
        lst_factors[-1] = np.zeros((left_dim, right_dim))

        final_lambda, final_factors, final_X, _, _ = \
            palm4msa_own(arr_X_target=target,
                     lst_S_init=lst_factors,
                     nb_factors=len(lst_factors),
                     lst_projection_functions=lst_proj_op_by_fac_step,
                     f_lambda_init=1.,
                     nb_iter=300,
                     update_right_to_left=True,
                     delta_objective_error_threshold=1e-10,
                     track_objective=False)
        final_X *= final_lambda

        error = np.linalg.norm(final_X - target) / np.linalg.norm(target)
        errors[j][i] = error
        print(f"Error own: {error}")


# print(np.mean(errors, axis=1))
print(errors)