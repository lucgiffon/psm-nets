from keras import Sequential
from keras.datasets import cifar10
from sklearn.datasets import load_digits

from pyfaust.factparams import StoppingCriterion, ParamsHierarchical
from scipy.linalg import hadamard
from pyfaust.proj import const
import numpy as np
from pyfaust.fact import palm4msa, hierarchical
from pyfaust.factparams import ParamsPalm4MSA, ConstraintList, StoppingCriterion, ConstraintInt, ParamsHierarchicalRectMat
from pyfaust.proj import splincol
# (x_train, y_train), (x_test, y_test) = cifar10.load_data()
from qkmeans.core.qmeans_fast import init_lst_factors
from qkmeans.core.utils import build_constraint_set_smart
from qkmeans.data_structures import SparseFactors
from qkmeans.palm.palm_fast import hierarchical_palm4msa
import time

# dim = 32
sparsity = 2
digits = load_digits()
size_train = 1000
size_test = 100
digits = digits.data
first_mat = digits[:size_train]
# first_mat = np.random.randn(n_elm, dim)
test_data = digits[size_train:size_train+size_test]
dim = digits.shape[1]
n_fac = int(np.log2(dim))
mat = np.random.randn(dim, dim)
# mat = hadamard(dim)


target = first_mat @ mat
target_norm = np.linalg.norm(target)

# lst_constraints = [const(first_mat).constraint] + [splincol((dim, dim), sparsity).constraint for _ in range(n_fac)]
# cons = ConstraintList(*lst_constraints)
# stop = StoppingCriterion(tol=1e-5, maxiter=100)
# param = ParamsPalm4MSA(cons, stop)
# faust, lambda_ = palm4msa(target, param, ret_lambda=True)
# faust = np.array(faust.todense())
# diff = np.linalg.norm((faust - target) / target_norm)
#
# S_constraints = [const(first_mat).constraint] + [splincol((dim, dim), sparsity).constraint for _ in range(n_fac-1)]
# R_constraints = [splincol((dim, dim), int(dim/(sparsity**i))).constraint for i in range(n_fac)]
# loc_stop = StoppingCriterion(num_its=100)
# glob_stop = StoppingCriterion(num_its=100)
# param = ParamsHierarchical(S_constraints, R_constraints, loc_stop, glob_stop)
# faust2, lambda2_ = hierarchical(target, param, ret_lambda=True)
# faust2 = np.array(faust2.todense())
# diff2 = np.linalg.norm((faust2 - target) / target_norm)

lst_constraint_sets, lst_constraint_sets_desc = build_constraint_set_smart(left_dim=dim,
                                                                           right_dim=dim,
                                                                           nb_factors=n_fac + 1,
                                                                           sparsity_factor=sparsity,
                                                                           residual_on_right=True,
                                                                           fast_unstable_proj=True,
                                                                           constant_first=True,
                                                                           hierarchical=True)
lst_factors = init_lst_factors(dim, dim, n_fac)
lst_factors = [first_mat] + lst_factors

_lambda_act, op_factors_act, recons_act, _, _ = \
    hierarchical_palm4msa(
        arr_X_target=target,
        lst_S_init=lst_factors,
        lst_dct_projection_function=lst_constraint_sets,
        f_lambda_init=1.,
        nb_iter=50,
        update_right_to_left=True,
        residual_on_right=True,
        track_objective_palm=False,
        delta_objective_error_threshold_palm=1e-6,
        return_objective_function=False)

lst_factors_final = op_factors_act.get_list_of_factors(copy=True)[1:]
op_factors_act_final = SparseFactors(lst_factors_final)
recons_act_final = op_factors_act_final.compute_product() * _lambda_act

diff3 = np.linalg.norm((recons_act - target) / target_norm)


lst_constraint_sets, lst_constraint_sets_desc = build_constraint_set_smart(left_dim=dim,
                                                                           right_dim=dim,
                                                                           nb_factors=n_fac,
                                                                           sparsity_factor=sparsity,
                                                                           residual_on_right=True,
                                                                           fast_unstable_proj=True,
                                                                           constant_first=False,
                                                                           hierarchical=True)
lst_factors = init_lst_factors(dim, dim, n_fac)

_lambda_no_act, op_factors_vanilla, recons_vanilla, _, _ = \
    hierarchical_palm4msa(
        arr_X_target=mat,
        lst_S_init=lst_factors,
        lst_dct_projection_function=lst_constraint_sets,
        f_lambda_init=1.,
        nb_iter=50,
        update_right_to_left=True,
        residual_on_right=True,
        track_objective_palm=False,
        delta_objective_error_threshold_palm=1e-6,
        return_objective_function=False)


## evaluation
target_act_train = target
target_act_train_norm = np.linalg.norm(target_act_train)
target_act_test = test_data @ mat
target_act_test_norm = np.linalg.norm(target_act_test)

mat_norm = np.linalg.norm(mat)

vanilla_approx_train = first_mat @ recons_vanilla
vanilla_approx_test = test_data @ recons_vanilla
error_vanilla_train = np.linalg.norm(vanilla_approx_train - target_act_train) / target_act_train_norm
error_vanilla_test = np.linalg.norm(vanilla_approx_test - target_act_test) / target_act_test_norm

act_approx_train = first_mat @ recons_act_final
act_approx_test = test_data @ recons_act_final
error_act_train = np.linalg.norm(act_approx_train - target_act_train) / target_act_train_norm
error_act_test = np.linalg.norm(act_approx_test - target_act_test) / target_act_test_norm

print("Train")
print(f"Error vanilla: {error_vanilla_train}")
print(f"Error act: {error_act_train}")
print("Test")
print(f"Error vanilla: {error_vanilla_test}")
print(f"Error act: {error_act_test}")

diff_vanilla = np.linalg.norm(recons_vanilla - mat) / mat_norm
diff_act = np.linalg.norm(recons_act_final - mat) / mat_norm

print("Matrix approximation")
print(f"Diff vanilla: {diff_vanilla}")
print(f"Diff act: {diff_act}")



# else:
# _lambda_tmp, op_factors, U_centroids, objective_palm, nb_iter_palm = \
#     palm4msa(
#         arr_X_target=np.eye(K_nb_cluster) @ X_centroids_hat,
#         lst_S_init=lst_factors,
#         nb_factors=len(lst_factors),
#         lst_projection_functions=lst_proj_op_by_fac_step[-1][
#             "finetune"],
#         f_lambda_init=init_lambda * eye_norm,
#         nb_iter=nb_iter_palm,
#         update_right_to_left=True,
#         track_objective=track_objective_palm,
#         delta_objective_error_threshold=delta_objective_error_threshold_inner_palm)

print(diff, diff2, diff3)