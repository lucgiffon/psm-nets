import numpy as np
from pyfaust.fact import palm4msa
from pyfaust.proj import splincol, ConstraintList, ParamsPalm4MSA, StoppingCriterion
from qkmeans.utils import logger
import logging
logger.setLevel(logging.ERROR)
little_dims = (64, 27)
big_dims = (576, 64)

N_fac = 2
sparsity = 2

repet = 5

errors = np.empty((4, repet))

for i in range(repet):
    U, S, V = np.linalg.svd(np.random.rand(64, 64))
    target3 = U[:27, :]  # 27 x 64 colones orthogonales
    target4 = np.random.rand(27, 64)

    for j, target in enumerate([target3, target3.T, target4, target4.T]):
        target = target.astype(float)
        left_dim, right_dim = target.shape

        # parameters faust
        stop = StoppingCriterion(tol=1e-10, maxiter=300)
        lst_constraints = [splincol((left_dim, right_dim), sparsity).constraint] + [splincol((right_dim, right_dim), sparsity).constraint for _ in range(N_fac - 1)]
        cons = ConstraintList(*lst_constraints)
        param = ParamsPalm4MSA(cons, stop, is_update_way_R2L=True)
        param.init_facts = [np.eye(left_dim, left_dim) for _ in range(N_fac-1)] + [np.zeros((left_dim, right_dim))]

        # call faust
        faust, final_lambda = palm4msa(target, param, ret_lambda=True)

        # approximation error
        final_X = np.array(faust.todense())
        error = np.linalg.norm(final_X - target) / np.linalg.norm(target)
        errors[j][i] = error
        print(f"Error Faust: {error}")


print("col ortho", "lin ortho", "rand", "rand.T")
print(np.mean(errors, axis=1))
print(errors)