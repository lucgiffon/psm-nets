from pyfaust.fact import palm4msa, hierarchical
from pyfaust.factparams import ParamsPalm4MSA, ConstraintList, StoppingCriterion, ConstraintInt, ParamsHierarchicalRectMat
from pyfaust.proj import splincol
import matplotlib.pyplot as plt
import numpy as np
#
# if __name__ == "__main__":
#     M = 100
#     N = 200
#     mat = np.random.rand(M, N)
#
#     N_fac = 5
#     sparsity_col = sparsity_row = sparsity = 10
#
#     stop = StoppingCriterion(tol=1e-5, maxiter=100)
#
#     lst_constraints = [splincol((M, N), sparsity).constraint, splincol((N, N), sparsity).constraint]
#     cons = ConstraintList(*lst_constraints)
#     param = ParamsPalm4MSA(cons, stop)
#
#     faust, lambda_ = palm4msa(mat, param, ret_lambda=True)
#
#     faust.imshow()
#     plt.show()
#
#     lst_factors = [faust.factors(i).todense() for i in range(len(faust))]
#
#     print([(f.shape, type(f)) for f in lst_factors])
#

def benchmark():
    nb_test = 50
    sparsity = 2
    N_fac = 3
    stop = StoppingCriterion(tol=1e-5, maxiter=100)

    for sparsity in range(2, 10, 2):
        for N_fac in range(2, 6, 2):
            lst_results_transpose = []
            lst_results_no_transpose = []

            for i in range(nb_test):
                M = 100
                N = 200
                mat = np.random.rand(M, N)
                for transpose in [True, False]:
                    if transpose:
                        target = mat.T
                        lst_constraints = [splincol((N, M), sparsity).constraint] + [splincol((M, M), sparsity).constraint for _ in range(N_fac - 1)]
                    else:
                        target = mat
                        lst_constraints = [splincol((M, M), sparsity).constraint for _ in range(N_fac - 1)] + [splincol((M, N), sparsity).constraint]

                    cons = ConstraintList(*lst_constraints)
                    param = ParamsPalm4MSA(cons, stop)

                    faust, lambda_ = palm4msa(target, param, ret_lambda=True)

                    if transpose:
                        result = faust.todense().T
                        lst_results_transpose.append(np.linalg.norm(result - mat) / np.linalg.norm(mat))
                    else:
                        result = faust.todense()
                        lst_results_no_transpose.append(np.linalg.norm(result - mat) / np.linalg.norm(mat))

            mean_result_transpose = np.mean(lst_results_transpose)
            mean_result_no_transpose = np.mean(lst_results_no_transpose)
            std_result_transpose = np.std(lst_results_transpose)
            std_result_no_transpose = np.std(lst_results_no_transpose)
            print("sparsity", sparsity, "N_fac", N_fac, "error transposed", mean_result_transpose, "std", std_result_transpose, "case L2R: big first", mat.shape)  #
            print("sparsity", sparsity, "N_fac", N_fac, "error no_transposed", mean_result_no_transpose, "std", std_result_no_transpose, "case L2R: big last", mat.shape)  # case L2R: big last


if __name__ == "__main__":
    sparsity = 2
    size = 32

    from pyfaust import wht
    FH = wht(size, normed=False)  # normed=False is to avoid column normalization
    H = FH.toarray()

    F, lambda_ = hierarchical(H, ['squaremat', int(np.log2(size)), sparsity, sparsity], ret_lambda=True)
    res = np.array(np.round(F.todense())).astype(int)
    print("h", H)
    print("res", res)
    print("lambda res", lambda_, (lambda_ * res).astype(int))

    F /= lambda_
    res = np.array(F.todense())
    print("h", H)
    print("res", res)
    print("lambda res", lambda_, np.round((lambda_ * res)))