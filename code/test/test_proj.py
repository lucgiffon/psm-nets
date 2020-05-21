import unittest
import numpy as np
from qkmeans.palm.projection_operators import prox_splincol
import matplotlib.pyplot as plt


class TestProj(unittest.TestCase):
    def test_proj(self):
        n = 32
        sp = 2
        mat = np.random.rand(n, n)
        p_mat = prox_splincol(mat, n*sp, fast_unstable=True)
        count_nnz = np.sum(p_mat.astype(bool))
        assert n * sp <= count_nnz <= n*sp*2
        plt.imshow(p_mat)
        plt.show()
        print(count_nnz)


if __name__ == '__main__':
    unittest.main()
