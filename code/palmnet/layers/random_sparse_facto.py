import numpy as np

from palmnet.layers.sparse_facto_conv2D_masked import SparseFactorisationConv2D
from palmnet.layers.sparse_facto_dense_masked import SparseFactorisationDense
from palmnet.utils import create_sparse_factorization_pattern


class RandomSparseFactorisationDense(SparseFactorisationDense):
    def __init__(self, units, sparsity_factor, nb_sparse_factors=None, permutation=True, **kwargs):

        self.nb_factor = nb_sparse_factors
        self.sparsity_factor = sparsity_factor
        self.permutation = permutation

        if 'sparsity_patterns' not in kwargs:
            super(RandomSparseFactorisationDense, self).__init__(units, None, **kwargs)
        else:
            super(RandomSparseFactorisationDense, self).__init__(units, **kwargs)

    def build(self, input_shape):

        if self.nb_factor is None:
            self.nb_factor = int(np.log(max(input_shape[-1], self.units)))
        self.sparsity_patterns = create_sparse_factorization_pattern((input_shape[-1], self.units), self.sparsity_factor, self.nb_factor, self.permutation)

        super(RandomSparseFactorisationDense, self).build(input_shape)

    def get_config(self):
        base_config = super().get_config()
        config = {
            'nb_sparse_factors': self.nb_factor,
            'sparsity_factor_lst': self.sparsity_factor,
        }
        config.update(base_config)
        return config

class RandomSparseFactorisationConv2D(SparseFactorisationConv2D):
    def __init__(self, sparsity_factor, nb_sparse_factors=None, permutation=True, **kwargs):
        self.nb_factor = nb_sparse_factors
        self.sparsity_factor = sparsity_factor
        self.permutation = permutation

        if 'sparsity_patterns' not in kwargs:
            super(SparseFactorisationConv2D, self).__init__(None, **kwargs)
        else:
            super(SparseFactorisationConv2D, self).__init__(**kwargs)

    def build(self, input_shape):
        dim1, dim2 = np.prod(self.kernel_size) * input_shape[-1], self.filters
        if self.nb_factor is None:
            self.nb_factor = int(np.log(max(dim1, dim2)))
        self.sparsity_patterns = create_sparse_factorization_pattern((dim1, dim2), self.sparsity_factor, self.nb_factor, self.permutation)

        super(SparseFactorisationConv2D, self).build(input_shape)

    def get_config(self):
        config = super().get_config()
        config['sparsity_factor_lst'] = self.sparsity_factor
        config['nb_sparse_factors'] = self.nb_factor
        return config
