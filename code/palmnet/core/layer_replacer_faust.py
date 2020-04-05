import numpy as np
from palmnet.core.layer_replacer_sparse_facto import LayerReplacerSparseFacto

class LayerReplacerFaust(LayerReplacerSparseFacto):

    @staticmethod
    def _get_factors_from_op_sparsefacto(op_sparse_facto):
        faust = op_sparse_facto
        factors = [np.array(faust.factors(i).todense()) if not isinstance(faust.factors(i), np.ndarray) else faust.factors(i) for i in range(len(faust))]
        return factors
