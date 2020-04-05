from palmnet.core.layer_replacer_sparse_facto import LayerReplacerSparseFacto

class LayerReplacerPalm(LayerReplacerSparseFacto):

    @staticmethod
    def _get_factors_from_op_sparsefacto(op_sparse_facto):
        factors = [fac.toarray() for fac in op_sparse_facto.get_list_of_factors()]
        return factors
