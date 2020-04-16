from keras.layers import Conv2D

from palmnet.layers import Conv2DCustom
from palmnet.layers.sparse_facto_conv2D_masked import SparseFactorisationConv2D
from palmnet.utils import cast_sparsity_pattern
from keras import initializers, regularizers, constraints, backend as K


class TuckerSparseFactoLayerConv(Conv2DCustom):
    def __init__(self, lst_sparsity_patterns_by_tucker_part, in_rank, out_rank,
                 scaler_initializer='glorot_uniform',
                 scaler_regularizer=None,
                 scaler_constraint=None,
                 **kwargs):
        super().__init__(**kwargs)

        # one element for each tucker part
        assert len(lst_sparsity_patterns_by_tucker_part) == 3

        self.lst_sparsity_patterns_by_tucker_part = list()
        self.lst_nb_factor = list()
        for sparsity_patterns in lst_sparsity_patterns_by_tucker_part:
            sparsity_pattern_tmp = [cast_sparsity_pattern(s) for s in sparsity_patterns]
            self.lst_sparsity_patterns_by_tucker_part.append(sparsity_pattern_tmp)
            self.lst_nb_factor.append(len(sparsity_pattern_tmp))

            assert [sparsity_pattern_tmp[i].shape[1] == sparsity_pattern_tmp[i + 1].shape[0] for i in range(len(sparsity_pattern_tmp) - 1)]

        self.in_rank = in_rank
        self.out_rank = out_rank

        assert self.lst_sparsity_patterns_by_tucker_part[0][-1].shape[1] == self.in_rank, "first sparsity pattern last dim should be equal to the in_rank dimension of tucker core in {}".format(__class__.__name__)

        assert self.lst_sparsity_patterns_by_tucker_part[1][0].shape[0] == self.in_rank * self.kernel_size[0] * self.kernel_size[1], "second sparsity pattern first dim should be equal to the in_rank dimension of tucker core in {}".format(__class__.__name__)
        assert self.lst_sparsity_patterns_by_tucker_part[1][-1].shape[1] == self.out_rank, "second sparsity pattern last dim should be equal to the out_rank dimension of tucker core in {}".format(__class__.__name__)

        assert self.lst_sparsity_patterns_by_tucker_part[-1][0].shape[0] == self.out_rank, "last sparsity pattern first dim should be equal to the out_rank dimension of tucker core in {}".format(__class__.__name__)
        assert self.lst_sparsity_patterns_by_tucker_part[-1][-1].shape[1] == self.filters, "last sparsity pattern last dim should be equal to the number of filters in {}".format(__class__.__name__)

        self.scaler_initializer = initializers.get(scaler_initializer)
        self.scaler_regularizer = regularizers.get(scaler_regularizer)
        self.scaler_constraint = constraints.get(scaler_constraint)


    def build(self, input_shape):

        self.in_factor = SparseFactorisationConv2D(self.lst_sparsity_patterns_by_tucker_part[0], filters=self.in_rank, kernel_size=(1,1),
                                                   scaler_initializer=self.scaler_initializer, scaler_regularizer=self.scaler_regularizer, scaler_constraint=self.scaler_constraint,
                                                   use_bias=False, padding='same', kernel_initializer=self.kernel_initializer, kernel_regularizer=self.kernel_regularizer)
        self.in_factor.build(input_shape)

        # strides and padding only for the core layer
        core_input_shape = self.in_factor.compute_output_shape(input_shape)
        self.core = SparseFactorisationConv2D(self.lst_sparsity_patterns_by_tucker_part[1], filters=self.out_rank, kernel_size=self.kernel_size,
                                              scaler_initializer=self.scaler_initializer, scaler_regularizer=self.scaler_regularizer, scaler_constraint=self.scaler_constraint,
                                              use_bias=False, padding=self.padding, strides=self.strides, kernel_initializer=self.kernel_initializer, kernel_regularizer=self.kernel_regularizer)
        self.core.build(core_input_shape)
        core_output_shape = self.core.compute_output_shape(core_input_shape)

        # bias and activation only on the last layer
        self.out_factor = SparseFactorisationConv2D(self.lst_sparsity_patterns_by_tucker_part[2], filters=self.filters, kernel_size=(1,1),
                                              scaler_initializer=self.scaler_initializer, scaler_regularizer=self.scaler_regularizer, scaler_constraint=self.scaler_constraint,
                                              use_bias=self.use_bias, padding='same', kernel_initializer=self.kernel_initializer, kernel_regularizer=self.kernel_regularizer)
        self.out_factor.build(core_output_shape)

        super().build(input_shape)

    def get_config(self):
        base_config = super().get_config()
        base_config.update({
            "in_rank": self.in_rank,
            "out_rank": self.out_rank
        })
        base_config["scaler_initializer"] = self.scaler_initializer
        base_config["scaler_regularizer"] = self.scaler_regularizer
        base_config["scaler_constraint"] = self.scaler_constraint
        base_config["lst_sparsity_patterns_by_tucker_part"] = self.lst_sparsity_patterns_by_tucker_part
        return base_config

    def convolution(self, X):
        return self.out_factor(self.core(self.in_factor(X)))

    def compute_output_shape(self, input_shape):
        core_input_shape = self.in_factor.compute_output_shape(input_shape)
        core_output_shape = self.core.compute_output_shape(core_input_shape)
        return self.out_factor.compute_output_shape(core_output_shape)