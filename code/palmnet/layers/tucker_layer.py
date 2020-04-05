from keras.layers import Conv2D

from palmnet.layers import Conv2DCustom


class TuckerLayerConv(Conv2DCustom):
    def __init__(self, in_rank, out_rank, **kwargs):

        self.in_rank = in_rank
        self.out_rank = out_rank
        super().__init__(**kwargs)

    def build(self, input_shape):

        self.in_factor = Conv2D(self.in_rank, (1, 1), kernel_initializer=self.kernel_initializer, padding='same', kernel_regularizer=self.kernel_regularizer, use_bias=False)  #
        self.in_factor.build(input_shape)

        # strides and padding only for the core layer
        core_input_shape = self.in_factor.compute_output_shape(input_shape)
        self.core = Conv2D(self.out_rank, kernel_size=self.kernel_size, kernel_initializer=self.kernel_initializer, padding=self.padding, kernel_regularizer=self.kernel_regularizer,
                           use_bias=False)  # core
        self.core.build(core_input_shape)
        core_output_shape = self.core.compute_output_shape(core_input_shape)

        # bias and activation only on the last layer
        self.out_factor = Conv2D(self.filters, (1, 1), use_bias=self.use_bias, kernel_initializer=self.kernel_initializer, padding='same', kernel_regularizer=self.kernel_regularizer)
        self.out_factor.build(core_output_shape)

        # self._trainable_weights = self.in_factor.trainable_weights + self.core.trainable_weights + self.out_factor.trainable_weights

        super().build(input_shape)

    def get_config(self):
        base_config = super().get_config()
        base_config.update({
            "in_rank": self.in_rank,
            "out_rank": self.out_rank
        })
        return base_config

    def convolution(self, X):
        return self.out_factor(self.core(self.in_factor(X)))

    def compute_output_shape(self, input_shape):
        core_input_shape = self.in_factor.compute_output_shape(input_shape)
        core_output_shape = self.core.compute_output_shape(core_input_shape)
        return self.out_factor.compute_output_shape(core_output_shape)