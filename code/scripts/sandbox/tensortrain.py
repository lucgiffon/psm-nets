from keras.layers import Conv2D, Activation, BatchNormalization, Dropout, MaxPooling2D, Flatten, Dense, GlobalAveragePooling2D
from keras import regularizers
from keras.models import Sequential

from palmnet.layers.tt_layer_conv import TTLayerConv
from palmnet.layers.tt_layer_dense import TTLayerDense


def _init_(self, train=True):
    self.num_classes = 10
    self.weight_decay = 0.0005
    self.x_shape = [32, 32, 3]
    self.tt_rank = [1, 4, 4, 4, 1]
    self.tt_rank_conv = [12, 12, 12, 12, 1]
    self.trained_model = self.train()
    # if train:
    #    self.model = self.train(self.model)
    # else:
    #    self.model.load_weights('cifar10vgg.h5')



if __name__ == "__main__":
    build_model()