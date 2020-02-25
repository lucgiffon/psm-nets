from keras.applications.resnet import ResNet50
from keras.applications.resnet_v2 import ResNet50V2
from keras_contrib.applications.resnet import ResNet18
import numpy as np

sparsity_fac = 10
# func_nb_param = lambda dim1, dim2: np.log2(max(dim1, dim2)) * min(dim1, dim2) * sparsity_fac**2 + max(dim1, dim2)
func_nb_param = lambda dim1, dim2: 2 * min(dim1, dim2) * sparsity_fac**2 + max(dim1, dim2)

if __name__ == "__main__":
    res50 = ResNet50(include_top=False, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=100)
    res50v2 = ResNet50V2(include_top=False, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=100)
    res18 = ResNet18(None, 100)

    count_param_base = 0
    count_param_compression_expectation = 0
    for l in res18.layers:
        l_name = l.name
        print(l_name)
        l_weights = l.weights
        for i, w in enumerate(l.get_weights()):
            w_name = l_weights[i].name
            if "kernel" in w_name:
                dim1 = np.prod(w.shape[:-1])
                dim2 = w.shape[-1]
                str_reshape = "{}x{}".format(dim1, dim2)
                count_param = func_nb_param(dim1, dim2)
                if count_param < np.prod(w.shape):
                    count_param_compression_expectation += count_param
                else:
                    count_param_compression_expectation += np.prod(w.shape)
            else:
                str_reshape = ""
                count_param_compression_expectation += np.prod(w.shape)

            count_param_base += np.prod(w.shape)

            print(w_name, ":", w.shape, str_reshape)

    ratio = count_param_compression_expectation / count_param_base
    print(ratio, 1/ratio)