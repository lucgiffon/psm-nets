from keras.datasets import cifar10
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


(x_train, y_train), (x_test, y_test) = cifar10.load_data()

dsize = 2
images = x_train[:dsize]

sess = tf.Session()

k_lin, k_col = 20,20
imagettes = tf.image.extract_image_patches(images, ksizes=(1, k_lin,k_col, 1), strides=(1, 1, 1, 1), rates=[1, 1, 1, 1], padding="VALID").eval(session=sess)
imagettes = tf.reshape(imagettes, shape=(-1, imagettes.shape[1]*imagettes.shape[2], k_lin*k_col*3)).eval(session=sess)
for i, im in enumerate(images):
    plt.imshow(im)
    plt.show()
    for k in range(imagettes.shape[2]):
        im2 = np.reshape(imagettes[i, k], (k_lin,k_col, 3))
        plt.imshow(im2)
        plt.show()

sess.close()