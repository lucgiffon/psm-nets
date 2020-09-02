from sklearn.datasets import make_circles
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    x, y = make_circles(500, noise=0.1, factor=0.2)

    # plt.subplot(121)

    plt.plot(x[y == 0][:, 0], x[y == 0][:, 1], 's', color="blue", marker="x", markersize=10)
    plt.plot(x[y == 1][:, 0], x[y == 1][:, 1], 's', color="red", marker="o", markersize=4)
    plt.xlabel("$x$")
    plt.ylabel("$y$")

    plt.savefig("kernel_poly_1.png")


    # plt.subplot(122)
    plt.cla()

    xx = x[:, 0] ** 2
    yy = x[:, 1] ** 2
    xxyy = np.sqrt(2) * x[:, 0] * x[:, 1]

    plt.plot(xx[y==0], yy[y == 0], 's', color="blue", marker="x", markersize=10)
    plt.plot(xx[y==1], yy[y == 1], 's', color="red", marker="o", markersize=4)

    plt.xlabel("$x^2$")
    plt.ylabel("$y^2$")

    plt.savefig("kernel_poly_2.png")
