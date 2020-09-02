from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LinearRegression

if __name__ == "__main__":
    x, y = datasets.make_s_curve(250, noise=0.1, )
    x = x[:, 0]

    tmp = x
    x = y
    y = tmp

    clf = LinearRegression().fit(x.reshape(-1, 1), y.reshape(-1, 1))
    y_pred = clf.predict(x.reshape(-1, 1))
    # plt.subplot(121)
    score = clf.score(x.reshape(-1, 1), y.reshape(-1, 1))

    plt.plot(x, y, 's', color="red", marker="x", markersize=4)
    plt.plot(x, y_pred, color="k")
    plt.xlabel("$x$")
    plt.ylabel("$y$")
    plt.title(f"R={score:.2f}")
    plt.savefig("kernel_poly_1.png")


    # plt.subplot(122)
    plt.cla()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    xx = x ** 2
    yy = y ** 2
    xxyy = np.sqrt(2) * x * y

    ax.plot(xx, yy, xxyy,  's', color="red", marker="x", markersize=4)
    clf = LinearRegression().fit(np.hstack([xx.reshape(-1, 1), yy.reshape(-1, 1)]), xxyy.reshape(-1, 1))
    score = clf.score(np.hstack([xx.reshape(-1, 1), yy.reshape(-1, 1)]), xxyy.reshape(-1, 1))
    z = lambda x, y: (clf.intercept_[0] + clf.coef_[0][0] * x + clf.coef_[0][1] * y)
    # tmp = np.linspace(min(xx), 5, 30)
    x, y = np.meshgrid(np.linspace(min(xx), max(xx), 30), np.linspace(min(yy), max(yy), 30))
    xxyy_pred = z(x, y)

    ax.plot_surface(x, y, xxyy_pred, alpha=0.7, color="black")
    ax.set_xlabel("$x^2$")
    ax.set_xticklabels([])
    ax.set_ylabel("$y^2$")
    ax.set_yticklabels([])
    ax.set_zlabel('$\sqrt{2}xy$')
    ax.set_zticklabels([])
    ax.set_title(f"R={score:.2f}")
    # plt.show()
    plt.savefig("kernel_poly_2.png")
