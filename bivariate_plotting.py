import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()


def np_bivariate_normal_pdf(domain, mean, variance=0.25):
    X = np.arange(-domain + mean, domain + mean, variance)
    Y = np.arange(-domain + mean, domain + mean, variance)
    X, Y = np.meshgrid(X, Y)
    R = np.sqrt(X ** 2 + Y ** 2)
    Z = ((1. / np.sqrt(2 * np.pi)) * np.exp(-.5 * R ** 2))*2
    print(Z)
    print(np.shape(Z))
    return X, Y, Z


def plt_plot_bivariate_normal_pdf(x, y, z):
    # fig = plt.figure(figsize=(12, 6))
    ax = fig.gca(projection='3d')
    ax.plot_surface(x, y, z,
                    cmap=cm.coolwarm,
                    linewidth=0,
                    antialiased=True)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    # plt.show()


# plt_plot_bivariate_normal_pdf(*np_bivariate_normal_pdf(6, 4))
plt_plot_bivariate_normal_pdf(*np_bivariate_normal_pdf(8, 0, 1))
plt.show()