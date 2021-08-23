# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.stats import multivariate_normal
# from mpl_toolkits.mplot3d import Axes3D
#
# # define normalized 2D gaussian
# def gaus2d(x=0, y=0, mx=0, my=0, sx=1, sy=1):
#     return 1. / (2. * np.pi * sx * sy) * np.exp(-((x - mx)**2. / (2. * sx**2.) + (y - my)**2. / (2. * sy**2.)))
#
# def main():
#     # Parameters to set
#     mu_x = 0
#     variance_x = 3
#
#     mu_y = 0
#     variance_y = 15
#
#     # Create grid and multivariate normal
#     x = np.linspace(-10, 10, 500)
#     y = np.linspace(-10, 10, 500)
#     X, Y = np.meshgrid(x, y)
#     pos = np.empty(X.shape + (2,))
#     pos[:, :, 0] = X
#     pos[:, :, 1] = Y
#     rv = multivariate_normal([mu_x, mu_y], [[variance_x, 0], [0, variance_y]])
#
#     # Make a 3D plot
#     fig = plt.figure()
#     ax = fig.gca(projection='3d')
#     ax.plot_surface(X, Y, rv.pdf(pos), cmap='viridis', linewidth=0)
#     ax.set_xlabel('X axis')
#     ax.set_ylabel('Y axis')
#     ax.set_zlabel('Z axis')
#     plt.show()
#
#
# if __name__ == '__main__':
#     main()


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import numpy as np
from scipy.stats import multivariate_normal

x, y = np.mgrid[-1.0:1.0:15j, -1.0:1.0:15j]
# Need an (N, 2) array of (x, y) pairs.
xy = np.column_stack([x.flat, y.flat])
mu = np.array([0.0, 0.0])
sigma = np.array([.5, .5])
covariance = np.diag(sigma**3)
z = multivariate_normal.pdf(xy, mean=mu, cov=covariance)
# Reshape back to a (30, 30) grid.
z = z.reshape(x.shape)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x,y,z)
ax.plot_wireframe(x,y,z)
plt.show()