import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
from matplotlib import cm
# from sklearn.datasets.samples_generator import make_blobs
import sklearn.datasets._samples_generator

# Getting the number of peaks
try:
    n = int(input("Enter the Number of Peaks in your Graph: "))
except Exception as e:
    print("Enter the Correct Value: ", e)

# Getting the height for each peak
l = []
for i in range(n):
    height = float(input(f"Enter the height for peak {i}: "))
    l.append(height)
n_components = n

X, truth = sklearn.datasets._samples_generator.make_blobs(n_samples=700, centers=n_components,
                                                          cluster_std=l,
                                                          random_state=500)
plt.scatter(X[:, 0], X[:, 1], s=50, c=truth)
plt.title(f"Example of a mixture of {n_components} distributions")
plt.xlabel("x")
plt.ylabel("y");

# Extract x and y
x = X[:, 0]
y = X[:, 1]
# Define the borders
deltaX = (max(x) - min(x)) / 10
deltaY = (max(y) - min(y)) / 10
xmin = min(x) - deltaX
xmax = max(x) + deltaX
ymin = min(y) - deltaY
ymax = max(y) + deltaY
print(xmin, xmax, ymin, ymax)
# Create meshgrid
xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]

positions = np.vstack([xx.ravel(), yy.ravel()])
values = np.vstack([x, y])
kernel = st.gaussian_kde(values)
f = np.reshape(kernel(positions).T, xx.shape)

fig = plt.figure(figsize=(13, 7))
ax = plt.axes(projection='3d')
surf = ax.plot_surface(xx, yy, f, rstride=1, cstride=1, cmap=cm.coolwarm, edgecolor='none')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('PDF')
ax.set_title('Surface plot of Gaussian 2D KDE')
fig.colorbar(surf, shrink=0.5, aspect=5)  # add color bar indicating the PDF
ax.view_init(60, 35)
plt.show()
