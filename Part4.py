from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import scipy.stats as st
import numpy as np
from matplotlib import cm


def plot_chart(peak_factor):
    X, Y = make_blobs(n_samples=[int(100 * peak_factor[0]), int(95 * peak_factor[1]), int(135 * peak_factor[2])],
                      centers=[(2, 4.5), (3.5, 4.5), (3.5, 2)], random_state=0,
                      cluster_std=[0.1, 0.25, 0.05])
    """ Converting To 3D Plot """
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
    """ Create Meshgrid """
    xx, yy = np.mgrid[xmin - 0.5:xmax + 0.5:100j, ymin - 0.5:ymax + 0.5:100j]
    positions = np.vstack([xx.ravel(), yy.ravel()])
    values = np.vstack([x, y])
    kernel = st.gaussian_kde(values)
    f = np.reshape(kernel(positions).T, xx.shape)
    fig = plt.figure(figsize=(13, 7))
    ax = plt.axes(projection='3d')
    surf = ax.plot_surface(xx, yy, f, rstride=3, cstride=3, cmap=cm.Purples, edgecolor='none', linewidth=5,
                           linestyle='dashed',
                           edgecolors='yellow')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('PDF')
    ax.set_title('Surface Plot')
    fig.colorbar(surf, shrink=0.5, aspect=5)  # add color bar indicating the PDF
    ax.view_init(25, 230)
    plt.show()


def main():
    peak_factor = [1, 1, 1]
    print(
        "You have been given the default Chart. To increase or decrease the heights of peaks"
        "\nEnter the value between 0 and 0.9 to "
        "decrease the height of the peak\nEnter the value between 1 and 2 to increase the height")
    for factor in range(len(peak_factor)):
        try:
            fact = input(f"Enter the Peak Factor for Peak {factor + 1}: ")
            if fact == "":
                continue
            fact = float(fact)
            if fact < 0:
                fact = 0
            elif fact > 2:
                fact = 2
            peak_factor[factor] = fact
        except Exception as e:
            print(e)
            pass
    plot_chart(peak_factor)


if __name__ == "__main__":
    main()
