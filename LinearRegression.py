import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

#this program uses direct matrices calculation
#house price(y), size(x1), number of rooms(x2)
def draw_graph(X1, X2, y, theta):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    ax.scatter(X1, X2, y, c='b', marker='o', label='Data points')
    
    # Create a meshgrid for the plane
    x1_surf, x2_surf = np.meshgrid(np.linspace(X1.min(), X1.max(), 10),
                                   np.linspace(X2.min(), X2.max(), 10))
    X_surf = np.c_[np.ones(x1_surf.ravel().shape), x1_surf.ravel(), x2_surf.ravel()]
    y_surf = X_surf.dot(theta).reshape(x1_surf.shape)
    
    ax.plot_surface(x1_surf, x2_surf, y_surf, alpha=0.5, color='r', label='Regression plane')
    
    ax.set_xlabel('X1 (Size)')
    ax.set_ylabel('X2 (Number of Rooms)')
    ax.set_zlabel('y (Price)')
    ax.set_title('3D Linear Regression')
    
    ax.view_init(elev=20, azim=120)  # Customize elevation and azimuth angles for rotation
    plt.legend()
    plt.show()


if __name__ == "__main__":
    # Generate some data
    np.random.seed(0)
    M = 100
    #y = 100x1 + 5x2 + 6
    X1 = np.random.rand(M, 1)
    X2 = np.random.rand(M, 1)
    y = 100 * X1 + 5 * X2 + 6 + np.random.randn(M, 1) * 5

    #theta = (X^TX)^-1X^T
    X_b = np.c_[np.ones((M, 1)), X1, X2]
    theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
    draw_graph(X1, X2, y, theta)