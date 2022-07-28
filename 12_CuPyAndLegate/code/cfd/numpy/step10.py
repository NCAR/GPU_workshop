# Solve Laplace and Poisson equations for next steps
# 
import numpy as np
# Import Cupy
from matplotlib import pyplot, cm
from mpl_toolkits.mplot3d import Axes3D
import timeit
import sys
    
# Handle plotting 
def plot2d(x, y, p):
    fig = pyplot.figure(figsize=(11, 7), dpi=100)
    ax = fig.add_subplot(projection='3d')
    # Handle to pull Cupy datatype but plot on the CPU
    X, Y = np.meshgrid(x, y)
    surf = ax.plot_surface(X, Y, p[:], rstride=1, cstride=1, cmap=cm.viridis,
                           linewidth=0, antialiased=False)
    ax.view_init(30, 225)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')


if __name__ == "__main__":
    # Parameter Init
    nx = 50
    ny = 50
    nt = 100
    xmin = 0
    xmax = 2
    ymin = 0
    ymax = 1
    dx = (xmax-xmin) / (nx-1)
    dy = (ymax-ymin) / (ny-1)

    # Change to Cupy arrays
    x = np.linspace(xmin, xmax, nx)
    y = np.linspace(xmin, xmax, ny)
    p = np.zeros((ny, nx))
    pd = np.zeros((ny, nx))
    b = np.zeros((ny, nx))

    # Initial Conditions
    b[int(ny/4), int(nx/4)] = 100
    b[int(3*ny/4), int(3*nx/4)] = -100

    # Loop through steps
    for it in range(nt):
        # Change to Cupy copy
        pd = p.copy()
        p[1:-1,1:-1] = (((pd[1:-1,2:] + pd[1:-1,:-2]) * dy**2
                        + (pd[2:,1:-1] + pd[:-2,1:-1]) * dx**2
                        - b[1:-1,1:-1] * dx**2 * dy**2) 
                        / (2 * (dx**2 + dy**2)))
        p[0,:] = 0
        p[ny-1,:] = 0
        p[:,0] = 0
        p[:,nx-1] = 0

    # Plotting Initial Conditions
    plot2d(x, y, p)