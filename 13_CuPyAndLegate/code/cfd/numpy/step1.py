# Implementing the linear convection algorithm
# Port this code using CuPy
import numpy as np
# Import cupy
from matplotlib import pyplot
import time
import sys


if __name__ == "__main__":
    nx = 41
    dx = 2 / (nx - 1)
    nt = 25
    dt = 0.025
    c = 1

    # Change to CuPy array
    u = np.ones(nx)
    u[int(.5 / dx):int(1 / dx + 1)] = 2
    print(u)

    # Copy data to the CPU so it can be plotted
    pyplot.plot(np.linspace(0, 2, nx), u)

    # Change to CuPy array
    un = np.ones(nx)
 
    for n in range(nt):
        # Change copy function to CuPy function
        un = u.copy() 
        for i in range(1, nx):
            u[i] = un[i] - c * dt/dx * (un[i] - un[i-1])

# Move the u data to the CPU so it can be plotted
pyplot.plot(np.linspace(0, 2, nx), u)