from cmath import log
import matplotlib
from matplotlib import scale
import numpy as np
import cupy as cp
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # # Load the performance data for double, constant timesteps
    # gridsize_500ts = np.array([41, 81, 128, 256, 512])
    # data_cpu_500ts = np.array([0.70, 1.30, 2.80, 13.93, 61.11])
    # data_gpu_500ts = np.array([6.49, 6.54, 6.44, 6.55, 6.42])

    # gridsize_4000ts = np.array([41, 81, 128, 256, 512])
    # data_cpu_4000ts = np.array([5.41, 10.76, 21.97, 51.22, 494.04])
    # data_gpu_4000ts = np.array([51.78, 51.53, 51.89, 51.22, 51.66])

    # Load the performance data for double, constant timesteps
    gridsize_100ts = np.array([32, 64, 128, 256, 512, 1024, 2048, 4096])
    data_cpu_100ts = np.array([0.14, 0.23, 0.65, 3.22, 15.64, 82.43, 462.66, 1959.14])
    data_gpu_100ts = np.array([1.34, 1.34, 1.32, 1.36, 1.36, 1.58, 5.55, 21.39])

    # Load the performance data for double, constant gridsize

    # Plot
    plt.figure(figsize=(12,6))
    plt.subplot(121)
    plt.plot(gridsize_100ts, data_cpu_100ts, marker='o', label='Numpy')
    plt.xticks([32, 64, 128, 256, 512, 1024, 2048, 4096])
    plt.xscale("log")
    plt.title("Cavity Flow Numpy (100 Timesteps)")
    plt.xlabel("Grid Size (Log)", size=14)
    plt.ylabel("Wall time (seconds)", size=14)
    plt.legend()
    plt.subplot(122)
    plt.plot(gridsize_100ts, data_gpu_100ts, marker='o', color='darkorange', label='Cupy')
    plt.xticks([32, 64, 128, 256, 512, 1024, 2048, 4096])
    plt.xscale("log")
    plt.title("Cavity Flow Cupy (100 Timesteps)")
    plt.xlabel("Grid Size (Log)", size=14)
    plt.ylabel("Wall time (seconds)", size=14)
    plt.legend()
    # plt.subplot(221)
    # plt.plot(gridsize_4000ts, data_cpu_4000ts, marker='o', label='Numpy')
    # plt.plot(gridsize_4000ts, data_gpu_4000ts, marker='o', label='Cupy')
    # plt.xticks([41, 81, 128, 256, 512])
    # plt.xlabel("Grid Size")
    # plt.ylabel("Wall time (seconds)")
    # plt.legend()
    # plt.subplot(222)
    # plt.plot(gridsize_4000ts, data_cpu_4000ts, marker='o', label='Numpy')
    # plt.plot(gridsize_4000ts, data_gpu_4000ts, marker='o', label='Cupy')
    # plt.xticks([41, 81, 128, 256, 512])
    # plt.xlabel("Grid Size")
    # plt.ylabel("Wall time (seconds)")
    # plt.legend()