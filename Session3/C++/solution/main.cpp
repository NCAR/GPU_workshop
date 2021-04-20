/* main.cpp
 * Contains the main for the lesson
 * by: G. Dylan Dickerson (gdicker@ucar.edu) and Supreeth Suresh (ssuresh@ucar.edu)
 */

/*
 * This lesson applies the Jacobi iterative solver to find the steady-state
 * of a system defined by the Lapalce Equation (second derivatives equal to a
 * constant) on a 2D grid (matrix). Given a boundary area where points aren't
 * updated, the solver finds the steady-state of the interior (a.k.a. 
 * non-boundary) points by doing repeated local averaging of the points above, 
 * to the right, below, and to the left of each interior point. This can be
 * interpreted as a version of the Jacobi iterative solver for a system Ax=b 
 * (which has guaranteed convergence if A is diagonally dominant). If we think
 * of a flattened vector (x) made from the points in M, then the coefficients
 * from the local averaging can be thought of as the elements of A (see another
 * source for a more complete explaination, but this A is diagonally dominant).
 * One problem with the Jacobi method does mean that we need 2 versions of the
 * M matrix to carry out the computation.
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <ctime>
#include <chrono>
#include <ratio> 
#include <mpi.h>
#include <math.h>
#include <algorithm>
#include <thread>
#include <unistd.h>
#include <openacc.h>
#include "pch.h"

using namespace std;
using namespace chrono;

int main(int argc, char** argv){
    int rows, cols, // The dimensions of the global matrix
        status, // return from MPI calls
        rank, nprocs;

    LJ_return gpu_ret;

    // Start the MPI and get rank and number of processes
    MPI_Comm cartComm = MPI_COMM_WORLD;
    status = MPI_Init(&argc,&argv);
    status = MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    status = MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    // Parse command line arguments
    if (argc > 1 && argc < 3){
        rows = cols = atoi(argv[1]);
    } else if (argc >= 3) {
        printf("Usage: mpirun -n <ranks> ./executable dimension \n");
        exit(1);
    } else {
        // set default dimensions 1024x1024
        rows = cols = V_SIZE;
    }
    // Check input arguments
    if (rows % nprocs != 0){
        printf("ERROR: rows %d can't be evenly divided by nprocs %d\n",rows, nprocs);
        exit(1);
    }
    
    int perProcessDim, // The number of interior points each process gets in each dimension
        l_rows, l_cols; // The actual (local) number of rows/cols per process
    // Ensure each process has the same values for rows, cols
    status = MPI_Bcast(&rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
    status = MPI_Bcast(&cols, 1, MPI_INT, 0, MPI_COMM_WORLD);
    perProcessDim = rows/nprocs;
    // Add 2 for the ghost area in each dimension
    l_rows = perProcessDim+2;
    l_cols = cols;

    // Create the Carteisian grid
    int dimSize[1] = {nprocs};  // Number of processes in each dimension
    int usePeriods[1] = {0}; // Set the grid to not repeat in each dimension
    int coords[1]; // Holds the coordinates of this rank in the cartesian grid
    int neighbors[2]; // Holds the ranks of the neighboring processes

    // Create a cartesian communicator
    MPI_Cart_create(MPI_COMM_WORLD, 1, dimSize, usePeriods, 0, &cartComm);

    // Obtain the 1D coordinate in the new communicator
    MPI_Cart_coords(cartComm, rank, 1, coords);

    // Obtain the direct neighbor ranks
    MPI_Cart_shift(cartComm, 0, 1, neighbors + DIR_TOP, neighbors + DIR_BOTTOM);

    // Have rank 0 print out some general info about this setup
    if(rank == 0){
        printf("Global matrix with %d by %d interior points is being divided across %d processes\n", rows, cols, nprocs);
        printf("Processes are laid out in 1D stripes\n");
        printf("Each thread is working on sub-matrices with %d by %d interior points\n", perProcessDim, l_cols);
        printf("------------------------------------------------------------------------------\n\n");
        fflush(stdout);
    } MPI_Barrier(MPI_COMM_WORLD);

    // Initialize acc and map ranks to GPUs
    acc_init(acc_device_nvidia);
    mapGPUToMPIRanks(rank);
    
    float *gpu_M;
    double t0, t1; 
    double elapsedT, maxT;

    t0 = MPI_Wtime(); 
    // Allocate memory for the submatrix on each process
    // Each dimension is padded with the size of the ghost area
    gpu_M = (float*)malloc(l_rows*l_cols*sizeof(float));

    // Fill the M matrices so that the j=0 row is 300 while the other
    // three sides of the matrix are set to 0
    InitializeLJMatrix_MPI(gpu_M, l_rows, l_cols, rank, coords);
    t1 = MPI_Wtime();
    elapsedT = t1 - t0;
    MPI_Reduce(&elapsedT, &maxT, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0){
        printf("Max init time was %f seconds. Begin compute\n", maxT);
        fflush(stdout);
    } MPI_Barrier(MPI_COMM_WORLD);

    // Calculate on device (GPU)
    t0 = MPI_Wtime();
    gpu_ret = LaplaceJacobi_MPIACC(gpu_M, l_rows, l_cols, rank, neighbors);
    t1 = MPI_Wtime();
    elapsedT = t1 - t0;
    MPI_Reduce(&elapsedT, &maxT, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0){
        printf("Max MPI ACC compute time was %f seconds for %d iterations to reach %f error.\n", maxT, gpu_ret.itr, gpu_ret.error);
        fflush(stdout);
    } MPI_Barrier(MPI_COMM_WORLD);

    // Free host memory
    free(gpu_M);
    
    MPI_Finalize();
    return 0;
}
