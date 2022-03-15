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
#include "pch.h"

using namespace std;
using namespace chrono;

int main(int argc, char** argv){
    int rows, cols, // The dimensions of the global matrix
        status, // return from MPI calls
        rank, nprocs,
        topo;  // number of processes in each dimension of grid

    LJ_return cpu_ret, mpi_ret;

    // Start the MPI and get rank and number of processes
    MPI_Comm cartComm = MPI_COMM_WORLD;
    status = MPI_Init(&argc,&argv);
    status = MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    status = MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    // Parse command line arguments
    if (argc > 1 && argc < 4){
        rows = cols = atoi(argv[1]);
        topo = atoi(argv[2]);
    } else if (argc >= 4) {
        printf("Usage: mpirun -n <ranks> ./executable dimension topology \n note: topology*topology should be equal to number of ranks and topology should evenly divide dimension \n");
        exit(1);
    } else {
        // set default dimensions 1024x1024
        rows = cols = V_SIZE;
        topo = sqrt(nprocs);
    }
    // Check input arguments
    if (rows % topo != 0){
        printf("ERROR: topology doesn't evenly divide the dimension given\n");
        printf("Usage: mpirun -n <ranks> ./executable dimension topology \n note: topology*topology should be equal to number of ranks and topology should evenly divide dimension \n");
        exit(1);
    }
    if (topo*topo != nprocs){
        printf("ERROR: Topology given isn't the square root of the number of ranks given\n");
        printf("Usage: mpirun -n <ranks> ./executable dimension topology \n note: topology*topology should be equal to number of ranks and topology should evenly divide dimension \n");
        exit(1);
    }
    
    int perProcessDim, // The number of interior points each process gets in each dimension
        l_rows, l_cols; // The actual (local) number of rows/cols per process
    // Ensure each process has the same values for rows, cols, and topo
    status = MPI_Bcast(&rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
    status = MPI_Bcast(&cols, 1, MPI_INT, 0, MPI_COMM_WORLD);
    status = MPI_Bcast(&topo, 1, MPI_INT, 0, MPI_COMM_WORLD);
    perProcessDim = (int)sqrt((rows*cols)/nprocs);
    // Add 2 for the ghost area in each dimension
    l_rows = l_cols = perProcessDim+2;
    // Add 2 for the border along the outer edges
    rows += 2;
    cols += 2;

    // Create the Carteisian grid
    int dimSize[2] = {nprocs/topo, nprocs/topo};  // Number of processes in each dimension
    int usePeriods[2] = {0, 0}; // Set the grid to not repeat in each dimension
    int coords[2]; // Holds the coordinates of this rank in the cartesian grid
    int neighbors[4]; // Holds the ranks of the neighboring processes

    // Create a cartesian communicator
    MPI_Cart_create(MPI_COMM_WORLD, 2, dimSize, usePeriods, 0, &cartComm);

    // Obtain the 2D coordinates in the new communicator
    MPI_Cart_coords(cartComm, rank, 2, coords);

    // Obtain the direct neighbor ranks
    MPI_Cart_shift(cartComm, 0, 1, neighbors + DIR_LEFT, neighbors + DIR_RIGHT);
    MPI_Cart_shift(cartComm, 1, 1, neighbors + DIR_TOP, neighbors + DIR_BOTTOM);

    // Have rank 0 print out some general info about this setup
    if(rank == 0){
        printf("Global matrix with %d by %d interior points is being divided across %d processes\n", rows-2, cols-2, nprocs);
        printf("(Actual global size is %d by %d with border)\n", rows, cols);
        printf("Processes are laid out in a Cartesian grid with %d processes in each dimension\n", topo);
        printf("Each thread is working on sub-matrices with %d by %d interior points\n", perProcessDim, perProcessDim);
        printf("------------------------------------------------------------------------------\n\n");
        fflush(stdout);
    } MPI_Barrier(MPI_COMM_WORLD);
    
    float *h_M, *global_M;
    double t0, t1; 
    double elapsedT, maxT;

    t0 = MPI_Wtime(); 
    // Allocate memory for the submatrix on each process
    // Each dimension is padded with the size of the ghost area
    h_M = (float*)malloc(l_rows*l_cols*sizeof(float));

    // Fill the M matrices so that the j=0 row is 300 while the other
    // three sides of the matrix are set to 0
    InitializeLJMatrix_MPI(h_M, l_rows, l_cols, rank, coords);
    t1 = MPI_Wtime();
    elapsedT = t1 - t0;
    MPI_Reduce(&elapsedT, &maxT, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0){
        printf("Max init time was %f seconds. Begin compute\n", maxT);
        fflush(stdout);
    } MPI_Barrier(MPI_COMM_WORLD);

    // Calculate on host (CPU)
    t0 = MPI_Wtime();
    mpi_ret = LaplaceJacobi_MPICPU(h_M, l_rows, l_cols, rank, neighbors);
    t1 = MPI_Wtime();
    elapsedT = t1 - t0;
    MPI_Reduce(&elapsedT, &maxT, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0){
        printf("Max MPI CPU compute time was %f seconds for %d iterations to reach %f error.\n", maxT, mpi_ret.itr, mpi_ret.error);
        fflush(stdout);
    } MPI_Barrier(MPI_COMM_WORLD);

// Uncomment the following lines to compare against the single thread version
/*
    if (rank == 0){
        global_M = (float*)malloc(rows*cols*sizeof(float));
        InitializeMatrixSame(global_M, rows, cols, 0.0f, "global_M");
        InitializeMatrixSame(global_M, 1, cols, 300.0f, "global_M");
        t0 = MPI_Wtime(); 
        cpu_ret = LaplaceJacobi_naiveCPU(global_M, rows, cols);
        t1 = MPI_Wtime();
        elapsedT = t1 - t0;
        MPI_Reduce(&elapsedT, &maxT, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0){
            printf("Max one thread CPU compute time was %f seconds for %d iterations to reach %f error.\n", maxT, mpi_ret.itr, mpi_ret.error);
        }
    } MPI_Barrier(MPI_COMM_WORLD);

    Verify_MPIvsOneThread(global_M, rows, cols, h_M, l_rows, l_cols, 
                  perProcessDim, rank, nprocs, coords);
    if (rank == 0){ free(global_M);}
*/
    
    // Free host memory
    free(h_M);
    
    MPI_Finalize();
    return 0;
}
