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
#include "pch.h"

using namespace std;
using namespace chrono;

int main(int argc, char** argv){
	int rows, cols, // The dimensions of the global matrix
	    status, // return from MPI calls
	    rank, nprocs,
	    oldrank,
	    topo;  // number of processes in each dimension of grid

	//1. Start the MPI and get rank and number of processes
	//printf("DEBUG:Pre mpi init\n"); fflush(stdout);
	MPI_Comm cartComm = MPI_COMM_WORLD;
	status = MPI_Init(&argc,&argv);
	if (status != MPI_SUCCESS){
		printf("ERROR: MPI_Init returned %d and not MPI_SUCCESS\n", status);
		exit(1);
	}
	status = MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	if (status != MPI_SUCCESS){
		printf("ERROR: MPI_Comm_rank returned %d and not MPI_SUCCESS\n", status);
		exit(1);
	}
	status = MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
	if (status != MPI_SUCCESS){
		printf("ERROR: MPI_Comm_size returned %d and not MPI_SUCCESS\n", status);
		exit(1);
	}
	//printf("DEBUG:Post mpi init\n"); fflush(stdout);
	oldrank = rank;

	// Parse command line arguments
	if (argc > 1 && argc < 4){
		rows = cols = atoi(argv[1]);
		topo = atoi(argv[2]);
	} else if (argc >= 4) {
		printf("Usage: mpirun -n <ranks> ./executable dimension topology \n note: topology*topology should be equal to number of ranks \n");
	} else {
		// set default dimensions 1024x1024
		rows = cols = V_SIZE;
		topo = sqrt(nprocs);
	}
	
	int perProcessDim, // The number of interior points each process gets in each dimension
	    l_rows, l_cols; // The actual number of rows/cols per process
	// Ensure each process has the same values for rows, cols, and topo
	status = MPI_Bcast(&rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
	status = MPI_Bcast(&cols, 1, MPI_INT, 0, MPI_COMM_WORLD);
	status = MPI_Bcast(&topo, 1, MPI_INT, 0, MPI_COMM_WORLD);
	perProcessDim = (int)sqrt((rows*cols)/nprocs);
	// Add 2 for the ghost area in each dimension
	l_rows = l_cols = perProcessDim+2;

	// 2. Create the Carteisian grid
	int dimSize[2] = {nprocs/topo, nprocs/topo};  // Number of processes in each dimension
	int usePeriods[2] = {0, 0}; // Set the grid to not repeat in each dimension
	int coords[2]; // Holds the coordinates of this rank in the cartesian grid
	int neighbors[4]; // Holds the ranks of the neighboring processes

	//printf("DEBUG:Pre mpi cart create\n"); fflush(stdout);
	// Create a cartesian communicator
	MPI_Cart_create(MPI_COMM_WORLD, 2, dimSize, usePeriods, 0, &cartComm);
	//printf("DEBUG:Post mpi cart create\n"); fflush(stdout);

	// Obtain the 2D coordinates in the new communicator
	MPI_Cart_coords(cartComm, rank, 2, coords);
	if ((rank) != oldrank)
	{
		printf("Rank change: from %d to %d\n", oldrank, rank);
	}

	// Obtain the direct neighbor ranks
	MPI_Cart_shift(cartComm, 0, 1, neighbors + DIR_LEFT, neighbors + DIR_RIGHT);
	MPI_Cart_shift(cartComm, 1, 1, neighbors + DIR_TOP, neighbors + DIR_BOTTOM);
	printf("Rank:%d \t Coords(x,y):(%d,%d) \t Neighbors(top, right, bottom, left):(%2d, %2d, %2d, %2d) \n", rank, coords[0], coords[1], neighbors[0], neighbors[1], neighbors[2], neighbors[3]); fflush(stdout);

	// Have rank 0 print out some general info about this setup
	if(rank == 0){
		printf("Global matrix size %d by %d is being divided across %d processes\n", rows, cols, nprocs);
		printf("Processes are laid out in a Cartesian grid with %d processes in each dimension\n", topo);
		printf("Each thread is working on sub-matrices %d by %d with a border of 1 extra cell along each edge\n", perProcessDim, perProcessDim);
	}
	
	float *h_M, *gpu_M;
        high_resolution_clock::time_point t0, t1;
	duration<double> t1sum;

	t0 = high_resolution_clock::now();
	// 3. Allocate memory for the submatrix on each process
	// Each dimension is padded with the size of the ghost area
	h_M = (float*)malloc(l_rows*l_cols*sizeof(float));
	gpu_M = (float*)malloc(l_rows*l_cols*sizeof(float));

	// 4. Fill the A matrices so that the j=0 row is 300 while the other
	// three sides of the matrix are set to 0
	InitializeMatrix_MPI(h_M, l_rows, l_cols, rank, coords);
	// Copy those results into the GPU matrix
	//copyMatrix(h_M, gpu_M, l_rows, l_cols);
	copy(h_M, h_M+(l_rows*l_cols), gpu_M);
	t1 = high_resolution_clock::now();
	t1sum = duration_cast<duration<double>>(t1-t0);
	printf("Rank:%d Init took %f seconds. Begin compute.\n", rank, t1sum.count());

	if (l_rows < 6){
		printf("rank:%d h_M\n", rank);
		PrintMatrix(h_M, l_rows, l_cols);
		printf("rank:%d: gpu_M\n", rank);
		PrintMatrix(gpu_M, l_rows, l_cols);
	}

	// Calculate on host (CPU)
	t0 = high_resolution_clock::now();
	LaplaceJacobi_MPICPU(h_M, l_rows, l_cols, JACOBI_MAX_ITR, JACOBI_TOLERANCE, rank, coords, neighbors);
	t1 = high_resolution_clock::now();
	t1sum = duration_cast<duration<double>>(t1-t0);
	printf("CPU Jacobi Iterative Solver took %f seconds.\n",t1sum.count());

	if (l_rows < 6){
                printf("rank:%d h_M\n");
                PrintMatrix(h_M, l_rows, l_cols);
        }

/*
	// Calculate on device (GPU)
	t0 = high_resolution_clock::now();
        LaplaceJacobi_naiveACC(gpu_M, 1, rows, cols, JACOBI_MAX_ITR, JACOBI_TOLERANCE);
        t1 = high_resolution_clock::now();
        t1sum = duration_cast<duration<double>>(t1-t0);
        printf("ACC Jacobi Iterative Solver took %f seconds.\n",t1sum.count());

	if (rows < 6){
                printf("gpu_M\n");
                PrintMatrix(gpu_M, rows, cols);
        }

	// Verify results
	MatrixVerification(h_M, gpu_M, rows, cols, VERIFY_TOL); 

*/
	// Free host memory
	free(h_M);
	free(gpu_M);

	MPI_Finalize();
	return 0;
}
