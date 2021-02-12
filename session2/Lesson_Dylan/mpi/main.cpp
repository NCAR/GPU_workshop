/* stencil.cpp
 * Contains the main for the lesson
 */

/*
 * The Jacobi method is an iterative solver for linear systems 
 * of equations of the form Ax=b where b is a vector of length N
 * A is a square matrix N by N, and x is a vector to be found of
 * length N. This method is guaranteed to converge for strictly
 * diagonally dominant (the absolute value of the entries on the
 * diagonal is greater than the sum of the absolute values of
 * the other entries on the same row)
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <ctime>
#include <chrono>
#include <ratio> 
#include "pch.h"
#include <mpi.h>
#include <iostream>

using namespace std;

int main(int argc, char** argv){
	int rank, size;
	int usePeriods[2] = {0, 0};
	int newCoords[2];
	MPI_Comm cartComm = MPI_COMM_WORLD;
	MPI_Init(&argc,&argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	int rows, cols;
	int topo;
	int oldrank = rank;

	// Parse command line arguments
	if (argc > 1 && argc < 4){
		rows = cols = atoi(argv[1]);
		topo = atoi(argv[2]);
	} else if (argc >= 4) {
		printf("Usage: mpirun -n <ranks> ./executable dimension topology \n note: topology*topology should be equal to number of ranks \n");
	} else {
		// set default dimensions 8192x8192
		rows = cols = V_SIZE;
		topo = sqrt(size);
	}
	
	int neighbors[4];
	int dimSize[2] = {size/topo,size/topo};
	int topIndex[2];

	// Create a carthesian communicator
	MPI_Cart_create(MPI_COMM_WORLD, 2, dimSize, usePeriods, 0, &cartComm);

	// Obtain the 2D coordinates in the new communicator
	MPI_Cart_coords(cartComm, rank, 2, newCoords);
	topIndex = {newCoords[0],newCoords[1]};
	if ((rank) != oldrank)
	{
		printf("Rank change: from %d to %d\n", oldrank, rank);
	}

	// Obtain the direct neighbor ranks
	MPI_Cart_shift(cartComm, 0, 1, neighbors + DIR_LEFT, neighbors + DIR_RIGHT);
	MPI_Cart_shift(cartComm, 1, 1, neighbors + DIR_TOP, neighbors + DIR_BOTTOM);

	printf("Rank: %d \t Neighbors(top, right, bottom, left): %2d, %2d, %2d, %2d \n", rank, neighbors[0], neighbors[1], neighbors[2], neighbors[3]);


/*
	using namespace std::chrono;
	float *h_A, *gpu_A;
        high_resolution_clock::time_point t0, t1;
        duration<double> t1sum;
        int rows, cols;

	t0 = high_resolution_clock::now();
	// 1. Allocate memory to host matrices
	h_A = (float*)malloc(rows*cols*sizeof(float));
	gpu_A = (float*)malloc(rows*cols*sizeof(float));

	// 2. Fill the A matrices so that the j=0 row is 300 while the other
	// three sides of the matrix are set to 0
	InitializeMatrixSame(h_A, rows, cols, 0.0f, "h_A");  
	InitializeMatrixSame(h_A, 1, cols, 300.0f, "h_A");
	copyMatrix(h_A, gpu_A, rows, cols);
	t1 = high_resolution_clock::now();
	t1sum = duration_cast<duration<double>>(t1-t0);
	printf("Init took %f seconds. Begin compute.\n", t1sum.count());

	if (rows < 6){
		printf("h_A\n");
		PrintMatrix(h_A, rows, cols);
		printf("gpu_A\n");
		PrintMatrix(gpu_A, rows, cols);
	}

	// Calculate on host (CPU)
	t0 = high_resolution_clock::now();
	LaplaceJacobi_naiveCPU(h_A, 1, rows, cols, JACOBI_MAX_ITR, JACOBI_TOLERANCE);
	t1 = high_resolution_clock::now();
	t1sum = duration_cast<duration<double>>(t1-t0);
	printf("CPU Jacobi Iterative Solver took %f seconds.\n",t1sum.count());

	if (rows < 6){
                printf("h_A\n");
                PrintMatrix(h_A, rows, cols);
        }

	// Calculate on device (GPU)
	t0 = high_resolution_clock::now();
        LaplaceJacobi_naiveACC(gpu_A, 1, rows, cols, JACOBI_MAX_ITR, JACOBI_TOLERANCE);
        t1 = high_resolution_clock::now();
        t1sum = duration_cast<duration<double>>(t1-t0);
        printf("ACC Jacobi Iterative Solver took %f seconds.\n",t1sum.count());

	if (rows < 6){
                printf("gpu_A\n");
                PrintMatrix(gpu_A, rows, cols);
        }

	// Verify results
	MatrixVerification(h_A, gpu_A, rows, cols, VERIFY_TOL); 


	// Free host memory
	free(h_A);
	free(gpu_A);
*/
	MPI_Finalize();
	return 0;
}
