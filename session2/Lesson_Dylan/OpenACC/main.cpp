/* main.cpp
 * Contains the main for the lesson
 * by: G. Dylan Dickerson (gdicker@ucar.edu)
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
#include "pch.h"

int main(int argc, char* argv[]){
	using namespace std::chrono;
	float *h_M, *gpu_M;
        high_resolution_clock::time_point t0, t1;
        duration<double> t1sum;
        int rows, cols;

	// Parse command line arguments
	if (argc > 1 && argc < 3){
		rows = cols = atoi(argv[1]);
	} else if (argc >= 3) {
		printf("Usage: ./executable dim\n");
	} else {
		// set default dimensions 1024x1024
		rows = cols = V_SIZE;
	}

	t0 = high_resolution_clock::now();
	// 1. Allocate memory to host matrices
	h_M = (float*)malloc(rows*cols*sizeof(float));
	gpu_M = (float*)malloc(rows*cols*sizeof(float));

	// 2. Fill the A matrices so that the j=0 row is 300 while the other
	// three sides of the matrix are set to 0
	InitializeMatrixSame(h_M, rows, cols, 0.0f, "h_M");  
	InitializeMatrixSame(h_M, 1, cols, 300.0f, "h_M");
	copyMatrix(h_M, gpu_M, rows, cols);
	t1 = high_resolution_clock::now();
	t1sum = duration_cast<duration<double>>(t1-t0);
	printf("Init took %f seconds. Begin compute.\n", t1sum.count());

	if (rows < 6){
		printf("h_M\n");
		PrintMatrix(h_M, rows, cols);
		printf("gpu_M\n");
		PrintMatrix(gpu_M, rows, cols);
	}

	// Calculate on host (CPU)
	t0 = high_resolution_clock::now();
	LaplaceJacobi_naiveCPU(h_M, 1, rows, cols, JACOBI_MAX_ITR, JACOBI_TOLERANCE);
	t1 = high_resolution_clock::now();
	t1sum = duration_cast<duration<double>>(t1-t0);
	printf("CPU Jacobi Iterative Solver took %f seconds.\n",t1sum.count());

	if (rows < 6){
                printf("h_M\n");
                PrintMatrix(h_M, rows, cols);
        }

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

	// Free host memory
	free(h_M);
	free(gpu_M);
	return 0;
}
