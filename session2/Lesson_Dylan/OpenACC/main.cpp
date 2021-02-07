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

int main(int argc, char* argv[]){
	using namespace std::chrono;
	float *h_A, *gpu_A;
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
	return 0;
}
