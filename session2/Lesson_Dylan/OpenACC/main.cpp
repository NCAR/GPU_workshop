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
	float *h_A, *h_b, *h_x, *gpu_x;
        high_resolution_clock::time_point t0, t1;
        duration<double> t1sum;
        int rows, cols;
	bool check;

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
	h_b = (float*)malloc(rows*sizeof(float));
	h_x = (float*)malloc(rows*sizeof(float));
	gpu_x = (float*)malloc(rows*sizeof(float));

	// 2. Fill the A matrix so it's a strictly diagonally dominant matrix
	InitializeDiagDomMat(h_A, rows, cols, "h_A");
	check = CheckDiagDomMat(h_A, rows, cols);
	if (!check) {
		printf("Error: Matrix is not diagonally dominant\n");
	}

	// Fill x and b with random values
	InitializeMatrixRand(h_b, 1, cols, "h_b");
	InitializeMatrixRand(h_x, 1, cols, "h_x");
	copyMatrix(h_x, gpu_x, 1, cols);
	t1 = high_resolution_clock::now();
	t1sum = duration_cast<duration<double>>(t1-t0);
	printf("Init took %f seconds. Begin compute.\n", t1sum.count());

	if (rows < 6){
		printf("A\n");
		PrintMatrix(h_A, rows, cols);
		printf("b\n");
		PrintMatrix(h_b, rows, 1);
		printf("hx\n");
		PrintMatrix(h_x, rows, 1);
		printf("gpux\n");
		PrintMatrix(gpu_x, rows, 1);
	}

	// Calculate on host (CPU)
	t0 = high_resolution_clock::now();
	Jacobi_naiveCPU(h_A, h_b, h_x, rows, cols, JACOBI_TOLERANCE);
	t1 = high_resolution_clock::now();
	t1sum = duration_cast<duration<double>>(t1-t0);
	printf("CPU Jacobi Iterative Solver took %f seconds.\n",t1sum.count());

	if (rows < 6){
		printf("A\n");
		PrintMatrix(h_A, rows, cols);
		printf("b\n");
		PrintMatrix(h_b, rows, 1);
		printf("hx\n");
		PrintMatrix(h_x, rows, 1);
		printf("gpux\n");
		PrintMatrix(gpu_x, rows, 1);
	}

	// Free host memory
	free(h_A);
	free(h_b);
	free(h_x);
	free(gpu_x);
	return 0;
}
