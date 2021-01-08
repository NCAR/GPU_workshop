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

#include "pch.h"
#include <stdio.h>

int main(){
	int rows,cols;
	rows = cols = V_SIZE;

	float hA[M_SIZE], hb[V_SIZE], hx[V_SIZE];
	float gpux[V_SIZE];

	// Dynamic memory allocation
	//float *hA, *hb, *hx, *gpux;
	//hA = new float[rows*cols];
	//hb = new float[rows];
	//hx = new float[rows];
	//gpux = new float[rows];

	// Fill the A matrix so it's a strictly diagonally dominant matrix
	InitializeDiagDomMat(hA, rows, cols);
	printf("Matrix a is diag dom? %s\n", CheckDiagDomMat(hA, rows, cols) ? "true" : "false");

	// Fill x and b with random values
	InitializeMatrixRand(hb, 1, cols);
	InitializeMatrixRand(hx, 1, cols);
	copyMatrix(hx, gpux, 1, cols);

	if (rows < 6){
		printf("A\n");
		PrintMatrix(hA, rows, cols);
		printf("b\n");
		PrintMatrix(hb, rows, 1);
		printf("hx\n");
		PrintMatrix(hx, rows, 1);
		printf("gpux\n");
		PrintMatrix(gpux, rows, 1);
	}

	Jacobi_naiveCPU(hA, hb, hx, rows, cols, JACOBI_TOLERANCE);

	if (rows < 6){
		printf("A\n");
		PrintMatrix(hA, rows, cols);
		printf("b\n");
		PrintMatrix(hb, rows, 1);
		printf("hx\n");
		PrintMatrix(hx, rows, 1);
		printf("gpux\n");
		PrintMatrix(gpux, rows, 1);
	}

	// Free any dynamically allocated memory
	//delete[] hA;
	//delete[] hb;
	//delete[] hx;
	//delete[] gpux;
}
