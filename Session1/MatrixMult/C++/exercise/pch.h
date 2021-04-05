/* pch.h
 * PreCompiled Header for this lesson
 * Contains the constants, headers, and function prototypes used in other files
 */
#include <stdio.h>
#include <time.h> 

#ifndef PCH_H_MATRIX_ADD
#define PCH_H_MATRIX_ADD

#define DEFAULT_DIM 1024;
#define BLOCK_SIZE 32 // The CUDA max is 1024 threads per block
#define MAT_A_VAL 3.0F
#define MAT_B_VAL 2.0F
#define VERIF_TOL 1.0E-6F

// Host routine
void cpu_matrix_mult(const float *A, const float *B, float *C, const int rowsA, const int colsB,\
  const int rowsB);

// Functions in common.cpp 
void InitializeMatrixSame(float *array, const int ny, const int nx, const float val,const char *name);
void InitializeMatrixRand(float *array, const int ny, const int nx, const char *name);
void MatrixVerification(float *hostC, float *gpuC, const int ny, const int nx, const float fTolerance);
void PrintMatrix(float *matrix, int ny, int nx);

// OpenACC routine
void openacc_matrix_mult(const float *A, const float *B, float *C, const int rowsA, const int colsB,\
  const int rowsB);

#endif // PCH_H_MATRIX_ADD
