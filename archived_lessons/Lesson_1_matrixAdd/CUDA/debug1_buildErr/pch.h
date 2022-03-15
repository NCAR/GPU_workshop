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
#define VERIF_TOL 1.0E-6F
#define MAT_A_VAL 3.0F
#define MAT_B_VAL 2.0F

#define cudaCheckErrors(msg) \
  do { \
    cudaError_t __err = cudaGetLastError(); \
    if (__err != cudaSuccess) { \
      fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
        msg, cudaGetErrorString(__err), \
        __FILE__, __LINE__); \
      fprintf(stderr, "*** FAILED - ABORTING\n"); \
      exit(1); \
    } \
  } while (0)

// Host routine
void cpu_matrix_add(const float *A, const float *B, float *C, const int dx,\
  const int dy);

// Functions in common.cpp 
void InitializeMatrixSame(float *array, const int ny, const int nx, const float val, const char *name);
void InitializeMatrixRand(float *array, const int ny, const int nx, const char *name);
void MatrixVerification(float *hostC, float *gpuC, const int ny, const int nx, const float fTolerance);
void PrintMatrix(float *matrix, int ny, int nx);

// Device routine
extern void gpu_matrix_add(const float *h_A, const float *h_B, float *h_C,\
   const int dx, const int dy);

// OpenACC function
void openacc_matrix_add(const float *A, const float *B, float *C, const int dx, \
const int dy); 

#endif // PCH_H_MATRIX_ADD
