/* pch.h
 * PreCompiled Header for this lesson
 * Contains the constants, headers, and function prototypes used in other files
 */
#include <stdio.h>
#include <time.h> 

#include <stdio.h>
#include <time.h>
#define VERIF_TOL 1.0E-6F
#define MAT_A_VAL 3.0F
#define MAT_B_VAL 2.0F
#define DEFAULT_DIM 1024;
#define BLOCK_SIZE 32 // The CUDA max is 1024 threads per block

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
void cpuMatmul(const float *A, const float *B, float *C, const int m, const int p, const int q);

// Device routine
extern void gpuMult(float *h_A, float *h_B, float *h_check, const int m, const int n, const int p, const int q, const int block_size);

// Functions in common.cpp 
void InitializeMatrixSame(float *array, const int ny, const int nx, const float val, const char *name);
void InitializeMatrixRand(float *array, const int ny, const int nx,const char *name);
void MatrixVerification(float *hostC, float *gpuC, const int ny, const int nx, const float fTolerance);
void PrintMatrix(float *matrix, int ny, int nx);
void copyMatrix(float *src, float *dest, const int ny, const int nx);

