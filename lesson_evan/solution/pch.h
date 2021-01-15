/* pch.h
 * PreCompiled Header for this lesson
 * Contains the constants, headers, and function prototypes used in other files
 */
//#include <stdio.h>
//#include <time.h> 

#ifndef PCH_H_MATMUL
#define PCH_H_MATMUL

//#define DEFAULT_DIM 8192
#define DEFAULT_DIM 1024;
#define BLOCK_SIZE 32 // The CUDA max is 1024 threads per block
#define MATMUL_A_VAL 3.0F
#define MATMUL_B_VAL 2.0F
#define MATMUL_TOL 1.0E-6F

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

// Functions in common.cpp 
void InitializeMatrixSame(float *array, const int ny, const int nx, const float val);
void InitializeMatrixRand(float *array, const int ny, const int nx);
void MatrixVerification(float *hostC, float *gpuC, const int ny, const int nx, const float fTolerance);
void PrintMatrix(float *matrix, int ny, int nx);

// Device routine (CUDA wrapper)
extern void gpuMatmul(const float *A, const float *B, float *C, const int m, const int p, const int q);

// OpenACC routine
void accMatmul(const float *A, const float *B, float *C, const int m, const int p, const int q);

#endif // PCH_H_MATMUL
