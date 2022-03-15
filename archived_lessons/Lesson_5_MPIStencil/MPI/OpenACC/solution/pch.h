/* pch.h
 * PreCompiled Header for this lesson
 * Contains the constants, headers, and function prototypes used in other files
 * by: G. Dylan Dickerson (gdicker@ucar.edu) and Supreeth Suresh (ssuresh@ucar.edu)
 */

#ifndef PCH_H_STENCIL
#define PCH_H_STENCIL

// Define the size of the vectors & matrix
#define V_SIZE 1024
#define M_SIZE V_SIZE*V_SIZE

#define DEFAULT_DOMAIN_SIZE_PER_RANK 256

#define DIR_TOP             0
#define DIR_RIGHT           1
#define DIR_BOTTOM          2
#define DIR_LEFT            3

#define HasNeighbor(neighbors, dir) (neighbors[dir] != MPI_PROC_NULL)

// Convergence tolerance
#define JACOBI_TOLERANCE 1.0E-6F
// Verification tolerance
#define VERIFY_TOL   1.0E-6F

// Convinience struct to return values from LapalceJacobi_x
struct LJ_return
{
    int itr;
    float error;
};

// =======================
// Functions in common.cpp
// =======================
void InitializeMatrixSame(float *array, const int ny, const int nx, const float val, const char *name);
void InitializeMatrixRand(float *array, const int ny, const int nx, const char *name);
void MatrixVerification(float *hostC, float *gpuC, const int ny, const int nx, const float fTolerance);
void PrintMatrix(float *matrix, int ny, int nx);
// MPI functions
void InitializeLJMatrix_MPI(float *M, const int ny, const int nx, const int rank, const int *coords);
void MatrixVerification_MPI(float *hostC, float *gpuC, const int ny, const int nx, const float fTolerance, int rank);

// =============
// Host routines
// =============

// ==========================
// Device and OpenACC Routines
// ==========================
void mapGPUToMPIRanks(int rank);
LJ_return LaplaceJacobi_naiveACC(float *M, const int ny, const int nx);
LJ_return LaplaceJacobi_MPIACC(float *M, const int ny, const int nx,
                               const int rank, const int *neighbors);
#endif // PCH_H_STENCIL
