////////////////////////////////////////////////////////////////////////////
//
// Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
//
// Please refer to the NVIDIA end user license agreement (EULA) associated
// with this source code for terms and conditions that govern your use of
// this software. Any use, reproduction, disclosure, or distribution of
// this software and related documentation outside the terms of the EULA
// is strictly prohibited.
//
////////////////////////////////////////////////////////////////////////////

//
// Matrix multiplication: C = A * B.
// Host code.
//
// This sample implements matrix multiplication as described in Chapter 3
// of the programming guide and uses the CUBLAS library to demonstrate
// the best performance.

// SOME PRECAUTIONS:
// IF WE WANT TO CALCULATE ROW-MAJOR MATRIX MULTIPLY C = A * B,
// WE JUST NEED CALL CUBLAS API IN A REVERSE ORDER: cublasSegemm(B, A)!
// The reason is explained as follows:

// CUBLAS library uses column-major storage, but C/C++ use row-major storage.
// When passing the matrix pointer to CUBLAS, the memory layout alters from
// row-major to column-major, which is equivalent to an implicit transpose.

// In the case of row-major C/C++ matrix A, B, and a simple matrix multiplication
// C = A * B, we can't use the input order like cublasSgemm(A, B)  because of
// implicit transpose. The actual result of cublasSegemm(A, B) is A(T) * B(T).
// If col(A(T)) != row(B(T)), equal to row(A) != col(B), A(T) and B(T) are not
// multipliable. Moreover, even if A(T) and B(T) are multipliable, the result C
// is a column-based cublas matrix, which means C(T) in C/C++, we need extra
// transpose code to convert it to a row-based C/C++ matrix.

// To solve the problem, let's consider our desired result C, a row-major matrix.
// In cublas format, it is C(T) actually (because of the implicit transpose).
// C = A * B, so C(T) = (A * B) (T) = B(T) * A(T). Cublas matrice B(T) and A(T)
// happen to be C/C++ matrice B and A (still because of the implicit transpose)!
// We don't need extra transpose code, we only need alter the input order!
//
// CUBLAS provides high-performance matrix multiplication.
// See also:
// V. Volkov and J. Demmel, "Benchmarking GPUs to tune dense linear algebra,"
// in Proc. 2008 ACM/IEEE Conf. on Supercomputing (SC '08),
// Piscataway, NJ: IEEE Press, 2008, pp. Art. 31:1-11.
//

// Utilities and system includes
#include <assert.h>
#include <helper_string.h>  // helper for shared functions common to CUDA Samples

// CUDA runtime
#include <cuda_runtime.h>
#include <cublas_v2.h>

// CUDA and CUBLAS functions
#include <helper_functions.h>
#include <helper_cuda.h>

#ifndef min
#define min(a,b) ((a < b) ? a : b)
#endif
#ifndef max
#define max(a,b) ((a > b) ? a : b)
#endif

typedef struct _matrixSize
{
    unsigned int uiWA, uiHA, uiWB, uiHB, uiWC, uiHC;
} sMatrixSize;

////////////////////////////////////////////////////////////////////////////////
//! Run a simple test matrix multiply using CUBLAS
////////////////////////////////////////////////////////////////////////////////
int matrixMultiplyCUBLAS(int row_A, int col_A, int col_B, double *h_A, double *h_B, double *h_C)
{
    sMatrixSize matrix_size;
    matrix_size.uiWA = col_A;
    matrix_size.uiHA = row_A;
    matrix_size.uiWB = col_B;
    matrix_size.uiHB = col_A;
    matrix_size.uiWC = col_B;
    matrix_size.uiHC = row_A;
    unsigned int mem_size_A = sizeof(double) * matrix_size.uiWA * matrix_size.uiHA;
    unsigned int mem_size_B = sizeof(double) * matrix_size.uiWB * matrix_size.uiHB;
    unsigned int mem_size_C = sizeof(double) * matrix_size.uiWC * matrix_size.uiHC;

    // Allocate device memory
    double *d_A, *d_B, *d_C;
    cudaMalloc((void **) &d_A, mem_size_A);
    cudaMalloc((void **) &d_B, mem_size_B);
    cudaMalloc((void **) &d_C, mem_size_C);

    // Copy host memory to device
    cudaMemcpy(d_A, h_A, mem_size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, mem_size_B, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C, mem_size_C, cudaMemcpyHostToDevice);

    // CUBLAS version 2.0
    const double alpha = 1.0;
    const double beta  = 1.0;
    cublasHandle_t handle;

    cublasCreate(&handle);

    // Note cublas is column primary!
    // Need to transpose the order
    cublasDgemm(handle,
                CUBLAS_OP_N,
                CUBLAS_OP_N,
                matrix_size.uiWB, matrix_size.uiHA, matrix_size.uiWA,
                &alpha, d_B, matrix_size.uiWB, d_A, matrix_size.uiWA,
                &beta, d_C, matrix_size.uiWB);

    // copy result from device to host
    cudaMemcpy(h_C, d_C, mem_size_C, cudaMemcpyDeviceToHost);

    // Destroy the handle
    cublasDestroy(handle);

    // Clean up memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // printf("matrixMultiplyCUBLAS is finished successfully.\n");

    return 0;
}

int vectorAddCUBLAS(int n, const double alpha, double *h_A, double *h_B)
{
    unsigned int mem_size_A = sizeof(double) * n;
    unsigned int mem_size_B = sizeof(double) * n;

    // Allocate device memory
    double *d_A, *d_B;
    cudaMalloc((void **) &d_A, mem_size_A);
    cudaMalloc((void **) &d_B, mem_size_B);

    // Copy host memory to device
    cudaMemcpy(d_A, h_A, mem_size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, mem_size_B, cudaMemcpyHostToDevice);

    // CUBLAS version 2.0
    // const double alpha = 1.0;
    const int inca = 1;
    const int incb = 1;
    cublasHandle_t handle;

    cublasCreate(&handle);

    cublasDaxpy(handle, n,
                &alpha,
                d_A, inca,
                d_B, incb);

    // copy result from device to host
    cudaMemcpy(h_B, d_B, mem_size_B, cudaMemcpyDeviceToHost);

    // Destroy the handle
    cublasDestroy(handle);

    // Clean up memory
    cudaFree(d_A);
    cudaFree(d_B);

    return 0;
}

extern "C" {
  int call_matrixMultiplyCUBLAS(int row_A, int col_A, int col_B, double *h_A, double *h_B, double *h_C)
  {
    return matrixMultiplyCUBLAS(row_A, col_A, col_B, h_A, h_B, h_C);
  }

  int call_vectorAddCUBLAS(int n, const double alpha, double *h_A, double *h_B)
  {
    return vectorAddCUBLAS(n, alpha, h_A, h_B);
  }
}
