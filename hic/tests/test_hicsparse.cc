/*
 * (C) Copyright 2024- ECMWF.
 *
 * This software is licensed under the terms of the Apache Licence Version 2.0
 * which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
 * In applying this licence, ECMWF does not waive the privileges and immunities
 * granted to it by virtue of its status as an intergovernmental organisation
 * nor does it submit to any jurisdiction.
 */
#include <iostream>
#include <functional>
#include <vector>
#include <utility>
#include <cassert>

#include "hic/hic.h"
#include "hic/hicsparse.h"

// -----------------------------------------------------------------------------

// void spmm(const SparseMatrixView& A,
//           const DenseMatrixView& B,
//           DenseMatrixView& C) {

//     std::cout << "(Debug) A.rows=" << A.rows() << std::endl;
//     std::cout << "(Debug) A.cols=" << A.cols() << std::endl;
//     std::cout << "(Debug) B.rows=" << B.shape(0) << std::endl;
//     std::cout << "(Debug) B.cols=" << B.shape(1) << std::endl;
//     std::cout << "(Debug) C.rows=" << C.shape(0) << std::endl;
//     std::cout << "(Debug) C.cols=" << C.shape(1) << std::endl;

//     assert(A.cols() == B.shape(0));
//     assert(A.rows() == C.shape(0));
//     assert(B.shape(1) == C.shape(1));
//     assert(B.stride(0) == 1 || B.stride(1) == 1);
//     assert(C.stride(0) == 1 || C.stride(1) == 1);

//     using Scalar = double;
//     using Index = int;
//     using Size = size_t;
    
//     const Size values_bytes = A.nonZeros() * sizeof(Scalar);
//     const Size column_indices_bytes = A.nonZeros() * sizeof(Index);
//     const Size row_offsets_bytes = (A.rows() + 1) * sizeof(Index);

//     // Copy the sparse matrix `A` on to the device
//     Index* drow_offsets;
//     Index* dcolumn_indices;
//     Scalar* dvalues;
//     {
//         hicMalloc((void**)&drow_offsets, row_offsets_bytes);
//         hicMalloc((void**)&dcolumn_indices, column_indices_bytes);
//         hicMalloc((void**)&dvalues, values_bytes);

//         hicMemcpy(drow_offsets, A.row_offsets(), row_offsets_bytes, hicMemcpyHostToDevice);
//         hicMemcpy(dcolumn_indices, A.column_indices(), column_indices_bytes, hicMemcpyHostToDevice);
//         hicMemcpy(dvalues, A.data(), values_bytes, hicMemcpyHostToDevice);
//     }

//     // Allocate memory on the device for the dense matrices and copy the data across
//     Scalar* d_B;
//     Scalar* d_C;
//     {
//         const Size B_bytes = B.shape(0) * B.shape(1) * sizeof(Scalar);
//         const Size C_bytes = C.shape(0) * C.shape(1) * sizeof(Scalar);

//         hicMalloc((void**)&d_B, B_bytes);
//         hicMalloc((void**)&d_C, C_bytes);

//         hicMemcpy(d_B, B.data(), B_bytes, hicMemcpyHostToDevice);
//         hicMemcpy(d_C, C.data(), C_bytes, hicMemcpyHostToDevice);
//     }
    

//     const auto dataType = HIC_R_64F;
//     const auto indexType = HICSPARSE_INDEX_32I;
//     const auto B_order = (B.stride(0)==1) ? HICSPARSE_ORDER_COL : HICSPARSE_ORDER_ROW;
//     const auto B_leadingAxis = (B.stride(0)==1) ? 0 : 1;
//     const auto C_order = (C.stride(0)==1) ? HICSPARSE_ORDER_COL : HICSPARSE_ORDER_ROW;
//     const auto C_leadingAxis = (C.stride(0)==1) ? 0 : 1;

//     hicsparseHandle_t handle;
//     hicsparseCreate(&handle);
 
//     // Create sparse matrix descriptor
//     hicsparseConstSpMatDescr_t matA;
//     hicsparseCreateConstCsr(
//         &matA,
//         A.rows(), A.cols(), A.nonZeros(),
//         drow_offsets,
//         dcolumn_indices,
//         dvalues,
//         indexType,
//         indexType,
//         HICSPARSE_INDEX_BASE_ZERO,
//         dataType);
    
//     // Create dense matrix descriptors
//     hicsparseConstDnMatDescr_t matB;
//     hicsparseCreateConstDnMat(
//         &matB,
//         B.shape(0),             // row
//         B.shape(1),             // cols
//         B.shape(B_leadingAxis), // leading dimension
//         d_B,
//         dataType,
//         B_order);

//     hicsparseDnMatDescr_t matC;
//     hicsparseCreateDnMat(
//         &matC,
//         C.shape(0),             // row
//         C.shape(1),             // cols
//         C.shape(C_leadingAxis), // leading dimension
//         d_C,
//         dataType,
//         C_order);

//     // Perform SpMM
//     const Scalar alpha = 1;
//     const Scalar beta = 0;
//     size_t bufferSize = 0;
//     void *externalBuffer = nullptr;

//     const hicsparseSpMMAlg_t algorithm = HICSPARSE_SPMM_ALG_DEFAULT;
    
//     hicsparseSpMM_bufferSize(handle, HICSPARSE_OPERATION_NON_TRANSPOSE, HICSPARSE_OPERATION_NON_TRANSPOSE,
//                             &alpha, matA, matB, &beta, matC, dataType, algorithm, &bufferSize);
    
//     hicMalloc(&externalBuffer, bufferSize);
    
//     hicsparseSpMM(handle, HICSPARSE_OPERATION_NON_TRANSPOSE, HICSPARSE_OPERATION_NON_TRANSPOSE,
//                  &alpha, matA, matB, &beta, matC, dataType, algorithm, externalBuffer);

//     // copy result from d_C back to host
//     hicMemcpy(C.data(), d_C, C.shape(0) * C.shape(1) * sizeof(Scalar), hicMemcpyDeviceToHost);

//     hicFree(externalBuffer);
//     hicsparseDestroyDnMat(matC);
//     hicsparseDestroyDnMat(matB);
//     hicsparseDestroySpMat(matA);
//     hicFree(dvalues);
//     hicFree(dcolumn_indices);
//     hicFree(drow_offsets);
//     hicFree(d_B);
//     hicFree(d_C);
//     hicsparseDestroy(handle);

// }

int test_hicsparseCreate() {
    std::cout << "--- " << __FUNCTION__ << std::endl;
    hicsparseHandle_t handle;
    HICSPARSE_CALL( hicsparseCreate(&handle) );
    HICSPARSE_CALL( hicsparseDestroy(handle) );
    std::cout << "--- " << __FUNCTION__ << " SUCCEEDED " << std::endl; 
    return 0; // success 
}

// -----------------------------------------------------------------------------

int test_hicsparseSpMV() {
    std::cout << "--- " << __FUNCTION__ << std::endl;

    // Create a sparse matrix
    const int rows = 3;
    const int cols = 3;
    const int nnz = 3;
    double values[nnz] = {1.0, 2.0, 3.0};
    int row_offsets[rows+1] = {0, 1, 2, 3};
    int column_indices[nnz] = {0, 1, 2};

    // Put the sparse matrix onto the device
    double* dvalues;
    int* drow_offsets;
    int* dcolumn_indices;
    HIC_CALL( hicMalloc((void**)&dvalues, nnz * sizeof(double)) );
    HIC_CALL( hicMalloc((void**)&drow_offsets, (rows+1) * sizeof(int)) );
    HIC_CALL( hicMalloc((void**)&dcolumn_indices, nnz * sizeof(int)) );
    HIC_CALL( hicMemcpy(dvalues, values, nnz * sizeof(double), hicMemcpyHostToDevice) );
    HIC_CALL( hicMemcpy(drow_offsets, row_offsets, (rows+1) * sizeof(int), hicMemcpyHostToDevice) );
    HIC_CALL( hicMemcpy(dcolumn_indices, column_indices, nnz * sizeof(int), hicMemcpyHostToDevice) );

    // Create a dense vector
    const int N = 3;
    double x[N] = {1.0, 2.0, 3.0};
    double y[N] = {0.0, 0.0, 0.0};

    // Put the dense vector onto the device
    double* dx;
    double* dy;
    HIC_CALL( hicMalloc((void**)&dx, N * sizeof(double)) );
    HIC_CALL( hicMalloc((void**)&dy, N * sizeof(double)) );
    HIC_CALL( hicMemcpy(dx, x, N * sizeof(double), hicMemcpyHostToDevice) );
    HIC_CALL( hicMemcpy(dy, y, N * sizeof(double), hicMemcpyHostToDevice) );

    // Create sparse library handle
    hicsparseHandle_t handle;
    HICSPARSE_CALL( hicsparseCreate(&handle) );

    // Create a sparse matrix descriptor
    hicsparseConstSpMatDescr_t matA;
    HICSPARSE_CALL( hicsparseCreateConstCsr(
        &matA,
        rows, cols, nnz,
        drow_offsets,
        dcolumn_indices,
        dvalues,
        HICSPARSE_INDEX_32I,
        HICSPARSE_INDEX_32I,
        HICSPARSE_INDEX_BASE_ZERO,
        HIC_R_64F) );
    
    // Create dense matrix descriptors
    hicsparseConstDnVecDescr_t vecX;
    HICSPARSE_CALL( hicsparseCreateConstDnVec(
        &vecX,
        N,
        dx,
        HIC_R_64F) );
    
    hicsparseDnVecDescr_t vecY;
    HICSPARSE_CALL( hicsparseCreateDnVec(
        &vecY,
        N,
        dy,
        HIC_R_64F) );

    // Set parameters
    const double alpha = 1.0;
    const double beta = 0.0;
    
    // Determine buffer size
    size_t bufferSize = 0;
    HICSPARSE_CALL( hicsparseSpMV_bufferSize(
        handle,
        HICSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha,
        matA,
        vecX,
        &beta,
        vecY,
        HIC_R_64F,
        HICSPARSE_SPMV_ALG_DEFAULT,
        &bufferSize) );

    // Allocate buffer
    char* buffer;
    HIC_CALL( hicMalloc(&buffer, bufferSize) );
    
    // Perform SpMV
    HICSPARSE_CALL( hicsparseSpMV(
        handle,
        HICSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha,
        matA,
        vecX,
        &beta,
        vecY,
        HIC_R_64F,
        HICSPARSE_SPMV_ALG_DEFAULT,
        buffer) );
    
    // Copy result back to host
    HIC_CALL( hicMemcpy(y, dy, N * sizeof(double), hicMemcpyDeviceToHost) );

    // Check result
    const double expected_y[N] = {1.0, 4.0, 9.0};
    for( int i = 0; i < N; ++i ) {
        if( y[i] != expected_y[i] ) {
            throw std::runtime_error("Error: y[" + std::to_string(i) + "] = " + std::to_string(y[i]) + " != " + std::to_string(expected_y[i]));
        }
    }

    // Clean up
    HIC_CALL( hicFree(dy) );
    HIC_CALL( hicFree(dx) );
    HIC_CALL( hicFree(dcolumn_indices) );
    HIC_CALL( hicFree(drow_offsets) );
    HIC_CALL( hicFree(dvalues) );
    HICSPARSE_CALL( hicsparseDestroyDnVec(vecY) );
    HICSPARSE_CALL( hicsparseDestroyDnVec(vecX) );
    HICSPARSE_CALL( hicsparseDestroySpMat(matA) );
    HICSPARSE_CALL( hicsparseDestroy(handle) );

    std::cout << "--- " << __FUNCTION__ << " SUCCEEDED " << std::endl; 
    return 0; // success 
}

// -----------------------------------------------------------------------------

int test_hicsparseSpMM() {
    std::cout << "--- " << __FUNCTION__ << std::endl;

    // Create a sparse matrix
    const int rows = 3;
    const int cols = 3;
    const int nnz = 3;
    double values[nnz] = {1.0, 2.0, 3.0};
    int row_offsets[rows + 1] = {0, 1, 2, 3};
    int column_indices[nnz] = {0, 1, 2};

    // Put the sparse matrix onto the device
    double* dvalues;
    int* drow_offsets;
    int* dcolumn_indices;
    HIC_CALL(hicMalloc((void**)&dvalues, nnz * sizeof(double)));
    HIC_CALL(hicMalloc((void**)&drow_offsets, (rows + 1) * sizeof(int)));
    HIC_CALL(hicMalloc((void**)&dcolumn_indices, nnz * sizeof(int)));
    HIC_CALL(hicMemcpy(dvalues, values, nnz * sizeof(double), hicMemcpyHostToDevice));
    HIC_CALL(hicMemcpy(drow_offsets, row_offsets, (rows + 1) * sizeof(int), hicMemcpyHostToDevice));
    HIC_CALL(hicMemcpy(dcolumn_indices, column_indices, nnz * sizeof(int), hicMemcpyHostToDevice));

    // Create dense matrices
    const int B_rows = 3;
    const int B_cols = 3;
    double B[B_rows * B_cols] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    double C[rows * B_cols] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

    // Put the dense matrices onto the device
    double* dB;
    double* dC;
    HIC_CALL(hicMalloc((void**)&dB, B_rows * B_cols * sizeof(double)));
    HIC_CALL(hicMalloc((void**)&dC, rows * B_cols * sizeof(double)));
    HIC_CALL(hicMemcpy(dB, B, B_rows * B_cols * sizeof(double), hicMemcpyHostToDevice));
    HIC_CALL(hicMemcpy(dC, C, rows * B_cols * sizeof(double), hicMemcpyHostToDevice));

    // Create sparse library handle
    hicsparseHandle_t handle;
    HICSPARSE_CALL(hicsparseCreate(&handle));

    // Create a sparse matrix descriptor
    hicsparseConstSpMatDescr_t matA;
    HICSPARSE_CALL(hicsparseCreateConstCsr(
        &matA,
        rows, cols, nnz,
        drow_offsets,
        dcolumn_indices,
        dvalues,
        HICSPARSE_INDEX_32I,
        HICSPARSE_INDEX_32I,
        HICSPARSE_INDEX_BASE_ZERO,
        HIC_R_64F));

    // Create dense matrix descriptors
    hicsparseConstDnMatDescr_t matB;
    HICSPARSE_CALL(hicsparseCreateConstDnMat(
        &matB,
        B_rows,
        B_cols,
        B_cols,
        dB,
        HIC_R_64F,
        HICSPARSE_ORDER_ROW));

    hicsparseDnMatDescr_t matC;
    HICSPARSE_CALL(hicsparseCreateDnMat(
        &matC,
        rows,
        B_cols,
        B_cols,
        dC,
        HIC_R_64F,
        HICSPARSE_ORDER_ROW));

    // Set parameters
    const double alpha = 1.0;
    const double beta = 0.0;

    // Determine buffer size
    size_t bufferSize = 0;
    HICSPARSE_CALL(hicsparseSpMM_bufferSize(
        handle,
        HICSPARSE_OPERATION_NON_TRANSPOSE,
        HICSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha,
        matA,
        matB,
        &beta,
        matC,
        HIC_R_64F,
        HICSPARSE_SPMM_ALG_DEFAULT,
        &bufferSize));

    // Allocate buffer
    char* buffer;
    HIC_CALL(hicMalloc(&buffer, bufferSize));

    // Perform SpMM
    HICSPARSE_CALL(hicsparseSpMM(
        handle,
        HICSPARSE_OPERATION_NON_TRANSPOSE,
        HICSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha,
        matA,
        matB,
        &beta,
        matC,
        HIC_R_64F,
        HICSPARSE_SPMM_ALG_DEFAULT,
        buffer));

    // Copy result back to host
    HIC_CALL(hicMemcpy(C, dC, rows * B_cols * sizeof(double), hicMemcpyDeviceToHost));

    // Check result
    const double expected_C[rows * B_cols] = {1.0, 2.0, 3.0, 8.0, 10.0, 12.0, 21.0, 24.0, 27.0};
    for (int i = 0; i < rows * B_cols; ++i) {
        if (C[i] != expected_C[i]) {
            throw std::runtime_error("Error: C[" + std::to_string(i) + "] = " + std::to_string(C[i]) + " != " + std::to_string(expected_C[i]));
        }
    }

    // Clean up
    HIC_CALL(hicFree(dC));
    HIC_CALL(hicFree(dB));
    HIC_CALL(hicFree(dcolumn_indices));
    HIC_CALL(hicFree(drow_offsets));
    HIC_CALL(hicFree(dvalues));
    HICSPARSE_CALL(hicsparseDestroyDnMat(matC));
    HICSPARSE_CALL(hicsparseDestroyDnMat(matB));
    HICSPARSE_CALL(hicsparseDestroySpMat(matA));
    HICSPARSE_CALL(hicsparseDestroy(handle));

    std::cout << "--- " << __FUNCTION__ << " SUCCEEDED " << std::endl;
    return 0; // success
}

// -----------------------------------------------------------------------------

std::vector<std::function<int()>> tests = {
    test_hicsparseCreate,
    test_hicsparseSpMV,
    test_hicsparseSpMM,
};

int main(int argc, char* argv[]) {
    int num_devices = 0;
    hicGetDeviceCount(&num_devices);
    if( num_devices == 0 ) {
        std::ignore = hicGetLastError();
        std::cout << "TEST IGNORED, hicGetDeviceCount -> 0" << std::endl; 
        return 0;
    }
    std::cout << "hicGetDeviceCount -> " << num_devices << std::endl; 
    int error = 0;
    for( auto& test: tests) {
        try {
            error += test();
        }
        catch( std::exception& e ) {
            error += 1;
            std::cout << e.what() << std::endl;
        }
    }
    return error;
}
