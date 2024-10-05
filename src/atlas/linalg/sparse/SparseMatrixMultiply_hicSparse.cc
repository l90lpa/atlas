/*
 * (C) Copyright 2013 ECMWF.
 *
 * This software is licensed under the terms of the Apache Licence Version 2.0
 * which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
 * In applying this licence, ECMWF does not waive the privileges and immunities
 * granted to it by virtue of its status as an intergovernmental organisation
 * nor does it submit to any jurisdiction.
 */

#include "atlas/linalg/sparse/SparseMatrixMultiply_hicSparse.h"

#include "atlas/parallel/omp/omp.h"
#include "atlas/runtime/Exception.h"

#include "hic/hic.h"
#include "hic/hicsparse.h"

namespace atlas {
namespace linalg {
namespace sparse {

template <typename SourceValue, typename TargetValue>
void SparseMatrixMultiply<backend::hicsparse, Indexing::layout_left, 1, SourceValue, TargetValue>::apply(
    const SparseMatrix& W, const View<SourceValue, 1>& src, View<TargetValue, 1>& tgt, const Configuration&) {
    using Value       = TargetValue;
    const auto outer  = W.outer();
    const auto index  = W.inner();
    const auto weight = W.data();
    const idx_t rows  = static_cast<idx_t>(W.rows());

    ATLAS_ASSERT(src.shape(0) >= W.cols());
    ATLAS_ASSERT(tgt.shape(0) >= W.rows());

    atlas_omp_parallel_for(idx_t r = 0; r < rows; ++r) {
        tgt[r] = 0.;
        for (idx_t c = outer[r]; c < outer[r + 1]; ++c) {
            idx_t n = index[c];
            Value w = static_cast<Value>(weight[c]);
            tgt[r] += w * src[n];
        }
    }

    // TODO: this is a WIP and needs to be completed
    if constexpr (false) {

        // if (src.deviceNeedsUpdate()) {
        //     src.updateDevice();
        // }
        // if (tgt.deviceNeedsUpdate()) {
        //     tgt.updateDevice();
        // }
        // if (W.deviceNeedsUpdate()) {
        //     W.updateDevice();
        // }

        // auto d_src = src.data();
        // auto d_tgt = tgt.data();

        // auto d_value = W.device_value();
        // auto d_outer = W.device_outer();
        // auto d_inner = W.device_inner();

        // // Create sparse library handle
        // hicsparseHandle_t handle;
        // HICSPARSE_CALL( hicsparseCreate(&handle) );

        // // Create a sparse matrix descriptor
        // hicsparseConstSpMatDescr_t matA;
        // HICSPARSE_CALL( hicsparseCreateConstCsr(
        //     &matA,
        //     W.rows(), W.cols(), W.nonZeros(),
        //     d_outer, // row_offsets
        //     d_inner, // column_indices
        //     d_value, // values
        //     HICSPARSE_INDEX_32I,
        //     HICSPARSE_INDEX_32I,
        //     HICSPARSE_INDEX_BASE_ZERO,
        //     HIC_R_64F) );
        
        // // Create dense matrix descriptors
        // hicsparseConstDnVecDescr_t vecX;
        // HICSPARSE_CALL( hicsparseCreateConstDnVec(
        //     &vecX,
        //     src.size(),
        //     d_src,
        //     HIC_R_64F) );
        
        // hicsparseDnVecDescr_t vecY;
        // HICSPARSE_CALL( hicsparseCreateDnVec(
        //     &vecY,
        //     tgt.size(),
        //     d_tgt,
        //     HIC_R_64F) );

        // // Set parameters
        // const double alpha = 1.0;
        // const double beta = 0.0;
        
        // // Determine buffer size
        // size_t bufferSize = 0;
        // HICSPARSE_CALL( hicsparseSpMV_bufferSize(
        //     handle,
        //     HICSPARSE_OPERATION_NON_TRANSPOSE,
        //     &alpha,
        //     matA,
        //     vecX,
        //     &beta,
        //     vecY,
        //     HIC_R_64F,
        //     HICSPARSE_SPMV_ALG_DEFAULT,
        //     &bufferSize) );

        // // Allocate buffer
        // char* buffer;
        // HIC_CALL( hicMalloc(&buffer, bufferSize) );
        
        // // Perform SpMV
        // HICSPARSE_CALL( hicsparseSpMV(
        //     handle,
        //     HICSPARSE_OPERATION_NON_TRANSPOSE,
        //     &alpha,
        //     matA,
        //     vecX,
        //     &beta,
        //     vecY,
        //     HIC_R_64F,
        //     HICSPARSE_SPMV_ALG_DEFAULT,
        //     buffer) );
        
        // HIC_CALL( hicFree(buffer) );
        // HICSPARSE_CALL(hicsparseDestroyDnVec(vecY));
        // HICSPARSE_CALL(hicsparseDestroyDnVec(vecX));
        // HICSPARSE_CALL(hicsparseDestroySpMat(matA));
        // HICSPARSE_CALL(hicsparseDestroy(handle));

        // tgt.setHostNeedsUpdate(true);
    }
}


template <typename SourceValue, typename TargetValue>
void SparseMatrixMultiply<backend::hicsparse, Indexing::layout_left, 2, SourceValue, TargetValue>::apply(
    const SparseMatrix& W, const View<SourceValue, 2>& src, View<TargetValue, 2>& tgt, const Configuration&) {
    using Value       = TargetValue;
    const auto outer  = W.outer();
    const auto index  = W.inner();
    const auto weight = W.data();
    const idx_t rows  = static_cast<idx_t>(W.rows());
    const idx_t Nk    = src.shape(1);

    ATLAS_ASSERT(src.shape(0) >= W.cols());
    ATLAS_ASSERT(tgt.shape(0) >= W.rows());

    atlas_omp_parallel_for(idx_t r = 0; r < rows; ++r) {
        for (idx_t k = 0; k < Nk; ++k) {
            tgt(r, k) = 0.;
        }
        for (idx_t c = outer[r]; c < outer[r + 1]; ++c) {
            idx_t n = index[c];
            Value w = static_cast<Value>(weight[c]);
            for (idx_t k = 0; k < Nk; ++k) {
                tgt(r, k) += w * src(n, k);
            }
        }
    }

    // TODO: this is a WIP and needs to be completed
    if constexpr (false) {

        // if (src.deviceNeedsUpdate()) {
        //     src.updateDevice();
        // }
        // if (tgt.deviceNeedsUpdate()) {
        //     tgt.updateDevice();
        // }
        // if (W.deviceNeedsUpdate()) {
        //     W.updateDevice();
        // }
        
        // auto d_src = src.data();
        // auto d_tgt = tgt.data();

        // auto d_value = W.device_value();
        // auto d_outer = W.device_outer();
        // auto d_inner = W.device_inner();

        // // Create sparse library handle
        // hicsparseHandle_t handle;
        // HICSPARSE_CALL(hicsparseCreate(&handle));

        // // Create a sparse matrix descriptor
        // hicsparseConstSpMatDescr_t matA;
        // HICSPARSE_CALL(hicsparseCreateConstCsr(
        //     &matA,
        //     W.rows(), W.cols(), W.nonZeros(),
        //     d_outer, // row_offsets
        //     d_inner, // column_indices
        //     d_value,  // values
        //     HICSPARSE_INDEX_32I,
        //     HICSPARSE_INDEX_32I,
        //     HICSPARSE_INDEX_BASE_ZERO,
        //     HIC_R_64F));

        // // Create dense matrix descriptors
        // hicsparseConstDnMatDescr_t matB;
        // HICSPARSE_CALL(hicsparseCreateConstDnMat(
        //     &matB,
        //     src.shape(0),
        //     src.shape(1),
        //     B_cols,
        //     d_src,
        //     HIC_R_64F,
        //     HICSPARSE_ORDER_ROW));

        // hicsparseDnMatDescr_t matC;
        // HICSPARSE_CALL(hicsparseCreateDnMat(
        //     &matC,
        //     tgt.shape(0),
        //     tgt.shape(1),
        //     B_cols,
        //     d_tgt,
        //     HIC_R_64F,
        //     HICSPARSE_ORDER_ROW));

        // // Set parameters
        // const double alpha = 1.0;
        // const double beta = 0.0;

        // // Determine buffer size
        // size_t bufferSize = 0;
        // HICSPARSE_CALL(hicsparseSpMM_bufferSize(
        //     handle,
        //     HICSPARSE_OPERATION_NON_TRANSPOSE,
        //     HICSPARSE_OPERATION_NON_TRANSPOSE,
        //     &alpha,
        //     matA,
        //     matB,
        //     &beta,
        //     matC,
        //     HIC_R_64F,
        //     HICSPARSE_SPMM_ALG_DEFAULT,
        //     &bufferSize));

        // // Allocate buffer
        // char* buffer;
        // HIC_CALL(hicMalloc(&buffer, bufferSize));

        // // Perform SpMM
        // HICSPARSE_CALL(hicsparseSpMM(
        //     handle,
        //     HICSPARSE_OPERATION_NON_TRANSPOSE,
        //     HICSPARSE_OPERATION_NON_TRANSPOSE,
        //     &alpha,
        //     matA,
        //     matB,
        //     &beta,
        //     matC,
        //     HIC_R_64F,
        //     HICSPARSE_SPMM_ALG_DEFAULT,
        //     buffer));

        // HIC_CALL(hicFree(buffer));
        // HICSPARSE_CALL(hicsparseDestroyDnMat(matC));
        // HICSPARSE_CALL(hicsparseDestroyDnMat(matB));
        // HICSPARSE_CALL(hicsparseDestroySpMat(matA));
        // HICSPARSE_CALL(hicsparseDestroy(handle));

        // tgt.setHostNeedsUpdate(true);
    }
}

// template <typename SourceValue, typename TargetValue>
// void SparseMatrixMultiply<backend::hicsparse, Indexing::layout_left, 3, SourceValue, TargetValue>::apply(
//     const SparseMatrix& W, const View<SourceValue, 3>& src, View<TargetValue, 3>& tgt, const Configuration& config) {
//     if (src.contiguous() && tgt.contiguous()) {
//         // We can take a more optimized route by reducing rank
//         auto src_v = View<SourceValue, 2>(src.data(), array::make_shape(src.shape(0), src.stride(0)));
//         auto tgt_v = View<TargetValue, 2>(tgt.data(), array::make_shape(tgt.shape(0), tgt.stride(0)));
//         SparseMatrixMultiply<backend::hicsparse, Indexing::layout_left, 2, SourceValue, TargetValue>::apply(W, src_v,
//                                                                                                          tgt_v, config);
//         return;
//     }
//     using Value       = TargetValue;
//     const auto outer  = W.outer();
//     const auto index  = W.inner();
//     const auto weight = W.data();
//     const idx_t rows  = static_cast<idx_t>(W.rows());
//     const idx_t Nk    = src.shape(1);
//     const idx_t Nl    = src.shape(2);

//     atlas_omp_parallel_for(idx_t r = 0; r < rows; ++r) {
//         for (idx_t k = 0; k < Nk; ++k) {
//             for (idx_t l = 0; l < Nl; ++l) {
//                 tgt(r, k, l) = 0.;
//             }
//         }
//         for (idx_t c = outer[r]; c < outer[r + 1]; ++c) {
//             idx_t n       = index[c];
//             const Value w = static_cast<Value>(weight[c]);
//             for (idx_t k = 0; k < Nk; ++k) {
//                 for (idx_t l = 0; l < Nl; ++l) {
//                     tgt(r, k, l) += w * src(n, k, l);
//                 }
//             }
//         }
//     }
// }

template <typename SourceValue, typename TargetValue>
void SparseMatrixMultiply<backend::hicsparse, Indexing::layout_right, 1, SourceValue, TargetValue>::apply(
    const SparseMatrix& W, const View<SourceValue, 1>& src, View<TargetValue, 1>& tgt, const Configuration& config) {
    return SparseMatrixMultiply<backend::hicsparse, Indexing::layout_left, 1, SourceValue, TargetValue>::apply(W, src, tgt,
                                                                                                            config);
}

template <typename SourceValue, typename TargetValue>
void SparseMatrixMultiply<backend::hicsparse, Indexing::layout_right, 2, SourceValue, TargetValue>::apply(
    const SparseMatrix& W, const View<SourceValue, 2>& src, View<TargetValue, 2>& tgt, const Configuration&) {
    using Value       = TargetValue;
    const auto outer  = W.outer();
    const auto index  = W.inner();
    const auto weight = W.data();
    const idx_t rows  = static_cast<idx_t>(W.rows());
    const idx_t Nk    = src.shape(0);

    ATLAS_ASSERT(src.shape(1) >= W.cols());
    ATLAS_ASSERT(tgt.shape(1) >= W.rows());

    atlas_omp_parallel_for(idx_t r = 0; r < rows; ++r) {
        for (idx_t k = 0; k < Nk; ++k) {
            tgt(k, r) = 0.;
        }
        for (idx_t c = outer[r]; c < outer[r + 1]; ++c) {
            idx_t n = index[c];
            Value w = static_cast<Value>(weight[c]);
            for (idx_t k = 0; k < Nk; ++k) {
                tgt(k, r) += w * src(k, n);
            }
        }
    }
}

// template <typename SourceValue, typename TargetValue>
// void SparseMatrixMultiply<backend::hicsparse, Indexing::layout_right, 3, SourceValue, TargetValue>::apply(
//     const SparseMatrix& W, const View<SourceValue, 3>& src, View<TargetValue, 3>& tgt, const Configuration& config) {
//     if (src.contiguous() && tgt.contiguous()) {
//         // We can take a more optimized route by reducing rank
//         auto src_v = View<SourceValue, 2>(src.data(), array::make_shape(src.shape(0), src.stride(0)));
//         auto tgt_v = View<TargetValue, 2>(tgt.data(), array::make_shape(tgt.shape(0), tgt.stride(0)));
//         SparseMatrixMultiply<backend::hicsparse, Indexing::layout_right, 2, SourceValue, TargetValue>::apply(
//             W, src_v, tgt_v, config);
//         return;
//     }
//     using Value       = TargetValue;
//     const auto outer  = W.outer();
//     const auto index  = W.inner();
//     const auto weight = W.data();
//     const idx_t rows  = static_cast<idx_t>(W.rows());
//     const idx_t Nk    = src.shape(1);
//     const idx_t Nl    = src.shape(0);

//     atlas_omp_parallel_for(idx_t r = 0; r < rows; ++r) {
//         for (idx_t k = 0; k < Nk; ++k) {
//             for (idx_t l = 0; l < Nl; ++l) {
//                 tgt(l, k, r) = 0.;
//             }
//         }
//         for (idx_t c = outer[r]; c < outer[r + 1]; ++c) {
//             idx_t n       = index[c];
//             const Value w = static_cast<Value>(weight[c]);
//             for (idx_t k = 0; k < Nk; ++k) {
//                 for (idx_t l = 0; l < Nl; ++l) {
//                     tgt(l, k, r) += w * src(l, k, n);
//                 }
//             }
//         }
//     }
// }

#define EXPLICIT_TEMPLATE_INSTANTIATION(TYPE)                                                           \
    template struct SparseMatrixMultiply<backend::hicsparse, Indexing::layout_left, 1, TYPE const, TYPE>;  \
    template struct SparseMatrixMultiply<backend::hicsparse, Indexing::layout_left, 2, TYPE const, TYPE>;  \
    template struct SparseMatrixMultiply<backend::hicsparse, Indexing::layout_right, 1, TYPE const, TYPE>; \
    template struct SparseMatrixMultiply<backend::hicsparse, Indexing::layout_right, 2, TYPE const, TYPE>;
    // template struct SparseMatrixMultiply<backend::hicsparse, Indexing::layout_left, 3, TYPE const, TYPE>;  \
    // template struct SparseMatrixMultiply<backend::hicsparse, Indexing::layout_right, 3, TYPE const, TYPE>;

EXPLICIT_TEMPLATE_INSTANTIATION(double);
EXPLICIT_TEMPLATE_INSTANTIATION(float);

}  // namespace sparse
}  // namespace linalg
}  // namespace atlas
