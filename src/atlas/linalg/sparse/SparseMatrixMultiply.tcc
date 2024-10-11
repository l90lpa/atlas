/*
 * (C) Copyright 2013 ECMWF.
 *
 * This software is licensed under the terms of the Apache Licence Version 2.0
 * which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
 * In applying this licence, ECMWF does not waive the privileges and immunities
 * granted to it by virtue of its status as an intergovernmental organisation
 * nor does it submit to any jurisdiction.
 */

#pragma once

#include "SparseMatrixMultiply.h"

#include "atlas/linalg/Indexing.h"
#include "atlas/linalg/Introspection.h"
#include "atlas/linalg/View.h"
#include "atlas/linalg/sparse/Backend.h"
#include "atlas/runtime/Exception.h"

// #if ATLAS_ECKIT_HAVE_ECKIT_585
// #include "eckit/linalg/LinearAlgebraSparse.h"
// #else
// #include "eckit/linalg/LinearAlgebra.h"
// #endif


namespace atlas {
namespace linalg {

namespace sparse {
namespace {

template <typename Backend, typename Matrix, typename SourceView, typename TargetView>
void dispatch_sparse_matrix_multiply( const Matrix& matrix, const SourceView& src, TargetView& tgt, Indexing indexing,
                                      const eckit::Configuration& config ) {
    auto src_v = make_view( src );
    auto tgt_v = make_view( tgt );

    using SourceValue = const typename std::remove_const<typename decltype(src_v)::value_type>::type;
    using TargetValue = typename std::remove_const<typename decltype(tgt_v)::value_type>::type;
    constexpr int src_rank = introspection::rank<SourceView>();
    constexpr int tgt_rank = introspection::rank<TargetView>();
    static_assert( src_rank == tgt_rank, "src and tgt need same rank" );

    if ( introspection::layout_right( src ) || introspection::layout_right( tgt ) ) {
        ATLAS_ASSERT( introspection::layout_right( src ) && introspection::layout_right( tgt ) );
        // Override layout with known layout given by introspection
        using SPMM = SparseMatrixMultiply<Backend, linalg::Indexing::layout_right, src_rank, SourceValue, TargetValue>;
        SPMM::multiply( matrix, src_v, tgt_v, config );
    }
    else {
        if( indexing == Indexing::layout_left ) {
            using SPMM = SparseMatrixMultiply<Backend, linalg::Indexing::layout_left, src_rank, SourceValue, TargetValue>;
            SPMM::multiply( matrix, src_v, tgt_v, config );
        }
        else if( indexing == Indexing::layout_right ) {
            using SPMM = SparseMatrixMultiply<Backend, linalg::Indexing::layout_right, src_rank, SourceValue, TargetValue>;
            SPMM::multiply( matrix, src_v, tgt_v, config );
        }
        else {
            throw_NotImplemented( "indexing not implemented", Here() );
        }
    }
}

template <typename Backend, typename Matrix, typename SourceView, typename TargetView>
void dispatch_sparse_matrix_multiply_add( const Matrix& matrix, const SourceView& src, TargetView& tgt, Indexing indexing,
                                      const eckit::Configuration& config ) {
    auto src_v = make_view( src );
    auto tgt_v = make_view( tgt );

    using SourceValue = const typename std::remove_const<typename decltype(src_v)::value_type>::type;
    using TargetValue = typename std::remove_const<typename decltype(tgt_v)::value_type>::type;
    constexpr int src_rank = introspection::rank<SourceView>();
    constexpr int tgt_rank = introspection::rank<TargetView>();
    static_assert( src_rank == tgt_rank, "src and tgt need same rank" );

    if ( introspection::layout_right( src ) || introspection::layout_right( tgt ) ) {
        ATLAS_ASSERT( introspection::layout_right( src ) && introspection::layout_right( tgt ) );
        // Override layout with known layout given by introspection
        using SPMM = SparseMatrixMultiply<Backend, linalg::Indexing::layout_right, src_rank, SourceValue, TargetValue>;
        SPMM::multiplyAdd( matrix, src_v, tgt_v, config );
    }
    else {
        if( indexing == Indexing::layout_left ) {
            using SPMM = SparseMatrixMultiply<Backend, linalg::Indexing::layout_left, src_rank, SourceValue, TargetValue>;
            SPMM::multiplyAdd( matrix, src_v, tgt_v, config );
        }
        else if( indexing == Indexing::layout_right ) {
            using SPMM = SparseMatrixMultiply<Backend, linalg::Indexing::layout_right, src_rank, SourceValue, TargetValue>;
            SPMM::multiplyAdd( matrix, src_v, tgt_v, config );
        }
        else {
            throw_NotImplemented( "indexing not implemented", Here() );
        }
    }
}
}
}

template <typename Matrix, typename SourceView, typename TargetView>
void sparse_matrix_multiply_add( const Matrix& matrix, const SourceView& src, TargetView& tgt, Indexing indexing,
                             const eckit::Configuration& config ) {
    std::string type = config.getString( "type", sparse::current_backend() );
    if ( type == sparse::backend::openmp::type() ) {
        sparse::dispatch_sparse_matrix_multiply_add<sparse::backend::openmp>( matrix, src, tgt, indexing, config );
    } else if ( type == sparse::backend::hicsparse::type() ) {
        sparse::dispatch_sparse_matrix_multiply_add<sparse::backend::hicsparse>( matrix, src, tgt, indexing, config );
    } else {
        throw_NotImplemented( "sparse_matrix_multiply_add cannot be performed with unsupported backend [" + type + "]",
                              Here() );
    }
}

template <typename Matrix, typename SourceView, typename TargetView>
void sparse_matrix_multiply_add( const Matrix& matrix, const SourceView& src, TargetView& tgt, const eckit::Configuration& config ) {
    sparse_matrix_multiply_add( matrix, src, tgt, Indexing::layout_left, config );
}

template <typename Matrix, typename SourceView, typename TargetView>
void sparse_matrix_multiply_add( const Matrix& matrix, const SourceView& src, TargetView& tgt, Indexing indexing ) {
    sparse_matrix_multiply_add( matrix, src, tgt, indexing, sparse::Backend() );
}

template <typename Matrix, typename SourceView, typename TargetView>
void sparse_matrix_multiply_add( const Matrix& matrix, const SourceView& src, TargetView& tgt ) {
    sparse_matrix_multiply_add( matrix, src, tgt, Indexing::layout_left, sparse::Backend() );
}

template <typename Matrix, typename SourceView, typename TargetView>
void sparse_matrix_multiply( const Matrix& matrix, const SourceView& src, TargetView& tgt, Indexing indexing,
                             const eckit::Configuration& config ) {
    std::string type = config.getString( "type", sparse::current_backend() );
    if ( type == sparse::backend::openmp::type() ) {
        sparse::dispatch_sparse_matrix_multiply<sparse::backend::openmp>( matrix, src, tgt, indexing, config );
    } else if ( type == sparse::backend::hicsparse::type() ) {
        sparse::dispatch_sparse_matrix_multiply<sparse::backend::hicsparse>( matrix, src, tgt, indexing, config );
    } else {
        throw_NotImplemented( "sparse_matrix_multiply cannot be performed with unsupported backend [" + type + "]",
                              Here() );
    }
}

template <typename Matrix, typename SourceView, typename TargetView>
void sparse_matrix_multiply( const Matrix& matrix, const SourceView& src, TargetView& tgt, const eckit::Configuration& config ) {
    sparse_matrix_multiply( matrix, src, tgt, Indexing::layout_left, config );
}

template <typename Matrix, typename SourceView, typename TargetView>
void sparse_matrix_multiply( const Matrix& matrix, const SourceView& src, TargetView& tgt, Indexing indexing ) {
    sparse_matrix_multiply( matrix, src, tgt, indexing, sparse::Backend() );
}

template <typename Matrix, typename SourceView, typename TargetView>
void sparse_matrix_multiply( const Matrix& matrix, const SourceView& src, TargetView& tgt ) {
    sparse_matrix_multiply( matrix, src, tgt, Indexing::layout_left, sparse::Backend() );
}

}  // namespace linalg
}  // namespace atlas

#undef ATLAS_ENABLE_IF
