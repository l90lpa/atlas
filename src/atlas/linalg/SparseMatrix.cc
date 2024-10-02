
#include "atlas/linalg/SparseMatrix.h"

#include <algorithm>
#include <limits>

#include "atlas/array/helpers/ArrayCopier.h"

#include "eckit/exception/Exceptions.h"

namespace {
    size_t countNonZeroTriplets(const std::vector<eckit::linalg::Triplet>& triplets) {
        return std::count_if(triplets.begin(), triplets.end(), [](const auto& tri) { return tri.nonZero(); });
    }
}

namespace atlas {
namespace linalg {

//----------------------------------------------------------------------------------------------------------------------

SparseMatrix::SparseMatrix() {}


SparseMatrix::SparseMatrix(Size rows, Size cols, Size nonZeros)
    : shape_{nonZeros, rows, cols} {
    resize(shape_.rows(), shape_.cols(), shape_.nonZeros());
}


SparseMatrix::SparseMatrix(Size rows, Size cols, const std::vector<eckit::linalg::Triplet>& triplets)
    : shape_{countNonZeroTriplets(triplets), rows, cols} {
    
    resize(shape_.rows(), shape_.cols(), shape_.nonZeros());

    if (auto max = static_cast<Size>(std::numeric_limits<Index>::max()); max < shape_.nonZeros()) {
        throw eckit::OutOfRange("SparseMatrix::SparseMatrix: too many non-zero entries, nnz=" + std::to_string(shape_.nonZeros())
                             + ", max=" + std::to_string(max),
                         Here());
    }

    Size pos = 0;
    Size row = 0;

    auto value_v = atlas::array::make_view<Scalar, 1>(*value_);
    auto outer_v  = atlas::array::make_view<Index, 1>(*outer_);
    auto inner_v  = atlas::array::make_view<Index, 1>(*inner_);

    outer_v(0) = 0; /* first entry (base) is always zero */

    // Build vectors of inner indices and values, update outer index per row
    for (const auto& tri : triplets) {
        if (tri.nonZero()) {

            // triplets are ordered by rows
            ASSERT(tri.row() >= row);
            ASSERT(tri.row() < shape_.rows_);
            ASSERT(tri.col() < shape_.cols_);

            // start a new row
            while (tri.row() > row) {
                outer_v(++row) = static_cast<Index>(pos);
            }

            inner_v(pos) = static_cast<Index>(tri.col());
            value_v(pos)  = tri.value();
            ++pos;
        }
    }

    while (row < shape_.rows_) {
        outer_v(++row) = static_cast<Index>(pos);
    }

    ASSERT(static_cast<Size>(outer_v(shape_.outerSize() - 1)) == nonZeros());
}


SparseMatrix::SparseMatrix(const SparseMatrix& other) {
    if (!other.empty()) {  // in case we copy an other that was constructed empty

        resize(other.rows(), other.cols(), other.nonZeros());

        // Make views of the arrays
        auto value_v = atlas::array::make_view<Scalar, 1>(*value_);
        auto outer_v  = atlas::array::make_view<Index, 1>(*outer_);
        auto inner_v  = atlas::array::make_view<Index, 1>(*inner_);

        // Make views od the other arrays
        auto other_value_v = atlas::array::make_view<Scalar, 1>(*other.value_);
        auto other_outer_v  = atlas::array::make_view<Index, 1>(*other.outer_);
        auto other_inner_v  = atlas::array::make_view<Index, 1>(*other.inner_);

        // Copy the data
        atlas::array::helpers::array_copier<Scalar, 1>::apply(value_v, other_value_v);
        atlas::array::helpers::array_copier<Index, 1>::apply(outer_v, other_outer_v);
        atlas::array::helpers::array_copier<Index, 1>::apply(inner_v, other_inner_v);
    }
}


SparseMatrix& SparseMatrix::operator=(const SparseMatrix& other) {
    SparseMatrix copy(other);
    swap(copy);
    return *this;
}


void SparseMatrix::resize(Size rows, Size cols, Size nonZeros) {
    ASSERT(nonZeros > 0);
    ASSERT(nonZeros <= rows * cols);
    ASSERT(rows > 0 && cols > 0);

    shape_ = Shape{nonZeros, rows, cols};
    value_->resize(nonZeros);
    outer_->resize(rows + 1);
    inner_->resize(nonZeros);
}


void SparseMatrix::swap(SparseMatrix& other) {
    std::swap(value_, other.value_);
    std::swap(outer_, other.outer_);
    std::swap(inner_, other.inner_);
    std::swap(shape_, other.shape_);
}


size_t SparseMatrix::footprint() const {
    return sizeof(*this) + shape_.allocSize();
}


// bool SparseMatrix::inSharedMemory() const {
//     ASSERT(owner_);
//     return owner_->inSharedMemory();
// }


// SparseMatrix& SparseMatrix::transpose() {
//     /// @note Can SparseMatrix::transpose() be done more efficiently?
//     ///       We are building another matrix and then swapping

//     std::vector<Triplet> triplets;
//     triplets.reserve(nonZeros());

//     for (Size r = 0; r < shape_.rows_; ++r) {
//         for (auto c = spm_.outer_[r]; c < spm_.outer_[r + 1]; ++c) {
//             ASSERT(spm_.inner_[c] >= 0);
//             triplets.emplace_back(static_cast<Size>(spm_.inner_[c]), r, spm_.data_[c]);
//         }
//     }

//     std::sort(triplets.begin(), triplets.end());  // triplets must be sorted by row

//     SparseMatrix tmp(shape_.cols_, shape_.rows_, triplets);

//     swap(tmp);

//     return *this;
// }


// SparseMatrix& SparseMatrix::prune(Scalar val) {
//     std::vector<Scalar> v;
//     std::vector<Index> inner;

//     Size nnz = 0;
//     for (Size r = 0; r < shape_.rows_; ++r) {
//         const auto start = spm_.outer_[r];
//         spm_.outer_[r]   = static_cast<Index>(nnz);
//         for (auto c = start; c < spm_.outer_[r + 1]; ++c) {
//             if (spm_.data_[c] != val) {
//                 v.push_back(spm_.data_[c]);
//                 inner.push_back(spm_.inner_[c]);
//                 ++nnz;
//             }
//         }
//     }
//     spm_.outer_[shape_.rows_] = static_cast<Index>(nnz);

//     SparseMatrix tmp;
//     tmp.reserve(shape_.rows_, shape_.cols_, nnz);

//     std::memcpy(tmp.spm_.data_, v.data(), nnz * sizeof(Scalar));
//     std::memcpy(tmp.spm_.outer_, spm_.outer_, shape_.outerSize() * sizeof(Index));
//     std::memcpy(tmp.spm_.inner_, inner.data(), nnz * sizeof(Index));

//     swap(tmp);

//     return *this;
// }


// SparseMatrix::const_iterator SparseMatrix::const_iterator::operator++(int) {
//     auto it = *this;
//     ++(*this);
//     return it;
// }


// bool SparseMatrix::const_iterator::operator==(const SparseMatrix::const_iterator& other) const {
//     ASSERT(other.matrix_ == matrix_);
//     return other.index_ == index_;
// }


// SparseMatrix::const_iterator::const_iterator(const SparseMatrix& matrix) :
//     matrix_(const_cast<SparseMatrix*>(&matrix)), index_(0) {
//     for (row_ = 0; matrix_->spm_.outer_[row_ + 1] == 0;) {
//         ++row_;
//     }
// }


// SparseMatrix::const_iterator::const_iterator(const SparseMatrix& matrix, Size row) :
//     matrix_(const_cast<SparseMatrix*>(&matrix)), row_(row) {
//     if (const Size rows = matrix_->rows(); row_ > rows) {
//         row_ = rows;
//     }
//     index_ = static_cast<Size>(matrix_->spm_.outer_[row_]);
// }


// Size SparseMatrix::const_iterator::col() const {
//     ASSERT(matrix_ && index_ < matrix_->nonZeros());
//     return static_cast<Size>(matrix_->inner()[index_]);
// }


// Size SparseMatrix::const_iterator::row() const {
//     return row_;
// }


// SparseMatrix::const_iterator& SparseMatrix::const_iterator::operator++() {
//     if (lastOfRow()) {
//         row_++;
//     }
//     index_++;
//     return *this;
// }


// const Scalar& SparseMatrix::const_iterator::operator*() const {
//     assert(matrix_ && index_ < matrix_->nonZeros());
//     return matrix_->data()[index_];
// }


// void SparseMatrix::const_iterator::print(std::ostream& os) const {
//     os << "SparseMatrix::iterator(row=" << row_ << ", col=" << col() << ", index=" << index_
//        << ", value=" << operator*() << ")" << std::endl;
// }


// Scalar& SparseMatrix::iterator::operator*() {
//     assert(matrix_ && index_ < matrix_->nonZeros());
//     return matrix_->spm_.data_[index_];
// }

//----------------------------------------------------------------------------------------------------------------------

}  // namespace linalg
}  // namespace atlas
