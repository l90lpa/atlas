
#include "atlas/linalg/SparseMatrix.h"

#include <algorithm>
#include <cstring>
#include <limits>

#include "atlas/array/helpers/ArrayCopier.h"

#include "eckit/exception/Exceptions.h"

namespace {

    size_t countNonZeroTriplets(const std::vector<eckit::linalg::Triplet>& triplets) {
        const auto nonZeros = std::count_if(triplets.begin(), triplets.end(), [](const auto& tri) { return tri.nonZero(); });
        std::cout << "(countNonZeroTriplets) nonZeros: " << nonZeros << std::endl;
        if (nonZeros == 0) {
            throw eckit::OutOfRange("SparseMatrix::SparseMatrix: no non-zero entries in triplets", Here());
        }
        return nonZeros;
    }

}

namespace atlas {
namespace linalg {

//----------------------------------------------------------------------------------------------------------------------

void SparseMatrix::Shape::print(std::ostream& os) const {
    os << "Shape["
       << "nnz=" << size_ << ","
       << "rows=" << rows_ << ","
       << "cols=" << cols_ << "]";
}


SparseMatrix::SparseMatrix() : SparseMatrix(0, 0, 0) {}


SparseMatrix::SparseMatrix(Size rows, Size cols, Size nnz) :
    shape_(rows, cols, nnz),
    value_(new atlas::array::ArrayT<Scalar>(nnz)),
    outer_(new atlas::array::ArrayT<Index>((rows > 0) ? rows + 1 : 0)),
    inner_(new atlas::array::ArrayT<Index>(nnz)) {
    ASSERT(nnz <= rows * cols);
}


SparseMatrix::SparseMatrix(Size rows, Size cols, const std::vector<eckit::linalg::Triplet>& triplets) :
    SparseMatrix(rows, cols, countNonZeroTriplets(triplets)) {

    // Count number of non-zeros, allocate memory 1 triplet per non-zero
    Size nnz = nonZeros();

    if (auto max = static_cast<Size>(std::numeric_limits<Index>::max()); max < nnz) {
        throw eckit::OutOfRange("SparseMatrix::SparseMatrix: too many non-zero entries, nnz=" + std::to_string(nnz)
                             + ", max=" + std::to_string(max),
                         Here());
    }

    resize(rows, cols, nnz);

    Size pos = 0;
    Size row = 0;

    auto value_v = atlas::array::make_view<Scalar, 1>(*value_);
    auto outer_v = atlas::array::make_view<Index, 1>(*outer_);
    auto inner_v = atlas::array::make_view<Index, 1>(*inner_);

    outer_v[0] = 0; /* first entry (base) is always zero */

    // Build vectors of inner indices and values, update outer index per row
    for (const auto& tri : triplets) {
        if (tri.nonZero()) {

            // triplets are ordered by rows
            ASSERT(tri.row() >= row);
            ASSERT(tri.row() < shape_.rows_);
            ASSERT(tri.col() < shape_.cols_);

            // start a new row
            while (tri.row() > row) {
                outer_v[++row] = static_cast<Index>(pos);
            }

            inner_v[pos] = static_cast<Index>(tri.col());
            value_v[pos]  = tri.value();
            ++pos;
        }
    }

    while (row < shape_.rows_) {
        outer_v[++row] = static_cast<Index>(pos);
    }

    ASSERT(static_cast<Size>(outer_v[shape_.outerSize() - 1]) == nonZeros());
}


SparseMatrix::SparseMatrix(const SparseMatrix& other) :
    SparseMatrix(other.rows(), other.cols(), other.nonZeros()) {

    if (!other.empty()) {  // in case we copy an other that was constructed empty
        auto value_v = atlas::array::make_view<Scalar, 1>(*value_);
        auto other_value_v = atlas::array::make_view<Scalar, 1>(*other.value_);
        atlas::array::helpers::array_copier<Scalar, 1>::apply(other_value_v, value_v);
        
        auto outer_v = atlas::array::make_view<Index, 1>(*outer_);
        auto other_outer_v = atlas::array::make_view<Index, 1>(*other.outer_);
        atlas::array::helpers::array_copier<Index, 1>::apply(other_outer_v, outer_v);

        auto inner_v = atlas::array::make_view<Index, 1>(*inner_);
        auto other_inner_v = atlas::array::make_view<Index, 1>(*other.inner_);
        atlas::array::helpers::array_copier<Index, 1>::apply(other_inner_v, inner_v);
    }
}


SparseMatrix& SparseMatrix::operator=(const SparseMatrix& other) {
    SparseMatrix copy(other);
    swap(copy);
    return *this;
}


void SparseMatrix::resize(Size rows, Size cols, Size nnz) {
    ASSERT(nnz > 0);
    ASSERT(nnz <= rows * cols);
    ASSERT(rows > 0 && cols > 0);

    shape_ = Shape(rows, cols, nnz);
    value_.reset(new atlas::array::ArrayT<Scalar>(nnz));
    outer_.reset(new atlas::array::ArrayT<Index>((rows > 0) ? rows + 1 : 0));
    inner_.reset(new atlas::array::ArrayT<Index>(nnz));
}


void SparseMatrix::swap(SparseMatrix& other) {
    std::swap(shape_, other.shape_);
    std::swap(value_, other.value_);
    std::swap(outer_, other.outer_);
    std::swap(inner_, other.inner_);
}


size_t SparseMatrix::footprint() const {
    return sizeof(*this) + shape_.allocSize();
}


SparseMatrix& SparseMatrix::transpose() {
    /// @note Can SparseMatrix::transpose() be done more efficiently?
    ///       We are building another matrix and then swapping

    std::vector<eckit::linalg::Triplet> triplets;
    triplets.reserve(nonZeros());

    auto value_v = value_view();
    auto outer_v = outer_view();
    auto inner_v = inner_view();

    for (Size r = 0; r < shape_.rows_; ++r) {
        for (auto c = outer_v[r]; c < outer_v[r + 1]; ++c) {
            ASSERT(inner_v[c] >= 0);
            triplets.emplace_back(static_cast<Size>(inner_v[c]), r, value_v[c]);
        }
    }

    std::sort(triplets.begin(), triplets.end());  // triplets must be sorted by row

    SparseMatrix tmp(shape_.cols_, shape_.rows_, triplets);

    swap(tmp);

    return *this;
}


SparseMatrix& SparseMatrix::prune(Scalar val) {
    std::vector<Scalar> new_value;
    std::vector<Index> new_inner;

    auto value_v = atlas::array::make_view<Scalar, 1>(*value_);
    auto outer_v = atlas::array::make_view<Index, 1>(*outer_);
    auto inner_v = atlas::array::make_view<Index, 1>(*inner_);

    Size nnz = 0;
    for (Size r = 0; r < shape_.rows_; ++r) {
        const auto start = outer_v[r];
        outer_v[r]   = static_cast<Index>(nnz);
        for (auto c = start; c < outer_v[r + 1]; ++c) {
            if (value_v[c] != val) {
                new_value.push_back(value_v[c]);
                new_inner.push_back(inner_v[c]);
                ++nnz;
            }
        }
    }
    outer_v[shape_.rows_] = static_cast<Index>(nnz);

    SparseMatrix tmp(shape_.rows_, shape_.cols_, nnz);

    auto tmp_value_v = atlas::array::make_view<Scalar, 1>(*tmp.value_);
    auto tmp_outer_v = atlas::array::make_view<Index, 1>(*tmp.outer_);
    auto tmp_inner_v = atlas::array::make_view<Index, 1>(*tmp.inner_);

    auto new_value_v = atlas::array::ArrayView<Scalar, 1>(
        new_value.data(),
        atlas::array::make_shape(new_value.size()),
        atlas::array::make_strides(1));

    auto new_inner_v = atlas::array::ArrayView<Index, 1>(
        new_inner.data(),
        atlas::array::make_shape(new_inner.size()),
        atlas::array::make_strides(1));

    atlas::array::helpers::array_copier<Scalar, 1>::apply(new_value_v, tmp_value_v);
    atlas::array::helpers::array_copier<Index, 1>::apply(outer_v, tmp_outer_v);
    atlas::array::helpers::array_copier<Index, 1>::apply(new_inner_v, tmp_inner_v);

    swap(tmp);

    return *this;
}


SparseMatrix::const_iterator SparseMatrix::const_iterator::operator++(int) {
    auto it = *this;
    ++(*this);
    return it;
}


bool SparseMatrix::const_iterator::operator==(const SparseMatrix::const_iterator& other) const {
    ASSERT(other.matrix_ == matrix_);
    return other.index_ == index_;
}


SparseMatrix::const_iterator::const_iterator(const SparseMatrix& matrix) :
    matrix_(const_cast<SparseMatrix*>(&matrix)), index_(0) {
    for (row_ = 0; matrix_->outer()[row_ + 1] == 0;) {
        ++row_;
    }
}


SparseMatrix::const_iterator::const_iterator(const SparseMatrix& matrix, Size row) :
    matrix_(const_cast<SparseMatrix*>(&matrix)), row_(row) {
    if (const Size rows = matrix_->rows(); row_ > rows) {
        row_ = rows;
    }
    index_ = static_cast<Size>(matrix_->outer()[row_]);
}


SparseMatrix::Size SparseMatrix::const_iterator::col() const {
    ASSERT(matrix_ && index_ < matrix_->nonZeros());
    return static_cast<Size>(matrix_->inner()[index_]);
}


SparseMatrix::Size SparseMatrix::const_iterator::row() const {
    return row_;
}


SparseMatrix::const_iterator& SparseMatrix::const_iterator::operator++() {
    if (lastOfRow()) {
        row_++;
    }
    index_++;
    return *this;
}


const SparseMatrix::Scalar& SparseMatrix::const_iterator::operator*() const {
    ASSERT(matrix_ && index_ < matrix_->nonZeros());
    return matrix_->value()[index_];
}


void SparseMatrix::const_iterator::print(std::ostream& os) const {
    os << "SparseMatrix::iterator(row=" << row_ << ", col=" << col() << ", index=" << index_
       << ", value=" << operator*() << ")" << std::endl;
}


SparseMatrix::Scalar& SparseMatrix::iterator::operator*() {
    ASSERT(matrix_ && index_ < matrix_->nonZeros());
    auto value_v = atlas::array::make_view<Scalar, 1>(*(matrix_->value_));
    return value_v[index_];
}

//----------------------------------------------------------------------------------------------------------------------
}  // namespace linalg
}  // namespace atlas
