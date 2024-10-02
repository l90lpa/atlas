#pragma once

#include <memory>
#include <vector>

#include "atlas/array.h"

#include "eckit/linalg/Triplet.h"
#include "eckit/linalg/types.h"

namespace atlas {
namespace linalg {

//----------------------------------------------------------------------------------------------------------------------

/// Sparse matrix in CRS (compressed row storage) format
class SparseMatrix {
public:
    using Scalar = eckit::linalg::Scalar;
    using Index = eckit::linalg::Index;
    using Size = eckit::linalg::Size;

    struct Shape {
        Shape() = default;
        Shape(Size rows, Size cols, Size nonZeros) : size_(nonZeros), rows_(rows), cols_(cols) {}

        void reset() {
            size_ = 0;
            rows_ = 0;
            cols_ = 0;
        }

        /// @returns number of rows
        Size rows() const { return rows_; }

        /// @returns number of columns
        Size cols() const { return cols_; }

        /// @returns number of non-zeros
        Size nonZeros() const { return size_; }

        /// @returns number of non-zeros
        Size dataSize() const { return nonZeros(); }

        /// @returns number of non-zeros
        Size innerSize() const { return nonZeros(); }

        /// @returns outer size is number of rows + 1
        Size outerSize() const { return rows_ + 1; }

        size_t allocSize() const { return sizeofData() + sizeofOuter() + sizeofInner(); }

        size_t sizeofData() const { return dataSize() * sizeof(Scalar); }
        size_t sizeofOuter() const { return outerSize() * sizeof(Index); }
        size_t sizeofInner() const { return innerSize() * sizeof(Index); }

        Size size_ = 0;  ///< Size of the container (AKA number of non-zeros nnz)
        Size rows_ = 0;  ///< Number of rows
        Size cols_ = 0;  ///< Number of columns
    };

public:
    // -- Constructors

    /// Default constructor, empty matrix
    SparseMatrix();

    /// Constructs an identity matrix with provided dimensions
    SparseMatrix(Size rows, Size cols, Size nonZeros);

    /// Copy constructor
    SparseMatrix(const SparseMatrix&);

    // /// Move constructor
    // SparseMatrix(SparseMatrix&&);

    /// Constructor from triplets
    SparseMatrix(Size rows, Size cols, const std::vector<eckit::linalg::Triplet>&);

    /// Assignment operator (allocates and copies data)
    SparseMatrix& operator=(const SparseMatrix&);

    // /// Assignment operator (moves data)
    // SparseMatrix& operator=(SparseMatrix&&);

public:
    void swap(SparseMatrix&);

    /// @returns number of rows
    Size rows() const { return shape_.rows_; }

    /// @returns number of columns
    Size cols() const { return shape_.cols_; }

    /// @returns number of non-zeros
    Size nonZeros() const { return shape_.size_; }

    /// @returns true if this matrix does not contain non-zero entries
    bool empty() const { return nonZeros() == 0; }

    /// @returns read-only view of the value vector
    atlas::array::ArrayView<const Scalar, 1> value_view() const { return atlas::array::make_view<Scalar, 1>(*value_); }
    
    /// @returns read-only view of the outer index vector
    atlas::array::ArrayView<const Index, 1> outer_view() const { return atlas::array::make_view<Index, 1>(*outer_); }

    /// @returns read-only view of the inner index vector
    atlas::array::ArrayView<const Index, 1> inner_view() const { return atlas::array::make_view<Index, 1>(*inner_); }

    /// @returns read-only view of the data vector
    const Scalar* value() const { return value_view().data(); }
    
    /// @returns read-only view of the outer index vector
    const Index* outer() const { return outer_view().data(); }

    /// @returns read-only view of the inner index vector
    const Index* inner() const { return inner_view().data(); }

    /// Resize memory for given number of non-zeros (invalidates all data arrays)
    void resize(Size rows, Size cols, Size nnz);

    /// @returns footprint of the matrix in memory
    size_t footprint() const;

    // /// @returns if allocation is in shared memory
    // bool inSharedMemory() const;

// public:  // iterators
//     struct iterator;

//     struct const_iterator {
//         const_iterator(const SparseMatrix&);
//         const_iterator(const SparseMatrix&, Size row);

//         const_iterator(const const_iterator&) = default;
//         const_iterator(const_iterator&&)      = default;

//         virtual ~const_iterator() = default;

//         Size col() const;
//         Size row() const;

//         operator bool() const { return matrix_ && (index_ < matrix_->nonZeros()); }

//         const_iterator& operator++();
//         const_iterator operator++(int);

//         const_iterator& operator=(const const_iterator&) = default;
//         const_iterator& operator=(const_iterator&&)      = default;

//         bool operator!=(const const_iterator& other) const { return !operator==(other); }
//         bool operator==(const const_iterator&) const;

//         const Scalar& operator*() const;

//         void print(std::ostream&) const;

//         bool lastOfRow() const { return ((index_ + 1) == static_cast<Size>(matrix_->spm_.outer_[row_ + 1])); }

//     private:
//         friend struct iterator;

//         SparseMatrix* matrix_;
//         Size index_;
//         Size row_;
//     };

//     struct iterator final : const_iterator {
//         using const_iterator::const_iterator;
//         Scalar& operator*();
//     };

//     /// const iterators to begin/end of row
//     const_iterator begin(Size row) const { return {*this, row}; }
//     const_iterator end(Size row) const { return {*this, row + 1}; }

//     /// const iterators to begin/end of matrix
//     const_iterator begin() const { return {*this}; }
//     const_iterator end() const { return {*this, rows()}; }

//     /// iterators to begin/end of row
//     iterator begin(Size row) { return {*this, row}; }
//     iterator end(Size row) { return {*this, row + 1}; }

//     /// const iterators to begin/end of matrix
//     iterator begin() { return {*this}; }
//     iterator end() { return {*this, rows()}; }

private:
    std::unique_ptr<atlas::array::Array> value_;
    std::unique_ptr<atlas::array::Array> outer_;
    std::unique_ptr<atlas::array::Array> inner_;

    Shape shape_;  ///< Matrix shape
};


//----------------------------------------------------------------------------------------------------------------------
}  // namespace linalg
}  // namespace atlas
