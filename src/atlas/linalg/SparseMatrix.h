#pragma once

#include <memory>
#include <vector>

#include "atlas/array.h"

#include "eckit/linalg/SparseMatrix.h"
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
    using iterator = eckit::linalg::SparseMatrix::iterator;
    using const_iterator = eckit::linalg::SparseMatrix::const_iterator;

public:
    // -- Constructors

    /// Default constructor, empty matrix
    SparseMatrix();

    /// Constructor from triplets
    SparseMatrix(Size rows, Size cols, const std::vector<eckit::linalg::Triplet>&);

    /// Copy constructor
    SparseMatrix(const SparseMatrix&);

    /// Assignment operator (allocates and copies data)
    SparseMatrix& operator=(const SparseMatrix&);

public:
    void swap(SparseMatrix&);

    /// @returns number of rows
    Size rows() const { return host_matrix_.rows(); }

    /// @returns number of columns
    Size cols() const { return host_matrix_.cols(); }

    /// @returns number of non-zeros
    Size nonZeros() const { return host_matrix_.nonZeros(); }

    /// @returns true if this matrix does not contain non-zero entries
    bool empty() const { return host_matrix_.empty(); }

    /// @returns footprint of the matrix in memory
    size_t footprint() const;

    /// Prune entries with exactly the given value
    SparseMatrix& prune(Scalar = 0);

    /// Transpose matrix in-place
    SparseMatrix& transpose();

public:
    void updateDevice() const { 
        outer_->updateDevice();
        inner_->updateDevice();
        value_->updateDevice();
    }

    void updateHost() const { 
        outer_->updateHost();
        inner_->updateHost();
        value_->updateHost();
    }

    bool hostNeedsUpdate() const { 
        return outer_->hostNeedsUpdate() ||
               inner_->hostNeedsUpdate() ||
               value_->hostNeedsUpdate();
    }

    bool deviceNeedsUpdate() const { 
        return outer_->deviceNeedsUpdate() ||
               inner_->deviceNeedsUpdate() ||
               value_->deviceNeedsUpdate();
    }

    void setHostNeedsUpdate(bool v) const { 
        outer_->setHostNeedsUpdate(v);
        inner_->setHostNeedsUpdate(v);
        value_->setHostNeedsUpdate(v);
    }

    void setDeviceNeedsUpdate(bool v) const {
        outer_->setDeviceNeedsUpdate(v);
        inner_->setDeviceNeedsUpdate(v);
        value_->setDeviceNeedsUpdate(v);
    }

    bool deviceAllocated() const {
        return outer_->deviceAllocated() &&
               inner_->deviceAllocated() &&
               value_->deviceAllocated();
    }

    void allocateDevice() {
        outer_->allocateDevice();
        inner_->allocateDevice();
        value_->allocateDevice();
    }

    void deallocateDevice() { 
        outer_->deallocateDevice();
        inner_->deallocateDevice();
        value_->deallocateDevice();
    }

    const eckit::linalg::SparseMatrix& host_matrix() const { return host_matrix_; }
    eckit::linalg::SparseMatrix& host_matrix() { return host_matrix_; }

    /// @returns read-only view of the value vector
    atlas::array::ArrayView<const Scalar, 1> value_view() const { return atlas::array::make_view<Scalar, 1>(*value_); }

    /// @returns read-only view of the outer index vector
    atlas::array::ArrayView<const Index, 1> outer_view() const { return atlas::array::make_view<Index, 1>(*outer_); }

    /// @returns read-only view of the inner index vector
    atlas::array::ArrayView<const Index, 1> inner_view() const { return atlas::array::make_view<Index, 1>(*inner_); }

    /// @returns read-only view of the value array
    const Scalar* value() const { return host_matrix_.data(); }

    /// @returns read-only view of the value array
    const Scalar* data() const { return host_value(); }
    
    /// @returns read-only view of the outer index array
    const Index* outer() const { return host_matrix_.outer(); }

    /// @returns read-only view of the inner index array
    const Index* inner() const { return host_matrix_.inner(); }

    /// @returns read-only view of the value array
    const Scalar* host_value() const { return host_matrix_.data(); }

    /// @returns read-only view of the value array
    const Scalar* host_data() const { return host_value(); }
    
    /// @returns read-only view of the outer index array
    const Index* host_outer() const { return host_matrix_.outer(); }

    /// @returns read-only view of the inner index array
    const Index* host_inner() const { return host_matrix_.inner(); }

    /// @returns read-only view of the value array
    const Scalar* device_value() const { return value_->device_data<Scalar>(); }

    /// @returns read-only view of the value array
    const Scalar* device_data() const { return device_value(); }
    
    /// @returns read-only view of the outer index array
    const Index* device_outer() const { return outer_->device_data<Index>(); }

    /// @returns read-only view of the inner index array
    const Index* device_inner() const { return inner_->device_data<Index>(); }

public:  // iterators
    
    /// const iterators to begin/end of row
    const_iterator begin(Size row) const { return host_matrix_.begin(row); }
    const_iterator end(Size row) const { return host_matrix_.end(row); }

    /// const iterators to begin/end of matrix
    const_iterator begin() const { return host_matrix_.begin(); }
    const_iterator end() const { return host_matrix_.end(); }

    /// iterators to begin/end of row
    iterator begin(Size row) { return host_matrix_.begin(row); }
    iterator end(Size row) { return host_matrix_.end(row); }

    /// const iterators to begin/end of matrix
    iterator begin() { return host_matrix_.begin(); }
    iterator end() { return host_matrix_.end(); }

private:
    eckit::linalg::SparseMatrix host_matrix_;
    std::unique_ptr<atlas::array::Array> outer_;
    std::unique_ptr<atlas::array::Array> inner_;
    std::unique_ptr<atlas::array::Array> value_;
};

//----------------------------------------------------------------------------------------------------------------------
}  // namespace linalg
}  // namespace atlas