
#include "atlas/linalg/SparseMatrix.h"

#include <algorithm>
#include <cstring>
#include <limits>

#include "atlas/array/helpers/ArrayCopier.h"

#include "eckit/exception/Exceptions.h"

namespace atlas {
namespace linalg {

//----------------------------------------------------------------------------------------------------------------------

SparseMatrix::SparseMatrix() : host_matrix_() {}


SparseMatrix::SparseMatrix(Size rows, Size cols, const std::vector<eckit::linalg::Triplet>& triplets) :
    host_matrix_(rows, cols, triplets) {

    outer_.reset(atlas::array::Array::wrap<Index>(const_cast<Index*>(outer()), atlas::array::make_shape(host_matrix_.rows() + 1)));
    inner_.reset(atlas::array::Array::wrap<Index>(const_cast<Index*>(inner()), atlas::array::make_shape(host_matrix_.nonZeros())));
    value_.reset(atlas::array::Array::wrap<Scalar>(const_cast<Scalar*>(value()), atlas::array::make_shape(host_matrix_.nonZeros())));
}


SparseMatrix::SparseMatrix(const SparseMatrix& other) : host_matrix_(other.host_matrix_) {
    if (!other.empty()) {  // in case we copy an other that was constructed empty
        outer_.reset(atlas::array::Array::wrap<Index>(const_cast<Index*>(outer()), atlas::array::make_shape(host_matrix_.rows() + 1)));
        inner_.reset(atlas::array::Array::wrap<Index>(const_cast<Index*>(inner()), atlas::array::make_shape(host_matrix_.nonZeros())));
        value_.reset(atlas::array::Array::wrap<Scalar>(const_cast<Scalar*>(value()), atlas::array::make_shape(host_matrix_.nonZeros())));
    }
}


SparseMatrix& SparseMatrix::operator=(const SparseMatrix& other) {
    SparseMatrix copy(other);
    swap(copy);
    return *this;
}


void SparseMatrix::swap(SparseMatrix& other) {
    host_matrix_.swap(other.host_matrix_);
    outer_.swap(other.outer_);
    inner_.swap(other.inner_);
    value_.swap(other.value_);
}


size_t SparseMatrix::footprint() const {
    return host_matrix_.footprint() +
           outer_->footprint() +
           inner_->footprint() +
           value_->footprint();
}


SparseMatrix& SparseMatrix::prune(Scalar val) {
    host_matrix_.prune(val);
    outer_.reset(atlas::array::Array::wrap<Index>(const_cast<Index*>(outer()), atlas::array::make_shape(rows() + 1)));
    inner_.reset(atlas::array::Array::wrap<Index>(const_cast<Index*>(inner()), atlas::array::make_shape(nonZeros())));
    value_.reset(atlas::array::Array::wrap<Scalar>(const_cast<Scalar*>(value()), atlas::array::make_shape(nonZeros())));
    setDeviceNeedsUpdate(true);
    return *this;
}


SparseMatrix& SparseMatrix::transpose() {
    host_matrix_.transpose();
    outer_.reset(atlas::array::Array::wrap<Index>(const_cast<Index*>(outer()), atlas::array::make_shape(rows() + 1)));
    inner_.reset(atlas::array::Array::wrap<Index>(const_cast<Index*>(inner()), atlas::array::make_shape(nonZeros())));
    value_.reset(atlas::array::Array::wrap<Scalar>(const_cast<Scalar*>(value()), atlas::array::make_shape(nonZeros())));
    setDeviceNeedsUpdate(true);
    return *this;
}

//----------------------------------------------------------------------------------------------------------------------
}  // namespace linalg
}  // namespace atlas