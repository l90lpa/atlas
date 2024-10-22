/*
 * (C) Copyright 1996- ECMWF.
 *
 * This software is licensed under the terms of the Apache Licence Version 2.0
 * which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
 * In applying this licence, ECMWF does not waive the privileges and immunities
 * granted to it by virtue of its status as an intergovernmental organisation
 * nor does it submit to any jurisdiction.
 */

#include <tuple>
#include <vector>

#include "eckit/linalg/Matrix.h"
#include "eckit/linalg/Vector.h"

#include "atlas/array.h"
#include "atlas/linalg/sparse.h"

#include "tests/AtlasTestEnvironment.h"


using namespace atlas::linalg;

namespace atlas {
namespace test {

//----------------------------------------------------------------------------------------------------------------------

// strings to be used in the tests
static std::string eckit_linalg = sparse::backend::eckit_linalg::type();
static std::string openmp       = sparse::backend::openmp::type();
static std::string hicsparse    = sparse::backend::hicsparse::type();

//----------------------------------------------------------------------------------------------------------------------

// Only reason to define these derived classes is for nicer constructors and convenience in the tests

class Vector : public eckit::linalg::Vector {
public:
    using Scalar = eckit::linalg::Scalar;
    using eckit::linalg::Vector::Vector;
    Vector(const std::initializer_list<Scalar>& v): eckit::linalg::Vector::Vector(v.size()) {
        size_t i = 0;
        for (auto& s : v) {
            operator[](i++) = s;
        }
    }
};

class Matrix : public eckit::linalg::Matrix {
public:
    using Scalar = eckit::linalg::Scalar;
    using eckit::linalg::Matrix::Matrix;

    Matrix(const std::initializer_list<std::vector<Scalar>>& m):
        eckit::linalg::Matrix::Matrix(m.size(), m.size() ? m.begin()->size() : 0) {
        size_t r = 0;
        for (auto& row : m) {
            for (size_t c = 0; c < cols(); ++c) {
                operator()(r, c) = row[c];
            }
            ++r;
        }
    }

    Matrix(const std::vector<std::vector<Scalar>>& m):
        eckit::linalg::Matrix::Matrix(m.size(), m.size() ? m.begin()->size() : 0) {
        size_t r = 0;
        for (auto& row : m) {
            for (size_t c = 0; c < cols(); ++c) {
                operator()(r, c) = row[c];
            }
            ++r;
        }
    }
};

template<typename Value, int Rank>
atlas::array::ArrayView<Value, Rank> make_device_synced_view(atlas::array::ArrayT<Value>& array) {
    array.updateDevice();
    return array::make_device_view<Value, Rank>(array);
}

// 2D array constructable from eckit::linalg::Matrix
// Indexing/memorylayout and data type can be customized for testing
template <typename Value, Indexing indexing = Indexing::layout_left>
struct ArrayMatrix {
    array::ArrayView<Value, 2> view() {
        array.syncHostDevice();
        return array::make_view<Value, 2>(array);
    }
    array::ArrayView<Value, 2> device_view() {
        array.syncHostDevice();
        return array::make_device_view<Value, 2>(array);
    }
    void setHostNeedsUpdate(bool b) {
        array.setHostNeedsUpdate(b);
    }
    ArrayMatrix(const eckit::linalg::Matrix& m): ArrayMatrix(m.rows(), m.cols()) {
        auto view_ = array::make_view<Value, 2>(array);
        for (int r = 0; r < m.rows(); ++r) {
            for (int c = 0; c < m.cols(); ++c) {
                auto& v = layout_left ? view_(r, c) : view_(c, r);
                v       = m(r, c);
            }
        }
    }
    ArrayMatrix(int r, int c): array(make_shape(r, c)) {}

private:
    static constexpr bool layout_left = (indexing == Indexing::layout_left);
    static array::ArrayShape make_shape(int rows, int cols) {
        return layout_left ? array::make_shape(rows, cols) : array::make_shape(cols, rows);
    }
    array::ArrayT<Value> array;
};

// 1D array constructable from eckit::linalg::Vector
template <typename Value>
struct ArrayVector {
    array::ArrayView<Value, 1> view() {
        if (array.hostNeedsUpdate()) {
            array.syncHostDevice();
        }
        return array::make_view<Value, 1>(array);
    }
    array::ArrayView<const Value, 1> const_view() {
        if (array.hostNeedsUpdate()) {
            array.syncHostDevice();
        }
        return array::make_view<const Value, 1>(array);
    }
    array::ArrayView<Value, 1> device_view() {
        array.syncHostDevice();
        return array::make_device_view<Value, 1>(array);
    }
    void setHostNeedsUpdate(bool b) {
        array.setHostNeedsUpdate(b);
    }
    ArrayVector(const eckit::linalg::Vector& v): ArrayVector(v.size()) {
        auto view_ = array::make_view<Value, 1>(array);
        for (int n = 0; n < v.size(); ++n) {
            view_[n] = v[n];
        }
    }
    ArrayVector(int size) : array(size) {}

private:
    array::ArrayT<Value> array;
};

SparseMatrix Identity(SparseMatrix::Size rows, SparseMatrix::Size cols) {
    ASSERT(rows > 0 && cols > 0);

    const auto nnz = std::min(rows, cols);

    std::vector<eckit::linalg::Triplet> triplets;
    triplets.reserve(nnz);

    for (SparseMatrix::Size i = 0; i < nnz; ++i) {
        triplets.emplace_back(i, i, 1);
    }

    return SparseMatrix(rows, cols, triplets);
}

template<typename T>
atlas::array::ArrayView<const T, 1> make_view(const std::vector<T>& v) {
    return atlas::array::ArrayView<const T, 1>(v.data(), atlas::array::make_shape(v.size()), atlas::array::make_strides(1));
}

template<typename T>
bool operator==(const atlas::array::ArrayView<T, 1>& x, const atlas::array::ArrayView<T, 1>& y) {
    if (x.size() != y.size()) {
        return false;
    }

    for (atlas::idx_t i = 0; i < x.size(); ++i) {
        if (x[i] != y[i]) {
            return false;
        }
    }

    return true;
}

template<typename T>
bool operator!=(const atlas::array::ArrayView<T, 1>& x, const atlas::array::ArrayView<T, 1>& y) {
    return !(x == y);
}

bool operator==(const SparseMatrix& A, const SparseMatrix& B) {
    if (A.rows() != B.rows() || A.cols() != B.cols() || A.nonZeros() != B.nonZeros()) {
        return false;
    }

    const auto A_value_v = A.value_view();
    const auto A_outer_v = A.outer_view();
    const auto A_inner_v = A.inner_view();

    const auto B_value_v = B.value_view();
    const auto B_outer_v = B.outer_view();
    const auto B_inner_v = B.inner_view();

    if (A_value_v != B_value_v || A_outer_v != B_outer_v || A_inner_v != B_inner_v) {
        return false;
    }

    return true;
}

bool operator!=(const SparseMatrix& A, const SparseMatrix& B) {
    return !(A == B);
}


//----------------------------------------------------------------------------------------------------------------------

template <typename T>
void expect_equal(T* v, T* r, size_t s) {
    EXPECT(is_approximately_equal(eckit::testing::make_view(v, s), eckit::testing::make_view(r, s), T(1.e-5)));
}

template <class T1, class T2>
void expect_equal(const T1& v, const T2& r) {
    expect_equal(v.data(), r.data(), std::min(v.size(), r.size()));
}

//----------------------------------------------------------------------------------------------------------------------

CASE("test introspection") {
    SECTION("ArrayView") {
        array::ArrayT<float> array(4, 3);
        auto view  = array::make_view<const float, 2>(array);
        using View = decltype(view);
        static_assert(introspection::has_contiguous<View>::value, "ArrayView expected to have contiguous");
        static_assert(introspection::has_rank<View>::value, "ArrayView expected to have rank");
        static_assert(introspection::has_shape<View>::value, "ArrayView expected to have shape");
        EXPECT(introspection::contiguous(view));
        EXPECT_EQ(introspection::shape<0>(view), 4);
        EXPECT_EQ(introspection::shape<1>(view), 3);
        EXPECT_EQ(view.stride<0>(), 3);
        EXPECT_EQ(view.stride<1>(), 1);
    }
    SECTION("std::vector") {
        using View = std::vector<double>;
        static_assert(not introspection::has_contiguous<View>::value, "std::vector does not have contiguous");
        static_assert(not introspection::has_rank<View>::value, "std::vector does not have rank");
        static_assert(not introspection::has_shape<View>::value, "std::vector does not have shape");
        static_assert(introspection::rank<View>() == 1, "std::vector is of rank 1");
        auto v = View{1, 2, 3, 4};
        EXPECT(introspection::contiguous(v));
        EXPECT_EQ(introspection::shape<0>(v), 4);
        EXPECT_EQ(introspection::shape<1>(v), 4);
    }
    SECTION("eckit::linlag::Vector") {
        using View = Vector;
        static_assert(not introspection::has_contiguous<View>::value, "eckit::linalg::Vector does not have contiguous");
        static_assert(not introspection::has_rank<View>::value, "eckit::linalg::Vector does not have rank");
        static_assert(not introspection::has_shape<View>::value, "seckit::linalg::Vector does not have shape");
        static_assert(introspection::rank<View>() == 1, "eckit::linalg::Vector is of rank 1");
        auto v = Vector{1, 2, 3, 4};
        EXPECT(introspection::contiguous(v));
        EXPECT_EQ(introspection::shape<0>(v), 4);
    }
    SECTION("eckit::linlag::Matrix") {
        using View = Matrix;
        static_assert(not introspection::has_contiguous<View>::value, "eckit::linalg::Matrix does not have contiguous");
        static_assert(not introspection::has_rank<View>::value, "eckit::linalg::Matrix does not have rank");
        static_assert(not introspection::has_shape<View>::value, "seckit::linalg::Matrix does not have shape");
        static_assert(introspection::rank<View>() == 2, "eckit::linalg::Matrix is of rank 1");
        auto m = Matrix{{1., 2.}, {3., 4.}, {5., 6.}};
        EXPECT(introspection::contiguous(m));
        // Following is reversed because M has column-major ordering
        EXPECT_EQ(introspection::shape<0>(m), 3);
        EXPECT_EQ(introspection::shape<1>(m), 2);
    }
}

//----------------------------------------------------------------------------------------------------------------------

CASE("test backend functionalities") {
    sparse::current_backend(openmp);
    EXPECT_EQ(sparse::current_backend().type(), openmp);
    EXPECT_EQ(sparse::current_backend().getString("backend", "undefined"), "undefined");

    sparse::current_backend(eckit_linalg);
    EXPECT_EQ(sparse::current_backend().type(), "eckit_linalg");
    EXPECT_EQ(sparse::current_backend().getString("backend", "undefined"), "undefined");
    
    sparse::current_backend().set("backend", "default");
    EXPECT_EQ(sparse::current_backend().getString("backend"), "default");

    sparse::current_backend(openmp);
    EXPECT_EQ(sparse::current_backend().getString("backend", "undefined"), "undefined");
    EXPECT_EQ(sparse::default_backend(eckit_linalg).getString("backend"), "default");

    sparse::current_backend(hicsparse);
    EXPECT_EQ(sparse::current_backend().type(), "hicsparse");
    EXPECT_EQ(sparse::current_backend().getString("backend", "undefined"), "undefined");

    sparse::default_backend(eckit_linalg).set("backend", "generic");
    EXPECT_EQ(sparse::default_backend(eckit_linalg).getString("backend"), "generic");

    const sparse::Backend backend_default      = sparse::Backend();
    const sparse::Backend backend_openmp       = sparse::backend::openmp();
    const sparse::Backend backend_eckit_linalg = sparse::backend::eckit_linalg();
    const sparse::Backend backend_hicsparse    = sparse::backend::hicsparse();
    EXPECT_EQ(backend_default.type(), hicsparse);
    EXPECT_EQ(backend_openmp.type(), openmp);
    EXPECT_EQ(backend_eckit_linalg.type(), eckit_linalg);
    EXPECT_EQ(backend_hicsparse.type(), hicsparse);

    EXPECT_EQ(std::string(backend_openmp), openmp);
    EXPECT_EQ(std::string(backend_eckit_linalg), eckit_linalg);
    EXPECT_EQ(std::string(backend_hicsparse), hicsparse);
}

//----------------------------------------------------------------------------------------------------------------------

CASE("SparseMatrix default constructor") {
    SparseMatrix A;
    EXPECT_EQ(A.rows(), 0);
    EXPECT_EQ(A.cols(), 0);
    EXPECT_EQ(A.nonZeros(), 0);
}

CASE("SparseMatrix copy constructor") {
    SparseMatrix A{3, 3, {{0, 0, 2.}, {0, 2, -3.}, {1, 1, 2.}, {2, 2, 2.}}};
    SparseMatrix B{A};
    EXPECT(A == B);
}

CASE("SparseMatrix assignment constructor") {
    SparseMatrix A{3, 3, {{0, 0, 2.}, {0, 2, -3.}, {1, 1, 2.}, {2, 2, 2.}}};
    auto B = A;
    EXPECT(A == B);
}

CASE("SparseMatrix assignment") {
    SparseMatrix A{3, 3, {{0, 0, 2.}, {0, 2, -3.}, {1, 1, 2.}, {2, 2, 2.}}};
    SparseMatrix B;
    B = A;
    EXPECT(A == B);
}

CASE("SparseMatrix triplet constructor") {
    SparseMatrix A{3, 3, {{0, 0, 2.}, {0, 2, -3.}, {1, 1, 2.}, {2, 2, 2.}}};
    EXPECT_EQ(A.rows(), 3);
    EXPECT_EQ(A.cols(), 3);
    EXPECT_EQ(A.nonZeros(), 4);
    auto value_v = A.value_view();
    auto outer_v = A.outer_view();
    auto inner_v = A.inner_view();

    std::vector<SparseMatrix::Scalar> value_exp{2., -3., 2., 2.};
    std::vector<SparseMatrix::Index> outer_exp{0, 2, 3, 4};
    std::vector<SparseMatrix::Index> inner_exp{0, 2, 1, 2};
    EXPECT(value_v == make_view(value_exp));
    EXPECT(outer_v == make_view(outer_exp));
    EXPECT(inner_v == make_view(inner_exp));
}

CASE("SparseMatrix triplet constructor 2") {
    SparseMatrix A{3, 3, {{0, 0, 2.}, {0, 2, 0.}, {1, 1, 2.}, {2, 2, 2.}}};
    EXPECT_EQ(A.rows(), 3);
    EXPECT_EQ(A.cols(), 3);
    EXPECT_EQ(A.nonZeros(), 3);
    auto value_v = A.value_view();
    auto outer_v = A.outer_view();
    auto inner_v = A.inner_view();

    std::vector<SparseMatrix::Scalar> value_exp{2., 2., 2.};
    std::vector<SparseMatrix::Index> outer_exp{0, 1, 2, 3};
    std::vector<SparseMatrix::Index> inner_exp{0, 1, 2};
    EXPECT(value_v == make_view(value_exp));
    EXPECT(outer_v == make_view(outer_exp));
    EXPECT(inner_v == make_view(inner_exp));
}

CASE("SparseMatrix swap") {
    SparseMatrix A_test{3, 3, {{0, 0, 2.}, {0, 2, 0.}, {1, 1, 2.}, {2, 2, 2.}}};
    SparseMatrix A{A_test};

    SparseMatrix B_test{1, 1, {{0, 0, 7.}}};
    SparseMatrix B{B_test};

    A.swap(B);
    EXPECT(A == B_test);
    EXPECT(B == A_test);
}

CASE("SparseMatrix transpose") {
    SparseMatrix A{3, 3, {{0, 0, 2.}, {0, 2, -3.}, {1, 1, 2.}, {2, 2, 2.}}};
    SparseMatrix AT{3, 3, {{0, 0, 2.}, {1, 1, 2.}, {2, 0, -3.}, {2, 2, 2.}}};
    A.transpose();
    EXPECT(A == AT);
}

CASE("SparseMatrix prune") {
    SparseMatrix A{3, 3, {{0, 0, 2.}, {0, 2, 0}, {1, 1, 2.}, {2, 2, 2.}}};
    SparseMatrix A_pruned{3, 3, {{0, 0, 2.}, {1, 1, 2.}, {2, 2, 2.}}};
    A.prune();
    EXPECT(A == A_pruned);
}

//----------------------------------------------------------------------------------------------------------------------

CASE("sparse_matrix vector multiply (spmv)") {
    // "square" matrix
    // A =  2  . -3
    //      .  2  .
    //      .  .  2
    // x = 1 2 3
    // y = 1 2 3
    SparseMatrix A{3, 3, {{0, 0, 2.}, {0, 2, -3.}, {1, 1, 2.}, {2, 2, 2.}}};

    for (std::string backend : {openmp, eckit_linalg}) {
        sparse::current_backend(backend);

        SECTION("test_identity [backend=" + sparse::current_backend().type() + "]") {
            {
                Vector y1(3);
                SparseMatrix B = Identity(3, 3);
                sparse_matrix_multiply(B, Vector{1., 2., 3.}, y1);
                expect_equal(y1, Vector{1., 2., 3.});
            }

            {
                SparseMatrix C = Identity(6, 3);
                Vector y2(6);
                sparse_matrix_multiply(C, Vector{1., 2., 3.}, y2);
                expect_equal(y2, Vector{1., 2., 3.});
                expect_equal(y2.data() + 3, Vector{0., 0., 0.}.data(), 3);
            }

            {
                SparseMatrix D = Identity(2, 3);
                Vector y3(2);
                sparse_matrix_multiply(D, Vector{1., 2., 3.}, y3);
                expect_equal(y3, Vector{1., 2., 3.});
            }
        }


        SECTION("eckit::Vector [backend=" + sparse::current_backend().type() + "]") {
            Vector y(3);
            sparse_matrix_multiply(A, Vector{1., 2., 3.}, y);
            expect_equal(y, Vector{-7., 4., 6.});
            // spmv of sparse matrix and vector of non-matching sizes should fail
            EXPECT_THROWS_AS(sparse_matrix_multiply(A, Vector(2), y), eckit::AssertionFailed);
        }

        SECTION("View of atlas::Array [backend=" + backend + "]") {
            ArrayVector<double> x(Vector{1., 2., 3.});
            ArrayVector<double> y(3);
            auto y_view = y.view();
            sparse_matrix_multiply(A, x.view(), y_view);
            expect_equal(y.view(), Vector{-7., 4., 6.});
            // sparse_matrix_multiply of sparse matrix and vector of non-matching sizes should fail
            {
                ArrayVector<double> x2(2);
                EXPECT_THROWS_AS(sparse_matrix_multiply(A, x2.view(), y_view), eckit::AssertionFailed);
            }
        }
    }
}

CASE("sparse-matrix vector multiply (spmv) [backend=hicsparse]") {
    // "square" matrix
    // A =  2  . -3
    //      .  2  .
    //      .  .  2
    // x = 1 2 3
    // y = 1 2 3
    sparse::current_backend(hicsparse);

    SparseMatrix A{3, 3, {{0, 0, 2.}, {0, 2, -3.}, {1, 1, 2.}, {2, 2, 2.}}};

    SECTION("View of atlas::Array [backend=hicsparse]") {
        ArrayVector<double> x(Vector{1., 2., 3.});
        ArrayVector<double> y(3);
        auto x_device_view = x.device_view();
        auto y_device_view = y.device_view();
        sparse_matrix_multiply(A, x_device_view, y_device_view);
        y.setHostNeedsUpdate(true);
        auto y_view = y.view();
        expect_equal(y.view(), Vector{-7., 4., 6.});
        // sparse_matrix_multiply of sparse matrix and vector of non-matching sizes should fail
        {
            ArrayVector<double> x2(2);
            auto x2_device_view = x2.device_view();
            EXPECT_THROWS_AS(sparse_matrix_multiply(A, x2_device_view, y_device_view), eckit::AssertionFailed);
        }
    }
}

CASE("sparse_matrix matrix multiply (spmm)") {
    // "square"
    // A =  2  . -3
    //      .  2  .
    //      .  .  2
    // x = 1 2 3
    // y = 1 2 3
    SparseMatrix A{3, 3, {{0, 0, 2.}, {0, 2, -3.}, {1, 1, 2.}, {2, 2, 2.}}};
    Matrix m{{1., 2.}, {3., 4.}, {5., 6.}};
    Matrix c_exp{{-13., -14.}, {6., 8.}, {10., 12.}};

    for (std::string backend : {openmp, eckit_linalg}) {
        sparse::current_backend(backend);

        SECTION("eckit::Matrix [backend=" + sparse::current_backend().type() + "]") {
            auto c = Matrix(3, 2);
            sparse_matrix_multiply(A, m, c);
            expect_equal(c, c_exp);
            // spmm of sparse matrix and matrix of non-matching sizes should fail
            EXPECT_THROWS_AS(sparse_matrix_multiply(A, Matrix(2, 2), c), eckit::AssertionFailed);
        }


        SECTION("View of eckit::Matrix [backend=" + backend + "]") {
            auto c  = Matrix(3, 2);
            auto mv = atlas::linalg::make_view(m);  // convert Matrix to array::LocalView<double,2>
            auto cv = atlas::linalg::make_view(c);
            sparse_matrix_multiply(A, mv, cv, Indexing::layout_right);
            expect_equal(c, c_exp);
        }

        SECTION("View of atlas::Array PointsRight [backend=" + sparse::current_backend().type() + "]") {
            ArrayMatrix<double, Indexing::layout_right> ma(m);
            ArrayMatrix<double, Indexing::layout_right> c(3, 2);
            auto c_view = c.view();
            sparse_matrix_multiply(A, ma.view(), c_view, Indexing::layout_right);
            expect_equal(c_view, c_exp);
        }
    }

    SECTION("sparse_matrix_multiply [backend=openmp]") {
        sparse::current_backend(eckit_linalg);  // expected to be ignored further
        auto backend = sparse::backend::openmp();
        ArrayMatrix<float> ma(m);
        ArrayMatrix<float> c(3, 2);
        auto c_view = c.view();
        sparse_matrix_multiply(A, ma.view(), c_view, backend);
        expect_equal(c_view, ArrayMatrix<float>(c_exp).view());
    }

    SECTION("SparseMatrixMultiply [backend=openmp] 1") {
        sparse::current_backend(eckit_linalg);  // expected to be ignored
        auto spmm = SparseMatrixMultiply{sparse::backend::openmp()};
        ArrayMatrix<float> ma(m);
        ArrayMatrix<float> c(3, 2);
        auto c_view = c.view();
        spmm(A, ma.view(), c_view);
        expect_equal(c_view, ArrayMatrix<float>(c_exp).view());
    }

    SECTION("SparseMatrixMultiply [backend=openmp] 2") {
        sparse::current_backend(eckit_linalg);  // expected to be ignored
        auto spmm = SparseMatrixMultiply{openmp};
        ArrayMatrix<float> ma(m);
        ArrayMatrix<float> c(3, 2);
        auto c_view = c.view();
        spmm(A, ma.view(), c_view);
        expect_equal(c_view, ArrayMatrix<float>(c_exp).view());
    }
}

CASE("sparse-matrix matrix multiply (spmm) [backend=hicsparse]") {
    // "square"
    // A =  2  . -3
    //      .  2  .
    //      .  .  2
    // x = 1 2 3
    // y = 1 2 3
    sparse::current_backend(hicsparse);

    SparseMatrix A{3, 3, {{0, 0, 2.}, {0, 2, -3.}, {1, 1, 2.}, {2, 2, 2.}}};
    Matrix m{{1., 2.}, {3., 4.}, {5., 6.}};
    Matrix c_exp{{-13., -14.}, {6., 8.}, {10., 12.}};

    SECTION("View of atlas::Array PointsRight [backend=hicsparse]") {
        ArrayMatrix<double, Indexing::layout_right> ma(m);
        ArrayMatrix<double, Indexing::layout_right> c(3, 2);
        auto ma_device_view = ma.device_view();
        auto c_device_view = c.device_view();
        sparse_matrix_multiply(A, ma_device_view, c_device_view, Indexing::layout_right);
        c.setHostNeedsUpdate(true);
        auto c_view = c.view();
        expect_equal(c_view, ArrayMatrix<double, Indexing::layout_right>(c_exp).view());
    }

    SECTION("sparse_matrix_multiply [backend=hicsparse]") {
        auto backend = sparse::backend::hicsparse();
        ArrayMatrix<double> ma(m);
        ArrayMatrix<double> c(3, 2);
        auto ma_device_view = ma.device_view();
        auto c_device_view = c.device_view();
        sparse_matrix_multiply(A, ma_device_view, c_device_view, backend);
        c.setHostNeedsUpdate(true);
        auto c_view = c.view();
        expect_equal(c_view, ArrayMatrix<double>(c_exp).view());
    }

    SECTION("SparseMatrixMultiply [backend=hicsparse] 1") {
        auto spmm = SparseMatrixMultiply{sparse::backend::hicsparse()};
        ArrayMatrix<double> ma(m);
        ArrayMatrix<double> c(3, 2);
        auto ma_device_view = ma.device_view();
        auto c_device_view = c.device_view();
        spmm(A, ma_device_view, c_device_view);
        c.setHostNeedsUpdate(true);
        auto c_view = c.view();
        expect_equal(c_view, ArrayMatrix<double>(c_exp).view());
    }

    SECTION("SparseMatrixMultiply [backend=hicsparse] 2") {
        auto spmm = SparseMatrixMultiply{hicsparse};
        ArrayMatrix<double> ma(m);
        ArrayMatrix<double> c(3, 2);
        auto ma_device_view = ma.device_view();
        auto c_device_view = c.device_view();
        spmm(A, ma_device_view, c_device_view);
        c.setHostNeedsUpdate(true);
        auto c_view = c.view();
        expect_equal(c_view, ArrayMatrix<double>(c_exp).view());
    }
}

//----------------------------------------------------------------------------------------------------------------------

}  // namespace test
}  // namespace atlas

int main(int argc, char** argv) {
    return atlas::test::run(argc, argv);
}
