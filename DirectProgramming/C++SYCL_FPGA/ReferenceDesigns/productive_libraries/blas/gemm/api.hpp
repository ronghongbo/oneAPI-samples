#pragma once

#include <cstdlib>
#include <sycl/sycl.hpp>
#include "oneapi/mkl.hpp"

// API of the reconfigurable matrix multiplication.The interface will be invoked by the GEMM implementation below.
#include "reconfigurable_matmul/api.hpp"

// Parameters of the reconfigurable matrix multiplication.
#include "reconfigurable_matmul/parameters.hpp"

// Data structures, etc. in Halide/T2SP
#include "Halide.h"
using namespace Halide;

namespace t2sp::blas::row_major {
// The API for GEMM. We choose the USM version of oneMKL DPC++ interface (https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-dpcpp/2023-0/gemm.html) with the
// restriction of standard data types (s, d, c, z) only. In this case, the matrices, alpha and beta all have the same data type according to the DPC++ interface. So we define our
// GEMM interface as a template with a single type T.
template<typename T>
sycl::event gemm(sycl::queue &queue,
                 oneapi::mkl::transpose transa,
                 oneapi::mkl::transpose transb,
                 std::int64_t m,
                 std::int64_t n,
                 std::int64_t k,
                 T alpha,
                 const T *a,
                 std::int64_t lda,
                 const T *b,
                 std::int64_t ldb,
                 T beta,
                 T *c,
                 std::int64_t ldc,
                 compute_mode mode = compute_mode::unset,
                 const std::vector<sycl::event> &dependencies = {}) {
    bool transpose_a = (transa == oneapi::mkl::transpose::T || transa == oneapi::mkl::transpose::C);
    bool transpose_b = (transb == oneapi::mkl::transpose::T || transb == oneapi::mkl::transpose::C);
    bool conjugate_a = (transa == oneapi::mkl::transpose::C);
    bool conjugate_b = (transb == oneapi::mkl::transpose::C);

    // Check parameters for constraints set by oneMKL DPC++ interface (https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-dpcpp/2023-0/gemm.html)
    _halide_user_assert(m >= 0 && n >= 0 && k >= 0) << "m = " << m << ", n = " << n << ", k = " << k;
    _halide_user_assert(a && b && c) << "a = " << (const void *)a << ", b = " << (const void *)b << ", c = " << (const void *)c;
    _halide_user_assert(!transpose_a && lda >= k || transpose_a && lda >= m)
        << std::boolalpha << "transa = " << transa << ", lda = " << lda << ", m = " << m << ", k = " << k;
    _halide_user_assert(!transpose_b && ldb >= n || transpose_b && ldb >= k)
        << std::boolalpha << "transb = " << transb << ", ldb = " << ldb << ", n = " << n << ", k = " << k;
    _halide_user_assert(ldc >= n) << "ldc = " << ldc << ", n = " << n;

    _halide_user_assert(constexpr (std::is_same_v<float, T>) || constexpr (std::is_same_v<double, T>) || constexpr (std::is_same_v<std::complex<float>, T>) ||
                        constexpr (std::is_same_v<std::complex<double>, T>)) << "Unsupported data type";

    // Constraints due to this implementation
    _halide_user_assert(mode == compute_mode::unset) << "The current implementation supports only compute_mode::unset";
    // TOREMOVE: These two constraints below should be checked by the reconfigurable matmul instead.
    _halide_user_assert(n % JJJ == 0) << "For performance reasons, the current implementation requires that n must be a multiple of " << JJJ
                              << "(the vectorized dimension for the output matrix), but n = " << n;
    _halide_user_assert(k % KKK == 0) << "For performance reasons, the current implementation requires that k must be a multiple of " << KKK
                              << "(the vectorized dimension for the input matrices), but k = " << k;

    using Halide::Runtime::Buffer;
    std::int64_t rows_of_A = !transpose_a ? m : k;
    std::int64_t cols_of_A = !transpose_a ? k : m;
    std::int64_t rows_of_B = !transpose_b ? k : n;
    std::int64_t cols_of_B = !transpose_b ? n : k;
    halide_dimension_t dim_a[]{{0, cols_of_A, 1}, {0, rows_of_A, lda}};
    Buffer<T> A_buffer{const_cast<T *>(a), 2, dim_a};
    halide_dimension_t dim_b[]{{0, cols_of_B, 1}, {0, rows_of_B, ldb}};
    Buffer<T> B_buffer{const_cast<T *>(b), 2, dim_b};
    halide_dimension_t dim_c[]{{0, n, 1}, {0, m, ldc}};
    Buffer<T> C_buffer{c, 2, dim_c};
    Buffer<T> Output_buffer{JJJ, JJ, II, III, (n + (JJJ * JJ - 1)) / (JJJ * JJ), (m + (III * II - 1)) / (III * II)};

    for (sycl::event e : dependencies) {
        e.wait();
    }

    bool FromSymmetricPosA = transpose_a;
    bool FromSymmetricPosB = transpose_b;
    bool FromSymmetricPosC = false;
    bool ConjugateA = conjugate_a;
    bool ConjugateB = conjugate_b;
    bool ConjugateC = false;
    bool HalfSpace = false;

    if constexpr (std::is_same_v<float, T>) {
        return t2sp::blas::row_major::smatmul(queue, FromSymmetricPosA, FromSymmetricPosB, FromSymmetricPosC,
                                              ConjugateA, ConjugateB, ConjugateC, HalfSpace, alpha, beta,
                                              A_buffer, B_buffer, C_buffer, Output_buffer);
    } else if constexpr (std::is_same_v<double, T>) {
        return t2sp::blas::row_major::dmatmul(queue, FromSymmetricPosA, FromSymmetricPosB, FromSymmetricPosC,
                                              ConjugateA, ConjugateB, ConjugateC, HalfSpace, alpha, beta,
                                              A_buffer, B_buffer, C_buffer, Output_buffer);
    } else if constexpr (std::is_same_v<std::complex<float>, T>) {
        return t2sp::blas::row_major::cmatmul(queue, FromSymmetricPosA, FromSymmetricPosB, FromSymmetricPosC,
                                              ConjugateA, ConjugateB, ConjugateC, HalfSpace, alpha, beta,
                                              A_buffer, B_buffer, C_buffer, Output_buffer);
    } else {
        return t2sp::blas::row_major::zmatmul(queue, FromSymmetricPosA, FromSymmetricPosB, FromSymmetricPosC,
                                              ConjugateA, ConjugateB, ConjugateC, HalfSpace, alpha, beta,
                                              A_buffer, B_buffer, C_buffer, Output_buffer);
    }
}

} // namespace t2sp::blas::row_major
