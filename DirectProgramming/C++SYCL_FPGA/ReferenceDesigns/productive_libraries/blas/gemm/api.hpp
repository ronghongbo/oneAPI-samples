#pragma once

#include <cstdlib>
#include <sycl/sycl.hpp>
#include "oneapi/mkl.hpp"

// API of the reconfigurable matrix multiplication.The interface will be invoked by the GEMM implementation below.
#include "../reconfigurable_matmul/api.hpp"

// Data structures, etc. in Halide/T2SP
#include "Halide.h"
using namespace Halide;



namespace t2sp::blas::row_major {
// Our interface shouldn't depend on these macro definitions,
// but since we're going to strip out extra zeros next,
// this requires the macros II/JJ/III/JJJ.
// When we improve our implementation in the future,
// we can remove this part.
// 
// As for why we don't use parameters.h,
// because the value it provides is determined by the macro,
// but we want to use the type to select the value.
template <typename T>
constexpr auto get_dimensions() {
#ifdef TINY
    return std::tuple{4, 4, 4, 4, 4};
#else
#ifdef S10
    constexpr bool run_on_s10 = true;
#else
    constexpr bool run_on_s10 = false;
#endif
    if constexpr (std::is_same_v<T, float>) {
        return run_on_s10 ? std::tuple{32, 32, 14, 16, 16} : std::tuple{32, 32, 10, 8, 16};
    } else if constexpr (std::is_same_v<T, double>) {
        return run_on_s10 ? std::tuple{32, 32, 8, 4, 8} : std::tuple{32, 32, 8, 4, 8};
    } else if constexpr (std::is_same_v<T, std::complex<float>>) {
        return run_on_s10 ? std::tuple{32, 32, 14, 16, 16} : std::tuple{32, 32, 10, 4, 8};
    } else if constexpr (std::is_same_v<T, std::complex<double>>) {
        return run_on_s10 ? std::tuple{32, 32, 4, 6, 4} : std::tuple{32, 32, 4, 4, 4};
    } else {
        return std::tuple{4, 4, 4, 4, 4};
    }
#endif
    }
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
                 const std::vector<sycl::event> &dependencies = {}) {
    bool transpose_a = (transa == oneapi::mkl::transpose::T || transa == oneapi::mkl::transpose::C);
    bool transpose_b = (transb == oneapi::mkl::transpose::T || transb == oneapi::mkl::transpose::C);
    bool conjugate_a = (transa == oneapi::mkl::transpose::C);
    bool conjugate_b = (transb == oneapi::mkl::transpose::C);

    // Check parameters for constraints set by oneMKL DPC++ interface (https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-dpcpp/2023-0/gemm.html)
    _halide_user_assert(m >= 0 && n >= 0 && k >= 0) << "m = " << m << ", n = " << n << ", k = " << k;
    _halide_user_assert(a && b && c) << "a = " << (const void *)a << ", b = " << (const void *)b << ", c = " << (const void *)c;
    _halide_user_assert(!transpose_a && lda >= k || transpose_a && lda >= m)
        << std::boolalpha << "transa = " << transpose_a << ", lda = " << lda << ", m = " << m << ", k = " << k;
    _halide_user_assert(!transpose_b && ldb >= n || transpose_b && ldb >= k)
        << std::boolalpha << "transb = " << transpose_b<< ", ldb = " << ldb << ", n = " << n << ", k = " << k;
    _halide_user_assert(ldc >= n) << "ldc = " << ldc << ", n = " << n;

    _halide_user_assert((std::is_same_v<float, T>) ||
                        (std::is_same_v<double, T>) ||
                        (std::is_same_v<std::complex<float>, T>) ||
                        (std::is_same_v<std::complex<double>, T>)) << "Unsupported data type";

    const auto [II, JJ, III, JJJ, KKK] = get_dimensions<T>();

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
    halide_dimension_t dim_a[]{{0, static_cast<int32_t>(cols_of_A), 1}, {0, static_cast<int32_t>(rows_of_A), static_cast<int32_t>(lda)}};
    Buffer<T> A_buffer{const_cast<T *>(a), 2, dim_a};
    halide_dimension_t dim_b[]{{0, static_cast<int32_t>(cols_of_B), 1}, {0, static_cast<int32_t>(rows_of_B), static_cast<int32_t>(ldb)}};
    Buffer<T> B_buffer{const_cast<T *>(b), 2, dim_b};
    halide_dimension_t dim_c[]{{0, static_cast<int32_t>(n), 1}, {0, static_cast<int32_t>(m), static_cast<int32_t>(ldc)}};
    Buffer<T> C_buffer{c, 2, dim_c};
    Buffer<T> Output_buffer{JJJ, JJ, II, III, (static_cast<int32_t>(n) + (JJJ * JJ - 1)) / (JJJ * JJ), (static_cast<int32_t>(m) + (III * II - 1)) / (III * II)};

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
        // Temporarily removed as we cannot now generate code for complex types
        //return t2sp::blas::row_major::cmatmul(queue, FromSymmetricPosA, FromSymmetricPosB, FromSymmetricPosC,
        //                                      ConjugateA, ConjugateB, ConjugateC, HalfSpace, alpha, beta,
        //                                      A_buffer, B_buffer, C_buffer, Output_buffer);
    } else {
        //return t2sp::blas::row_major::zmatmul(queue, FromSymmetricPosA, FromSymmetricPosB, FromSymmetricPosC,
        //                                      ConjugateA, ConjugateB, ConjugateC, HalfSpace, alpha, beta,
        //                                      A_buffer, B_buffer, C_buffer, Output_buffer);
    }
}

} // namespace t2sp::blas::row_major
