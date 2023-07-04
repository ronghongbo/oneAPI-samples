#pragma once

#include <cstdlib>
#include <sycl/sycl.hpp>
#include "oneapi/mkl.hpp"

// API of the reconfigurable matrix multiplication.The interface will be invoked by the SRYK implementation below.
#include "../reconfigurable_matmul/api.hpp"

// Data structures, etc. in Halide/T2SP
#include "Halide.h"
using namespace Halide;

namespace t2sp::blas::row_major {
// The API for SRYK. We choose the USM version of oneMKL DPC++ interface (https://oneapi-src.github.io/oneMKL/domains/blas/syrk.html).
template<typename T, typename T_REAL>
sycl::event herk(sycl::queue &queue,
                 oneapi::mkl::uplo upper_lower,
                 oneapi::mkl::transpose trans,
                 std::int64_t n,
                 std::int64_t k,
                 T_REAL alpha,
                 const T* a,
                 std::int64_t lda,
                 T_REAL beta,
                 T* c,
                 std::int64_t ldc,
                 const std::vector<sycl::event> &dependencies = {})
{
    // Check parameters for constraints set by oneMKL DPC++ interface
    _halide_user_assert(n >= 0 && k >= 0) << "n = " << n << ", k = " << k;
    _halide_user_assert(a && c) << "a = " << (const void *)a << ", c = " << (const void *)c;
    _halide_user_assert(lda > 0 && (trans == oneapi::mkl::transpose::N ? lda >= k : lda >= n)) << "lda = " << lda << ", k = " << k << ", n = " << n;
    _halide_user_assert(ldc > 0 && ldc >= n) << "ldc = " << ldc << ", n = " << n;

    _halide_user_assert(((std::is_same_v<std::complex<float>, T>)  && (std::is_same_v<float, T_REAL>)) ||
                        ((std::is_same_v<std::complex<double>, T>) && (std::is_same_v<double, T_REAL>))) << "Unsupported data type";

    const auto [KKK, JJJ, III, JJ, II, KK] = get_systolic_array_dimensions<T>();

    // TOREMOVE: These two constraints below should be checked by the reconfigurable matmul instead.
    _halide_user_assert(n % JJJ == 0) << "For performance reasons, the current implementation requires that n must be a multiple of " << JJJ
                              << "(the vectorized dimension for the output matrix), but n = " << n;
    _halide_user_assert((trans == oneapi::mkl::transpose::N ? k : n) % KKK == 0)
                              << "For performance reasons, the current implementation requires that the reduction dimension must be a multiple of " << KKK
                              << "(the vectorized dimension for the input matrix), but the reduction dimension = " << (trans == oneapi::mkl::transpose::N ? k : n);


    using Halide::Runtime::Buffer;
    halide_dimension_t dim_a[]{{0, trans == oneapi::mkl::transpose::N ? k : n, 1}, {0, trans == oneapi::mkl::transpose::N ? n : k, lda}};
    Buffer<T> A_buffer{const_cast<T *>(a), 2, dim_a};
    halide_dimension_t dim_c[]{{0, n, 1}, {0, n, ldc}};
    Buffer<T> C_buffer{c, 2, dim_c};
    Buffer<T> Output_buffer{JJJ, JJ, II, III, (n + (JJJ * JJ - 1)) / (JJJ * JJ), (n + (III * II - 1)) / (III * II)};

    for (sycl::event e : dependencies) {
        e.wait();
    }

    // Below we will use A_buffer as both matrix A and B to matmul, but in different ways (Upper_From_Upper_A/Lower_From_Lower_A are opposite to Upper_From_Upper_B/Lower_From_Lower_B).
    bool Upper_From_Upper_A = (trans == oneapi::mkl::transpose::N ? true : false);
    bool Upper_From_Upper_B = (trans == oneapi::mkl::transpose::N ? false : true);
    bool Upper_From_Upper_C = (upper_lower == oneapi::mkl::uplo::U ? true : false);
    bool Lower_From_Lower_A = (trans == oneapi::mkl::transpose::N ? true : false);
    bool Lower_From_Lower_B = (trans == oneapi::mkl::transpose::N ? false : true);
    bool Lower_From_Lower_C = (upper_lower == oneapi::mkl::uplo::U ? false : true);
    bool ConjugateTransposedA = (trans == oneapi::mkl::transpose::N ? false : true);
    bool ConjugateTransposedB = (trans == oneapi::mkl::transpose::N ? true : false);
    bool ConjugateTransposedC = false;
    bool HalfSpaceOut = true;

    sycl::event done;
    if constexpr (std::is_same_v<std::complex<float>, T>) {
        done = t2sp::blas::row_major::cmatmul::cmatmul(queue, Upper_From_Upper_A, Upper_From_Upper_B, Upper_From_Upper_C,
                                                              Lower_From_Lower_A, Lower_From_Lower_B, Lower_From_Lower_C,
                                                              ConjugateTransposedA, ConjugateTransposedB, ConjugateTransposedC,
                                                              HalfSpaceOut, alpha, beta,
                                                              A_buffer, A_buffer, C_buffer, Output_buffer);
    } else {
        done = t2sp::blas::row_major::zmatmul::zmatmul(queue, Upper_From_Upper_A, Upper_From_Upper_B, Upper_From_Upper_C,
                                                              Lower_From_Lower_A, Lower_From_Lower_B, Lower_From_Lower_C,
                                                              ConjugateTransposedA, ConjugateTransposedB, ConjugateTransposedC,
                                                              HalfSpaceOut, alpha, beta,
                                                              A_buffer, A_buffer, C_buffer, Output_buffer);
    }
    for (int i = 0; i < (n + (III * II - 1)) / (III * II); i++) {
        for (int j = 0; j < (n + (JJJ * JJ - 1)) / (JJJ * JJ); j++) {
            for (int ii = 0; ii < II; ii++) {
                for (int jj = 0; jj < JJ; jj++) {
                    for (int iii = 0; iii < III; iii++) {
                        for (int jjj = 0; jjj < JJJ; jjj++) {
                            int total_i = iii + III * ii + III * II * i;
                            int total_j = jjj + JJJ * jj + JJJ * JJ * j;
                            if (total_i < n && total_j < n) {
                                if ((upper_lower == oneapi::mkl::uplo::U && total_i <= total_j) || (upper_lower == oneapi::mkl::uplo::L && total_i >= total_j)) {
                                    c[total_j + total_i * ldc] = Output_buffer(jjj, jj, ii, iii, j, i);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    return done;
}

} // namespace t2sp::blas::row_major
