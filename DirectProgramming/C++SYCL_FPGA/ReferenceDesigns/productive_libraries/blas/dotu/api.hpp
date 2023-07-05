#pragma once

#include <cstdlib>
#include <sycl/sycl.hpp>
#include "oneapi/mkl.hpp"

// API of the reconfigurable dot. The interface will be invoked by the DOTU implementation below.
#include "../reconfigurable_dot/api.hpp"

// Data structures, etc. in Halide/T2SP
#include "Halide.h"
using namespace Halide;

namespace t2sp::blas::row_major {
// The API for DOTU. We choose the USM version of oneMKL DPC++ interface (https://oneapi-src.github.io/oneMKL/domains/blas/dotu.html) with the
// restriction of standard data types (c, z) only. In this case, the matrices, alpha and beta all have the same data type according to the DPC++ interface.
// So we define our DOTU interface as a template with a single type T.
template<typename T>
sycl::event dotu(sycl::queue &queue,
                 std::int64_t n,
                 const T *x,
                 std::int64_t incx,
                 const T *y,
                 std::int64_t incy,
                 T *result,
                 const std::vector<sycl::event> &dependencies = {}) {

    _halide_user_assert((std::is_same_v<std::complex<float>, T>) ||
                        (std::is_same_v<std::complex<double>, T>)) << "Unsupported data type";

    const auto KKK = get_systolic_array_dimensions<T>();

    // TOREMOVE: These two constraints below should be checked by the reconfigurable matmul instead.
    _halide_user_assert(n % KKK == 0) << "For performance reasons, the current implementation requires that n must be a multiple of " << KKK
                              << "(the vectorized dimension for the input vectors), but n = " << n;

    using Halide::Runtime::Buffer;
    halide_dimension_t dim_x[]{{0, n, std::abs(incx)}, {0, 1, 1}};
    halide_dimension_t dim_y[]{{0, n, std::abs(incy)}, {0, 1, 1}};
    halide_dimension_t dim_res[]{{0, 1, 1}};

    Buffer<T> X_buffer{const_cast<T *>(x), 2, dim_x};
    Buffer<T> Y_buffer{const_cast<T *>(y), 2, dim_y};
    Buffer<T> Res_buffer{result, 1, dim_res};

    for (sycl::event e : dependencies) {
        e.wait();
    }

    bool ConjugateX = false;
    bool SqrtRet = false;

    sycl::event done;

    if constexpr (std::is_same_v<std::complex<float>, T>) {
        done = t2sp::blas::row_major::cdot::cdot(queue, ConjugateX,
                                                 X_buffer, std::abs(static_cast<int>(incx)),
                                                 Y_buffer, std::abs(static_cast<int>(incy)), SqrtRet, Res_buffer);
    } else {
        done = t2sp::blas::row_major::zdot::zdot(queue, ConjugateX,
                                                 X_buffer, std::abs(static_cast<int>(incx)),
                                                 Y_buffer, std::abs(static_cast<int>(incy)), SqrtRet, Res_buffer);
    }
    return done;
}

} // namespace t2sp::blas::row_major