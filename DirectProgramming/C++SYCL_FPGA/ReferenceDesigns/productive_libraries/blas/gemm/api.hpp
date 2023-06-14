#pragma once

#include <cstdlib>
#include <sycl/sycl.hpp>
#include "oneapi/mkl.hpp"

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
                 const std::vector<sycl::event> &dependencies = {});
}
